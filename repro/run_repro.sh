#!/usr/bin/env bash
# Reproduce RA-CFGCache Table 1 (TeaCache / TaylorSeer / RA-CFGCache, plus Vanilla T=50/T=25)
# on CogVideoX-2B and Wan2.1-T2V-1.3B, using RA-CFGCache's own harness at commit e17d6a7.
#
#   bash repro/run_repro.sh <cogvideox|wan> <start> <end> [method ...]
#   e.g. bash repro/run_repro.sh cogvideox 0 32                 # all methods, prompts [0,32)
#        bash repro/run_repro.sh wan 0 1 vanilla50 teacache     # smoke test
#
# Protocol (paper Sec. 4 + App. B.1): 100 VBench prompts (resources/prompts/prompt_video.txt),
# seed = 0 + global prompt index, 50 steps, negative prompt "animation", batch 1, CFG branches sequential.
# Prompts are run in [start,end) chunks with the true global start_index, so chunks are an exact
# prefix/suffix of the paper's 100-prompt run (same prompt -> same seed -> same initial noise).
set -uo pipefail

MODEL=$1; START=$2; END=$3; shift 3
METHODS=("$@")
[ ${#METHODS[@]} -eq 0 ] && METHODS=(vanilla50 vanilla25 teacache taylorseer racfg_slow racfg_fast)

VIEWDIT=${VIEWDIT:-/root/viewdit}
RACFG=$VIEWDIT/repro/upstream/RA-CFGCache
OUT_ROOT=${OUT_ROOT:-$VIEWDIT/results/repro}
PROMPTS=$RACFG/resources/prompts/prompt_video.txt

source /root/miniconda3/etc/profile.d/conda.sh
conda activate racfg
export TORCHDYNAMO_DISABLE=1 TORCHINDUCTOR_DISABLE=1 PYTORCH_DISABLE_CUDA_COMPILE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# V100 32GB (vs H200 141GB): TaylorSeer's per-layer derivative cache OOMs with the text encoder
# resident, so model CPU offload is ON for every method (same setting for all rows -> fair).
OFFLOAD=(); [ "${CPU_OFFLOAD:-1}" = "1" ] && OFFLOAD=(--cpu_offload)

case $MODEL in
  cogvideox)
    MP=$VIEWDIT/weights/CogVideoX-2b
    export RACFG_DTYPE=${RACFG_DTYPE:-float16}      # CogVideoX-2B is an fp16 checkpoint
    COMMON=(--model_name cogvideox --model_path "$MP" --width 720 --height 480 --num_frames 49 --fps 8
            --guidance_scale 6.0 --proxy_tables_path "$RACFG/calibration/cogvideox_rho_cfg6.npz")
    TEA=0.2; SLOW=0.6; FAST=1.4 ;;
  wan)
    MP=$VIEWDIT/weights/Wan2.1-T2V-1.3B-Diffusers
    export RACFG_DTYPE=${RACFG_DTYPE:-float16}      # V100: no bf16; verified in smoke test
    COMMON=(--model_name wan --model_path "$MP" --width 832 --height 480 --num_frames 81 --fps 16
            --guidance_scale 5.0 --scheduler unipc --flow_shift 3.0
            --proxy_tables_path "$RACFG/calibration/wan_rho_cfg5.npz")
    TEA=0.15; SLOW=0.4; FAST=0.6 ;;
  *) echo "unknown model $MODEL"; exit 1 ;;
esac

CHUNK=$(printf "p%03d-%03d" "$START" "$END")
mkdir -p "$OUT_ROOT/$MODEL/_prompts" "$OUT_ROOT/$MODEL/_logs"
PF=$OUT_ROOT/$MODEL/_prompts/$CHUNK.txt
awk 'NF' "$PROMPTS" | sed -n "$((START+1)),$((END))p" > "$PF"

run_one() {   # name mode steps thresh
  local name=$1 mode=$2 steps=$3 thr=$4
  local od=$OUT_ROOT/$MODEL/$name/$CHUNK
  local log=$OUT_ROOT/$MODEL/_logs/${name}_$CHUNK.log
  local n_need=$((END-START))
  local n_have=$(ls "$od/videos" 2>/dev/null | grep -c '\.mp4$')
  if [ "$n_have" -ge "$n_need" ]; then echo "[skip] $name $CHUNK ($n_have videos)"; return 0; fi
  echo "[run] $MODEL $name $CHUNK mode=$mode steps=$steps thr=$thr dtype=$RACFG_DTYPE offload=${#OFFLOAD[@]} $(date +%F_%T)"
  ( cd "$RACFG" && python -m models.$MODEL.sample "${COMMON[@]}" \
      --taylor_method "$mode" --num_steps "$steps" --rel_l1_thresh "$thr" \
      --interval 3 --max_order 1 --first_enhance 3 --hicache_scale 0.5 \
      --batch_size 1 --num_videos_per_prompt 1 --seed 0 --negative_prompt animation \
      --prompt_file "$PF" --start_index "$START" --output_dir "$od" --no_save_eval_frames "${OFFLOAD[@]}" ) > "$log" 2>&1
  local rc=$?
  echo "[done] $name $CHUNK rc=$rc $(grep -E 'Avg sec / video sampling|Avg sampling sec/video' "$log" | tail -1)"
  return $rc
}

for m in "${METHODS[@]}"; do
  case $m in
    vanilla50)  run_one vanilla50  original 50 0 ;;
    vanilla25)  run_one vanilla25  original 25 0 ;;
    teacache)   run_one teacache   TeaCache 50 "$TEA" ;;
    taylorseer) run_one taylorseer Taylor   50 0 ;;      # N=3, O=1 (App. B.1)
    racfg_slow) run_one racfg_slow CFGCache 50 "$SLOW" ;;
    racfg_fast) run_one racfg_fast CFGCache 50 "$FAST" ;;
    *) echo "unknown method $m" ;;
  esac
done
