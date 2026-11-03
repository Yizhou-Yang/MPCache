#!/usr/bin/env bash
# Produce the two settings the main table needs, at the competitors' spec:
#   Cog 49f 720x480 100 prompts 50 steps   -> orig, cfg, ours
#   Wan 81f 720x480 100 prompts 50 steps   -> orig, cfg, ours [, tea, taylor]
#
# Usage:  RA_REPO=/path/to/RA-CFGCache bash run_freeze.sh [cog|wan|both]
# Cost on 4 GPUs at 49f: ~56 min/arm; at 81f: ~135 min/arm. Use a dedicated machine.
set -euo pipefail
REPO_RA=${RA_REPO:?set RA_REPO to the RA-CFGCache checkout}
COG=${COG_MODEL:-$HOME/weights/CogVideoX-2b}
WAN=${WAN_MODEL:-$HOME/weights/Wan2.1-T2V-1.3B}
OUT=${OUT:-$HOME/results/freeze}
WHAT=${1:-both}
S49="0,1,2,3,4,5,6,7,8,9,10,11,13,15,18,22,27,33,40,48"
S81="0,1,2,3,4,5,6,7,8,9,11,12,14,17,20,24,28,33,38,43,49"
cd "$REPO_RA"
export PATH=${CONDA_PREFIX:-/opt/conda}/bin:$PATH

cog() {
  bash RUN/demo_cogvideox.sh --mode original --gpus 0,1,2,3 --limit 100 --num_frames 49 \
       --model_path "$COG" --output_dir "$OUT/cog49_100/orig"
  MPC_SCHED="$S49" bash RUN/demo_cogvideox.sh --mode TeaCache --gpus 0,1,2,3 --limit 100 --num_frames 49 \
       --model_path "$COG" --output_dir "$OUT/cog49_100/ours"
  bash RUN/demo_cogvideox.sh --mode CFGCache --gpus 0,1,2,3 --limit 100 --num_frames 49 \
       --model_path "$COG" --output_dir "$OUT/cog49_100/cfg"
}
wan() {
  bash RUN/demo_wan.sh --mode original --gpus 0,1,2,3 --limit 100 \
       --model_path "$WAN" --output_dir "$OUT/wan81_100/orig"
  MPC_SCHEDW="$S81" bash RUN/demo_wan.sh --mode CFGCache --gpus 0,1,2,3 --limit 100 \
       --model_path "$WAN" --output_dir "$OUT/wan81_100/ours"
  bash RUN/demo_wan.sh --mode CFGCache --gpus 0,1,2,3 --limit 100 \
       --model_path "$WAN" --output_dir "$OUT/wan81_100/cfg"
  if [ "${WITH_SLOW_BASELINES:-0}" = "1" ]; then
    bash RUN/demo_wan.sh --mode TeaCache --gpus 0,1,2,3 --limit 100 --model_path "$WAN" --output_dir "$OUT/wan81_100/tea"
    bash RUN/demo_wan.sh --mode Taylor   --gpus 0,1,2,3 --limit 100 --model_path "$WAN" --output_dir "$OUT/wan81_100/taylor"
  fi
}
case "$WHAT" in
  cog) cog ;;
  wan) wan ;;
  both) cog; wan ;;
  *) echo "usage: $0 [cog|wan|both]"; exit 2 ;;
esac

echo "=== evaluation (LPIPS-VGG / SSIM / PSNR) ==="
HERE=$(cd "$(dirname "$0")" && pwd)
for arm in cfg ours; do [ -d "$OUT/cog49_100/$arm" ] && python "$HERE/eval_lpips_vgg.py" "$OUT/cog49_100/orig" "$OUT/cog49_100/$arm"; done
for arm in cfg ours tea taylor; do [ -d "$OUT/wan81_100/$arm" ] && python "$HERE/eval_lpips_vgg.py" "$OUT/wan81_100/orig" "$OUT/wan81_100/$arm"; done
