#!/usr/bin/env bash
# Build the two qualitative figures. Each figure is generated inside ONE harness, because
# the two harnesses do not share absolute numbers.
#
#   group A (RA-CFGCache harness): Original | TeaCache | RA-CFGCache | Ours(head schedule)
#   group B (plain diffusers):     Original | reuse | damped 0.5 | scaled by gap
#
# usage: RA_REPO=/path/to/RA-CFGCache COG=/weights/CogVideoX-2b bash make_qualitative.sh
set -euo pipefail
RA_REPO=${RA_REPO:?set RA_REPO}
COG=${COG:?set COG}
OUT=${OUT:-$PWD/qual}
N=${N:-3}
FR=${FR:-49}
S49="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,18,22,28,35,42,49"
PY=${PYTHON:-python}
HERE=$(cd "$(dirname "$0")" && pwd)

mkdir -p "$OUT"

# ---- group A: one harness, four policies -------------------------------------
cd "$RA_REPO"
for spec in "0 orig original" "1 tea TeaCache" "2 cfg CFGCache"; do
  set -- $spec
  CUDA_VISIBLE_DEVICES=$1 bash RUN/demo_cogvideox.sh --mode "$3" --gpus "$1" --limit "$N" \
    --num_frames "$FR" --model_path "$COG" --output_dir "$OUT/A/$2" \
    > "$OUT/A_$2.log" 2>&1 &
done
CUDA_VISIBLE_DEVICES=3 MPC_SCHED="$S49" bash RUN/demo_cogvideox.sh --mode TeaCache --gpus 3 --limit "$N" \
  --num_frames "$FR" --model_path "$COG" --output_dir "$OUT/A/ours" > "$OUT/A_ours.log" 2>&1 &
wait

"$PY" "$HERE/compose_qualitative.py" --out "$OUT/qual_A.pdf" --rows "$N" \
  --arm Original="$OUT/A/orig" --arm TeaCache="$OUT/A/tea" \
  --arm RA-CFGCache="$OUT/A/cfg" --arm Ours="$OUT/A/ours"

# ---- group B: our harness, four reuse policies --------------------------------
for spec in "0 full full 0" "1 reuse latest 0" "2 e05 extrap 0.5" "3 scaled scaled 0"; do
  set -- $spec
  CUDA_VISIBLE_DEVICES=$1 "$PY" "$HERE/reuse_policy.py" --model "$COG" \
    --prompts "$RA_REPO/resources/prompts/prompt_video.txt" --frames "$FR" --policy "$2" \
    --alpha "$4" --limit "$N" --out "$OUT/B/$2" > "$OUT/B_$2.log" 2>&1 &
done
wait

"$PY" "$HERE/compose_qualitative.py" --out "$OUT/qual_B.pdf" --rows "$N" \
  --arm Original="$OUT/B/full" --arm "reuse"="$OUT/B/latest" \
  --arm "damped 0.5"="$OUT/B/e05" --arm "scaled by gap"="$OUT/B/scaled"
echo "done: $OUT/qual_A.pdf and $OUT/qual_B.pdf"
