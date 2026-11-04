# The path that produced the paper's numbers: the released RA-CFGCache pipeline with the
# schedule injected through MPC_SCHED. Use `run_arm.sh` for the SDK runner instead.
#!/usr/bin/env bash
# Regenerate one arm of the main table. Usage: run_main_table.sh <mode> <frames> <limit> <outdir> [schedule]
set -euo pipefail
MODE=${1:?mode}; FRAMES=${2:?frames}; LIMIT=${3:?limit}; OUT=${4:?outdir}; SCHED=${5:-}
REPO_RA=${RA_REPO:-$HOME/RA-CFGCache}
MODEL=${MODEL:-$HOME/weights/CogVideoX-2b}
cd "$REPO_RA"
if [ -n "$SCHED" ]; then export MPC_SCHED="$SCHED"; else unset MPC_SCHED; fi
bash RUN/demo_cogvideox.sh --mode "$MODE" --gpus 0,1,2,3 --limit "$LIMIT" \
  --num_frames "$FRAMES" --model_path "$MODEL" --output_dir "$OUT"
