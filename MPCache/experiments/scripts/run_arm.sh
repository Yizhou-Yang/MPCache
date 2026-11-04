#!/usr/bin/env bash
# One arm of any benchmark, through the SDK runner. No per-benchmark shell.
#
#   bash run_arm.sh cogvideox /weights/CogVideoX-2b prompts.txt 49 50 20 out/ours
#
# Arguments: backend model prompts frames steps budget out [schedule]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SDK=$(cd "$HERE/../../sdk" && pwd)
BACKEND=${1:?backend}; MODEL=${2:?model}; PROMPTS=${3:?prompts}
FRAMES=${4:?frames}; STEPS=${5:?steps}; BUDGET=${6:?budget}; OUT=${7:?out}
SCHED=${8:-}
LIMIT=${LIMIT:-100}
LEDGER=${LEDGER:-$(dirname "$OUT")/metrics.jsonl}

SCHED_ARG="--budget $BUDGET"
[ -n "$SCHED" ] && SCHED_ARG="--schedule $SCHED"

PYTHONPATH=$SDK python -m mpcache.runner \
  --backend "$BACKEND" --model "$MODEL" --prompts "$PROMPTS" \
  --frames "$FRAMES" --steps "$STEPS" $SCHED_ARG \
  --limit "$LIMIT" --out "$OUT" --ledger "$LEDGER" --tag "$(basename "$OUT")"
