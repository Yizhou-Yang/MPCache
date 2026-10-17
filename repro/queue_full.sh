#!/usr/bin/env bash
# Full reproduction queue. Waits for G0a, then runs chunks in priority order.
# Each chunk is resumable: run_repro.sh skips (method, chunk) pairs that already have all videos.
set -uo pipefail
VIEWDIT=${VIEWDIT:-/root/viewdit}
cd "$VIEWDIT" || exit 2
failed=0
while [ ! -f logs/G0A_DONE ]; do sleep 120; done
CHUNKS=${CHUNKS:-"cogvideox:0:32 wan:0:32 cogvideox:32:100 wan:32:100"}
for c in $CHUNKS; do
  IFS=: read -r model s e <<< "$c"
  if bash repro/run_repro.sh "$model" "$s" "$e" >> "logs/repro_full_${model}.log" 2>&1; then
    touch "logs/REPRO_DONE_${model}_${s}_${e}"
  else
    failed=1
  fi
done
exit "$failed"
