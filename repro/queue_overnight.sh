#!/usr/bin/env bash
# GPU-only resumable reproduction supervisor. Existing videos are never deleted.
# WAIT_PID lets a running sampling process finish before this supervisor takes over.
set -uo pipefail
cd /root/viewdit
export PATH=/root/miniconda3/envs/racfg/bin:/root/miniconda3/bin:$PATH
export CPU_OFFLOAD=0 PYTHONUNBUFFERED=1
log() { echo "[repro $(date +%F_%T)] $*"; }
exec 9>logs/repro_supervisor.lock
flock -n 9 || { log "another supervisor owns the lock"; exit 1; }
if [ -n "${WAIT_PID:-}" ]; then
  log "waiting for existing GPU process $WAIT_PID; preserving its output"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30 9>&-; done
fi
# G0a failure is a research result, not a reason to stop competitor reproduction.
PROMPTS=repro/upstream/RA-CFGCache/resources/prompts/prompt_video.txt
N_PROMPTS=$(python -c 'from pathlib import Path; import sys; print(sum(bool(x.strip()) for x in Path(sys.argv[1]).read_text().splitlines()))' "$PROMPTS")
log "available prompt IDs [0,$N_PROMPTS); verified nonempty count, no invented prompts"
CHUNKS=${CHUNKS:-"cogvideox:0:16 wan:0:16 cogvideox:16:32 wan:16:32 cogvideox:32:$N_PROMPTS wan:32:$N_PROMPTS"}
failed=0
for c in $CHUNKS; do
  IFS=: read -r model s e <<< "$c"
  if [ "$s" -ge "$e" ]; then continue; fi
  log "chunk start $model [$s,$e), GPU-only"
  bash repro/run_repro.sh "$model" "$s" "$e" >> "logs/repro_full_${model}.log" 2>&1
  rc=$?
  if [ "$rc" = "0" ]; then
    touch "logs/REPRO_DONE_${model}_${s}_${e}"
    log "chunk complete $model [$s,$e)"
  else
    failed=1
    log "chunk incomplete $model [$s,$e) rc=$rc; no success marker, no CPU fallback"
  fi
  log "evaluate $model [0,$e) with exact common IDs and source fingerprints"
  python repro/aggregate.py --model "$model" --n "$e" --skip_eval >> logs/repro_aggregate.log 2>&1 || {
    failed=1; log "evaluation incomplete; see per-method errors, continuing remaining GPU jobs"
  }
  bash repro/push_overnight.sh >> logs/repro_push.log 2>&1 || log "GitHub sync failed; retry next checkpoint"
done
if [ "$failed" = "0" ]; then
  log "all scheduled chunks completed; prompt-list discrepancy remains documented"
else
  log "queue exhausted with failures; not claiming full reproduction completion"
fi
bash repro/push_overnight.sh >> logs/repro_push.log 2>&1 || failed=1
exit "$failed"
