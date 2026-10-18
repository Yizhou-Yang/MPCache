#!/usr/bin/env bash
# Overnight second-line supervisor for the V100 reproduction.
#
# The primary queue (queue_overnight.sh) runs one fixed CHUNKS list and then exits, whether or
# not every chunk finished.  This watchdog waits for it to release its lock, then re-queues only
# the rows that are still missing ids, aggregates, and repeats a bounded number of rounds.
# It never touches Wan TaylorSeer (deferred to H200: 90-135 GiB of cache) and never deletes a
# video, so it is safe to leave running overnight.
#
#   setsid nohup bash repro/overnight_watchdog.sh >> logs/watchdog.out 2>&1 < /dev/null &
#   tail -f logs/watchdog.out
#
# Tunables: ROUNDS (default 3), EXCLUDE (default wan:taylorseer), RACFG_PROFILE (v100).
set -uo pipefail
VIEWDIT=${VIEWDIT:-/root/viewdit}
cd "$VIEWDIT" || exit 2
export PATH=/root/miniconda3/envs/racfg/bin:/root/miniconda3/bin:$PATH
# The system python3 on this host is 3.6; ledger.py/plan_missing.py need 3.9+.
PY=${PY:-/root/miniconda3/envs/racfg/bin/python}
export RACFG_PROFILE=${RACFG_PROFILE:-v100}
export CPU_OFFLOAD=0 ALLOW_GIT_PUSH=0 PYTHONUNBUFFERED=1
mkdir -p logs
log() { echo "[watchdog $(date +%F_%T)] $*"; }

if [ -f logs/WATCHDOG_DONE ]; then
  log "logs/WATCHDOG_DONE exists; nothing to do"
  exit 0
fi
exec 8>logs/repro_watchdog.lock
flock -n 8 || { log "another watchdog already owns logs/repro_watchdog.lock"; exit 1; }

ROUNDS=${ROUNDS:-3}
EXCLUDE=${EXCLUDE:-wan:taylorseer}
log "start profile=$RACFG_PROFILE rounds=$ROUNDS exclude=$EXCLUDE py=$("$PY" -c 'import sys;print(sys.version.split()[0])')"

# 1) Wait for the primary supervisor.  flock returns 0 only when nobody holds the lock.
log "waiting for the primary supervisor to release logs/repro_supervisor.lock"
n=0
until flock -n logs/repro_supervisor.lock -c true 2>/dev/null; do
  n=$((n + 1))
  [ $((n % 10)) -eq 0 ] && log "still waiting (${n} checks)"
  sleep 120
done
log "primary supervisor is gone"

# 2) A killed supervisor can leave an orphaned sampling child; never stack two GPU jobs.
while pgrep -f 'repro/run_repro\.sh' >/dev/null 2>&1 \
   || pgrep -f 'models\.(wan|cogvideox)\.sample' >/dev/null 2>&1; do
  log "a GPU job is still alive after the supervisor; waiting before touching the queue"
  sleep 60
done

# 3) Missing-row rounds.
for r in $(seq 1 "$ROUNDS"); do
  find results/repro results/repro_h20 -name 'video_*.mp4' -printf '%p\t%s\n' 2>/dev/null \
    > logs/ledger_manifest.tsv
  "$PY" repro/ledger.py logs/ledger_manifest.tsv --n 100 --json logs/ledger.json \
    --md logs/ledger.md > logs/ledger.txt 2>&1
  log "ledger: $(tail -n 1 logs/ledger.txt)"

  PLAN=$(python3 repro/plan_missing.py logs/ledger.json --n 100 --exclude "$EXCLUDE")
  if [ -z "$PLAN" ]; then
    log "no incomplete rows left (excluding $EXCLUDE)"
    break
  fi
  log "round $r plan:"
  printf '    %s\n' "$PLAN"

  while IFS= read -r spec; do
    [ -z "$spec" ] && continue
    IFS=: read -r model s e methods <<< "$spec"
    log "run $model [$s,$e) methods=$methods"
    bash repro/run_repro.sh "$model" "$s" "$e" ${methods//,/ } \
      >> "logs/watchdog_${model}.log" 2>&1 8>&- 9>&-
    rc=$?
    if [ "$rc" = 0 ]; then log "ok $spec"; else log "incomplete $spec rc=$rc"; fi
  done <<< "$PLAN"

  log "aggregate both models at n=100"
  "$PY" repro/aggregate.py --model cogvideox --n 100 --skip_eval \
    >> logs/watchdog_aggregate.log 2>&1 8>&- 9>&- || log "aggregate cogvideox failed"
  "$PY" repro/aggregate.py --model wan --n 100 --skip_eval \
    >> logs/watchdog_aggregate.log 2>&1 8>&- 9>&- || log "aggregate wan failed"
done

# 4) Final ledger and completion marker.
find results/repro results/repro_h20 -name 'video_*.mp4' -printf '%p\t%s\n' 2>/dev/null \
  > logs/ledger_manifest.tsv
"$PY" repro/ledger.py logs/ledger_manifest.tsv --n 100 --json logs/ledger_final.json \
  --md logs/ledger_final.md > logs/ledger_final.txt 2>&1
cp logs/ledger.txt logs/ledger_before_final.txt 2>/dev/null || true
touch logs/WATCHDOG_DONE
log "done: $(tail -n 1 logs/ledger_final.txt)"
