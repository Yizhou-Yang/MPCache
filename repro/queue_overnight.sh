#!/usr/bin/env bash
# Overnight supervisor (2026-10-15): wait for G0a, then run the reproduction in an order
# that yields *complete* small-N benchmark tables first, then continues to the full paper
# protocol. Every chunk is resumable (run_repro.sh skips completed method/chunk pairs), and
# after each chunk we refresh repro/STATUS.md so the morning push is just read + push.
#
# Order: G0a (already running) -> Wan smoke -> Cog[0,8) -> Wan[0,8) -> Cog[8,16) -> Wan[8,16)
#        -> Cog[16,32) -> Wan[16,32) -> Cog[32,100) -> Wan[32,100)
set -u
cd /root/viewdit
export PATH=/root/miniconda3/bin:$PATH

log() { echo "[overnight $(date +%F_%T)] $*"; }

# 1) wait for G0a (its own process writes experiments/video/g0a_costmodel.log and exits)
log "waiting for G0a to finish..."
while ! grep -q '^\[done\]' experiments/video/g0a_costmodel.log 2>/dev/null; do sleep 120; done
log "G0a done"

# 2) Wan smoke (de-risks the Wan pipeline before the overnight marathon)
if [ ! -f logs/REPRO_SMOKE_WAN_DONE ]; then
  log "wan smoke"
  bash repro/run_repro.sh wan 0 1 vanilla50 teacache >> logs/repro_smoke_wan.log 2>&1
  echo "wan smoke rc=$? $(date +%F_%T)" >> logs/repro_smoke_wan.log
  touch logs/REPRO_SMOKE_WAN_DONE
fi

# 3) benchmark chunks, small first
CHUNKS=${CHUNKS:-"cogvideox:0:16 wan:0:16 cogvideox:16:32 wan:16:32 cogvideox:32:100 wan:32:100"}
for c in $CHUNKS; do
  IFS=: read -r model s e <<< "$c"
  log "chunk start $model [$s,$e)"
  bash repro/run_repro.sh "$model" "$s" "$e" >> "logs/repro_full_${model}.log" 2>&1
  rc=$?
  touch "logs/REPRO_DONE_${model}_${s}_${e}"
  log "chunk done $model [$s,$e) rc=$rc"
  # refresh the status table after every completed chunk
  for m in cogvideox wan; do
    n=$(ls results/repro/$m/vanilla50/*/videos/*.mp4 2>/dev/null | wc -l)
    if [ "$n" -ge 1 ]; then
      log "aggregate $m (n=$n)"
      ( cd repro && python aggregate.py --model "$m" --n "$n" --skip_eval >> /root/viewdit/logs/repro_aggregate.log 2>&1 ) || log "aggregate $m failed"
    fi
  done
  bash repro/push_overnight.sh >> logs/repro_push.log 2>&1 || log "push failed"
done
log "ALL CHUNKS DONE"
bash repro/push_overnight.sh >> logs/repro_push.log 2>&1
