#!/usr/bin/env bash
# Final aggregator: waits until NO sampling process remains (including the extra
# racfg_fast workers started later on GPU0/1), then aggregates both models and pushes.
# Safe to run alongside autostart_h20_queue.sh: it re-aggregates after that exits.
set -uo pipefail

cd /root/viewdit || exit 2
export VIEWDIT=/root/viewdit
export RACFG_PROFILE=h20
export RACFG_DTYPE=bfloat16
export CPU_OFFLOAD=0
export ALLOW_GIT_PUSH=0
export PATH=/root/miniconda3/envs/racfg/bin:/root/miniconda3/bin:$PATH
LOG=/root/viewdit/logs/finalize_h20.log
log() { echo "[finalize $(date +%F_%T)] $*" >> "$LOG"; }

log "=== finalizer start ==="

# 1) let the original orchestrator finish (it aggregates early; we redo it later)
while pgrep -f 'autostart_h20_queue.sh' >/dev/null 2>&1; do sleep 60; done
log "autostart exited"

# 2) wait for every sampler / runner, including the late racfg_fast workers
while pgrep -f 'models\..*sample' >/dev/null 2>&1; do sleep 120; done
log "no sampler running"
while pgrep -f 'repro/run_repro.sh' >/dev/null 2>&1; do sleep 60; done
log "no runner running"

# 3) report coverage before aggregating
for m in cogvideox wan; do
  for meth in vanilla50 vanilla25 teacache taylorseer racfg_slow racfg_fast; do
    c=$(find "results/repro_h20/$m/$meth" -name 'video_*.mp4' 2>/dev/null | wc -l)
    log "coverage $m/$meth = $c"
  done
done

# 4) aggregate on an idle GPU (h20 root only)
python repro/aggregate.py --model cogvideox --n 100 --device cuda:0 >> logs/aggregate_final_cog.log 2>&1
log "aggregate cogvideox rc=$?"
python repro/aggregate.py --model wan --n 100 --device cuda:0 >> logs/aggregate_final_wan.log 2>&1
log "aggregate wan rc=$?"

# 5) push structured artifacts only (videos are excluded by repo .gitignore)
cd /root/viewdit_git || exit 1
mkdir -p results/repro_h20
( cd /root/viewdit/results/repro_h20 \
  && find . -type f \( -name '*.json' -o -name '*.log' -o -name '*.txt' -o -name '*.md' \) -print0 \
     | xargs -0 -r -I{} cp --parents -f {} /root/viewdit_git/results/repro_h20/ )
git add -f results/repro_h20
if git diff --cached --quiet 2>/dev/null; then
  log "nothing new to commit"
else
  git commit -q -m "h20: final aggregate results for cogvideox + wan (4x H20, bf16, GPU-only) [ready-for-review]"
  log "committed"
fi
timeout 300 git push origin h20 >> "$LOG" 2>&1
log "push done rc=$?"
log "=== finalizer end ==="
