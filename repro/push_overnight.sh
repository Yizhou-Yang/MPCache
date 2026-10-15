#!/usr/bin/env bash
# Push overnight benchmark progress from the V100 to GitHub (branch: overnight).
# Commits repro/ scripts, G0a code + results, and the refreshed STATUS/table files.
set -u
cd /root/viewdit

# refresh STATUS.md snapshot (tables + progress) so the branch always carries the latest state
if [ -f repro/_merged/table_n*.md ]; then
  cp repro/_merged/table_n*.md repro/STATUS_auto.md 2>/dev/null || true
fi
{
  echo "# MPCache reprogress snapshot"
  echo
  echo "Generated: $(date '+%F %T %Z')"
  echo
  echo "## G0a"
  tail -5 experiments/video/g0a_costmodel.log 2>/dev/null | sed 's/^/    /'
  echo
  echo "## Videos per (model, method)"
  for m in cogvideox wan; do
    for meth in vanilla50 vanilla25 teacache taylorseer racfg_slow racfg_fast; do
      n=$(ls results/repro/$m/$meth/*/videos/*.mp4 2>/dev/null | wc -l)
      echo "  $m/$meth = $n"
    done
  done
} > STATUS_auto.md

git add -A repro STATUS_auto.md experiments/video/g0a_costmodel.py experiments/video/g0a_analyze.py \
  experiments/video/g0a_costmodel.log 2>/dev/null
if ! git diff --cached --quiet; then
  git commit -q -m "overnight: benchmark progress $(date '+%F_%T')" || true
  git push -q origin overnight 2>&1 | tail -2 || echo "push failed (will retry next cycle)"
else
  echo "nothing new to push"
fi
