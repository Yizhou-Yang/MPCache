#!/usr/bin/env bash
# As soon as ours finishes, compute the fidelity metrics and push them.
# Videos stay on disk (too big for git), so the metrics json/csv are the only
# durable artifact -- do not wait for the 23:15 finalizer.
set -uo pipefail
VIEW=/root/viewdit
REPO=/root/MPCache
PY=/root/miniconda3/envs/racfg/bin/python
LOG=$VIEW/logs/after_ours.log

nmp4() { find "$1" -name 'video_*.mp4' 2>/dev/null | wc -l; }
say() { echo "[$(date +%H:%M)] $*" | tee -a "$LOG"; }

N13=$VIEW/results/ours_wan/n13/tea_gap
FULL=$VIEW/results/ours_wan/full50/tea_gap

say "waiting for ours n13 (now $(nmp4 "$N13")/100)"
while [ "$(nmp4 "$N13")" -lt 100 ]; do sleep 60; done
sleep 30
say "ours done: $(nmp4 "$N13") videos"

mkdir -p "$REPO/repro/results"
"$PY" "$VIEW/repro/metrics.py" --ref "$FULL" --cmp "$N13" \
  --out-json "$REPO/repro/results/metrics_ours_vs_ourfull.json" \
  --out-csv  "$REPO/repro/results/metrics_ours_vs_ourfull.csv" >> "$LOG" 2>&1

say "metrics done: $(grep -o '"lpips_mean": [0-9.]*' "$REPO/repro/results/metrics_ours_vs_ourfull.json" 2>/dev/null | head -1)"
cd "$REPO" && bash "$VIEW/repro/push_results.sh" >> "$LOG" 2>&1
say "pushed"
