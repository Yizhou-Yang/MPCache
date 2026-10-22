#!/usr/bin/env bash
# Final snapshot, scheduled to fire at 23:15 local regardless of whether anyone is
# watching. Runs the metric pairs that exist, writes a summary block, and pushes.
set -uo pipefail

VIEW=/root/viewdit
REPO=/root/MPCache
PY=/root/miniconda3/envs/racfg/bin/python
LOG=$VIEW/logs/finalize.log

TARGET=$(date -d "today 23:15" +%s)
NOW=$(date +%s)
if [ "$NOW" -lt "$TARGET" ]; then
  echo "[$(date +%H:%M)] sleeping $(( (TARGET-NOW)/60 )) min until 23:15" >> "$LOG"
  sleep $((TARGET - NOW))
fi
echo "[$(date +%H:%M)] finalize start" >> "$LOG"

vdir() { find "$1" -type d -name videos 2>/dev/null | head -1; }
mkdir -p "$REPO/repro/results"

run_pair() {  # $1 label  $2 ref dir  $3 cmp dir
  local label=$1 ref=$2 cmp=$3
  [ -d "$ref" ] && [ -d "$cmp" ] || { echo "[skip] $label (missing dir)" >> "$LOG"; return; }
  local n_ref n_cmp
  n_ref=$(find "$ref" -name 'video_*.mp4' | wc -l)
  n_cmp=$(find "$cmp" -name 'video_*.mp4' | wc -l)
  [ "$n_cmp" -ge 1 ] || { echo "[skip] $label (no cmp videos)" >> "$LOG"; return; }
  echo "[run] $label ref=$n_ref cmp=$n_cmp" >> "$LOG"
  "$PY" "$VIEW/repro/metrics.py" --ref "$ref" --cmp "$cmp" \
    --out-json "$REPO/repro/results/metrics_${label}.json" \
    --out-csv  "$REPO/repro/results/metrics_${label}.csv" >> "$LOG" 2>&1
}

# ours vs our Full (same pipeline)
OUR_FULL=$VIEW/results/ours_wan/full50
[ -d "$OUR_FULL/tea_gap" ] && OUR_FULL="$OUR_FULL/tea_gap"
OUR_N13=$VIEW/results/ours_wan/n13/tea_gap
run_pair ours_vs_ourfull "$OUR_FULL" "$OUR_N13"

# baselines vs RA Full (same pipeline)
RA_FULL=$(vdir "$VIEW/results/ra_wan/wan_original_full")
[ -n "$RA_FULL" ] || RA_FULL=$(vdir "$VIEW/results/ra_wan")
for name in ra_wan_tea ra_wan_cfg ra_wan_cfgfast; do
  d=$(vdir "$VIEW/results/$name")
  [ -n "$d" ] && [ -n "$RA_FULL" ] && run_pair "${name}_vs_rafull" "$RA_FULL" "$d"
done

# compact summary
{
  echo ""
  echo "## 7. 自动收尾快照（$(date +%H:%M)）"
  echo ""
  echo '```'
  for f in "$REPO"/repro/results/metrics_*.json; do
    [ -f "$f" ] || continue
    echo "--- $(basename "$f" .json)"
    "$PY" -c "import json,sys;d=json.load(open('$f'))['summary'];print({k:round(v,4) if isinstance(v,float) else v for k,v in d.items()})" 2>/dev/null
  done
  echo '```'
  echo ""
  echo "视频计数：our Full $(find "$OUR_FULL" -name 'video_*.mp4' 2>/dev/null | wc -l) / ours $(find "$OUR_N13" -name 'video_*.mp4' 2>/dev/null | wc -l)"
} >> "$REPO/h20/EXPERIMENT_SUMMARY.md"

cd "$REPO" && bash "$VIEW/repro/push_results.sh" >> "$LOG" 2>&1
echo "[$(date +%H:%M)] finalize done" >> "$LOG"
