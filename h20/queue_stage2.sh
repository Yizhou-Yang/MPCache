#!/usr/bin/env bash
# Stage 2: after the main table queue finishes, keep the 4 cards busy with the
# quality-NFE curve (N=10 and N=17, ex1) — same runner, same protocol, no new code.
set -uo pipefail
P=/root/miniconda3/envs/racfg/bin/python
cd /root/viewdit || exit 1
mkdir -p logs results/wan_arms

while pgrep -f "queue_wan.sh" > /dev/null; do sleep 60; done
echo "[stage2] start $(date +%H:%M)"

arm() {   # arm <name> <N>
  local name=$1 n=$2
  local out="results/wan_arms/$name"
  echo "[stage2] $name (N=$n) at $(date +%H:%M)"
  for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i setsid nohup $P experiments/video/mpcache_wan.py \
      --seed-from-prompt --negative "animation" --solver unipc --sched sigma \
      --frames 81 --no-full --save-video --save-op ex1 --save-n "$n" \
      --Ns "$n" --planners ex1 \
      --nprompt 25 --prompt-offset $((i*25)) \
      --out "$out" > "logs/wan_${name}_p$i.log" 2>&1 < /dev/null &
  done
  sleep 20
  while pgrep -f "mpcache_wan.py.*--out $out" > /dev/null; do sleep 30; done
  echo "[stage2] done $name at $(date +%H:%M)"
}

arm ex1_n10 10
arm ex1_n17 17

echo "[stage2] all done at $(date +%H:%M)"
