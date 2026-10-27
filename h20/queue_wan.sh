#!/usr/bin/env bash
# Keeps all 4 GPUs busy: ex1 (N=13) -> tea_gap (N=13) -> our full (N=50) -> eval.
set -uo pipefail
P=/root/miniconda3/envs/racfg/bin/python
cd /root/viewdit || exit 1
mkdir -p logs results/wan_arms

wait_dir() {           # wait_dir <out-dir-substring>
  sleep 20
  while pgrep -f "mpcache_wan.py.*--out $1" > /dev/null; do sleep 30; done
  echo "[queue] done: $1 at $(date +%H:%M)"
}

arm() {                # arm <name> <Ns> <op> <extra...>
  local name=$1 ns=$2 op=$3; shift 3
  local out="results/wan_arms/$name"
  [ "$name" = full ] && out="results/wan_full/n50"
  echo "[queue] start $name -> $out at $(date +%H:%M)"
  for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i setsid nohup $P experiments/video/mpcache_wan.py \
      --seed-from-prompt --negative "animation" --solver unipc --sched sigma \
      --frames 81 --no-full --save-video --save-op "$op" --save-n "$ns" \
      --Ns "$ns" --planners "$op" \
      --nprompt 25 --prompt-offset $((i*25)) \
      --out "$out" > "logs/wan_${name}_p$i.log" 2>&1 < /dev/null &
  done
  wait_dir "$out"
}

arm ex1 13 ex1
arm tea_gap 13 tea_gap
arm full 50 ex1

echo "[queue] all arms done at $(date +%H:%M)"
$P repro/metrics.py --ref results/wan_full/n50/videos/ex1 --cmp results/wan_arms/ex1/videos/ex1 \
   --out-csv results/metrics_ex1_vs_ownfull.csv --out-json results/metrics_ex1_vs_ownfull.json > logs/eval_ex1.log 2>&1
$P repro/metrics.py --ref results/wan_full/n50/videos/ex1 --cmp results/wan_arms/tea_gap/videos/tea_gap \
   --out-csv results/metrics_teagap_vs_ownfull.csv --out-json results/metrics_teagap_vs_ownfull.json > logs/eval_teagap.log 2>&1
echo "[queue] eval done at $(date +%H:%M)"
