#!/usr/bin/env bash
# 决定性实验：Wan 上 ours(tea_gap) vs reuse（同一参考、同一 NFE）
# 用途：回答"我们的方法是否真的优于纯复用"。没有这个数字，论文不能声称方法有效。
# 前提：新 H20 已完成 h20/RUNBOOK_DEPLOY.md 的部署（环境 + Wan 权重）
set -u
cd /root/viewdit
P=/root/miniconda3/envs/racfg/bin/python
LOG=logs/wan_reuse_check.log
say(){ echo "[$(date +%H:%M)] $*" >> $LOG; }
say "start"

# 1) RA Full 参考（若尚未生成）
if [ "$(find results/ref/ra_full -name '*.mp4' 2>/dev/null | wc -l)" -lt 50 ]; then
  export PATH=/root/miniconda3/envs/racfg/bin:$PATH
  bash /root/RA-CFGCache/RUN/demo_wan.sh --mode original --gpus 0,1,2,3 --limit 50 \
    --model_path /root/viewdit/weights/Wan2.1-T2V-1.3B \
    --output_dir /root/viewdit/results/ref/ra_full >> logs/ra_full.log 2>&1
  say "RA full done: $(find results/ref/ra_full -name '*.mp4' | wc -l)"
fi

# 2) 三条臂，各 50 prompt，4 分片
run_arm(){  # $1 = planners, $2 = outdir, $3 = extra
  for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i $P experiments/video/mpcache_wan.py --seed-from-prompt \
      --negative animation --solver unipc --sched uniform --frames 81 --no-full --save-video \
      --save-n 19 --warmup 3 --Ns 19 --planners "$1" --save-op "$1" $3 \
      --nprompt 13 --prompt-offset $((i*13)) --out "$2/p$i" >> logs/${2//\//_}_p$i.log 2>&1 &
  done
  wait
  say "$1: $(find $2 -name '*.mp4' | wc -l) videos"
}
run_arm reuse   results/chk/reuse   ""
run_arm tea_gap results/chk/tea_gap ""

say "WAN_REUSE_CHECK_DONE"
