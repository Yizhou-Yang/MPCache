#!/usr/bin/env bash
# RA-CFGCache Wan "original" = 50-step full evaluation, the reference for LPIPS.
set -uo pipefail
cd /root/RA-CFGCache || exit 1
export WAN_MODEL_PATH=/root/viewdit/weights/Wan2.1-T2V-1.3B
exec bash RUN/demo_wan.sh \
  --mode original \
  --gpus 0,1,2,3 \
  --limit 100 \
  --output_dir /root/viewdit/results/ra_wan \
  --run-name wan_original_full \
  --model_path "$WAN_MODEL_PATH" \
  --python /root/miniconda3/envs/racfg/bin/python \
  --width 832 --height 480 --num_frames 81 --fps 16 --num_steps 50 \
  --guidance_scale 5.0 --negative_prompt animation \
  --scheduler unipc --flow_shift 3.0
