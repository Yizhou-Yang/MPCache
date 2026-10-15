#!/usr/bin/env bash
# GPU queue (2026-10-15, v2): CogVideoX smoke done by hand -> G0a (base env, no download needed)
# -> Wan smoke once the weights are complete -> touch G0A_DONE, which releases queue_full.sh.
cd /root/viewdit
cd experiments/video && /root/miniconda3/bin/python g0a_costmodel.py --seeds 3000,3001 --nprompt 8 \
    --skips 1,3,5,7,9,11,13,15,17,19,21,23 > /root/viewdit/experiments/video/g0a_costmodel.log 2>&1
cd /root/viewdit
while ! grep -q "\[setup\] DONE" logs/repro_setup.log 2>/dev/null; do sleep 60; done
bash repro/run_repro.sh wan 0 1 vanilla50 teacache > logs/repro_smoke_wan.log 2>&1
touch logs/REPRO_SMOKE_DONE
touch logs/G0A_DONE
