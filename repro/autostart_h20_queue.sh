#!/usr/bin/env bash
# H20 4-GPU orchestration wrapper.
# - Does NOT modify any upstream/repro script; it only launches them with per-GPU CUDA_VISIBLE_DEVICES.
# - Waits for setup_h20.sh, verifies weights, runs a smoke gate, then fills all 4 cards.
# - Never enables CPU offload, never lowers frames/resolution/steps.
set -uo pipefail

export VIEWDIT=${VIEWDIT:-/root/viewdit}
export RACFG_PROFILE=h20
export RACFG_DTYPE=bfloat16
export CPU_OFFLOAD=0
export ALLOW_GIT_PUSH=0
export PUSH_BRANCH=h20
export PATH=/root/miniconda3/envs/racfg/bin:/root/miniconda3/bin:$PATH
cd "$VIEWDIT" || exit 2
mkdir -p logs
LOG="$VIEWDIT/logs/autostart.log"
log() { echo "[autostart $(date +%F_%T)] $*" >> "$LOG"; }

log "=== autostart begin pid=$$ ==="

# 1) wait for the setup script to finish (weights download included)
while pgrep -f 'bash repro/setup_h20.sh' >/dev/null 2>&1; do sleep 20; done
log "setup process finished"

# 2) weights must be complete (model_index.json is necessary, not sufficient -> smoke will prove load)
for w in CogVideoX-2b Wan2.1-T2V-1.3B-Diffusers; do
  if [ ! -f "weights/$w/model_index.json" ]; then
    log "FATAL: weights/$w/model_index.json missing; not launching GPU jobs"
    exit 1
  fi
  log "weights/$w present ($(du -sh "weights/$w" 2>/dev/null | cut -f1))"
done

# 3) smoke gate: 1 prompt, taylorseer (heaviest cache path) on CogVideoX
log "smoke: cogvideox [0,1) taylorseer on gpu0"
CUDA_VISIBLE_DEVICES=0 bash repro/run_repro.sh cogvideox 0 1 taylorseer >> logs/smoke_cog_taylor.log 2>&1
smoke_rc=$?
if [ "$smoke_rc" != "0" ]; then
  log "smoke FAILED rc=$smoke_rc; preserving logs, NOT starting the full queue"
  exit 1
fi
log "smoke passed"

# 4) fill all 4 cards. Chunks are disjoint and resumable (existing videos are skipped).
launch() { # gpu model start end
  local g=$1 m=$2 s=$3 e=$4
  CUDA_VISIBLE_DEVICES=$g nohup bash repro/run_repro.sh "$m" "$s" "$e" \
    >> "logs/queue_gpu${g}_${m}_${s}_${e}.log" 2>&1 &
  log "launch gpu=$g $m [$s,$e) pid=$!"
}
launch 0 cogvideox 0 50
launch 1 cogvideox 50 100
launch 2 wan 0 50
launch 3 wan 50 100

log "all 4 cards launched; waiting for completion"
wait
log "all workers returned"

# 5) aggregate with exact common IDs (h20 root only, never mixed with V100)
python repro/aggregate.py --model cogvideox --n 100 --skip_eval >> logs/aggregate_cog.log 2>&1 \
  && log "aggregate cogvideox ok" || log "aggregate cogvideox FAILED (see logs/aggregate_cog.log)"
python repro/aggregate.py --model wan --n 100 --skip_eval >> logs/aggregate_wan.log 2>&1 \
  && log "aggregate wan ok" || log "aggregate wan FAILED (see logs/aggregate_wan.log)"

log "=== autostart done; check logs/queue_gpu*.log and results/repro_h20 ==="
# Git sync intentionally disabled: no credentials on this host (ALLOW_GIT_PUSH=0).
