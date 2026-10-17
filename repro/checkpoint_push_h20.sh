#!/usr/bin/env bash
# Periodic + final checkpoint push of H20 reproduction artifacts.
# Only structured artifacts (json/log/txt/md) are pushed; videos are excluded by repo .gitignore.
set -uo pipefail

REPO=/root/viewdit_git
SRC=/root/viewdit/results/repro_h20
LOG=/root/viewdit/logs/checkpoint_push.log
log() { echo "[ckpt $(date +%F_%T)] $*" >> "$LOG"; }

cd "$REPO" || exit 2
mkdir -p results/repro_h20

sync_results() {
  [ -d "$SRC" ] || { log "results dir absent"; return 1; }
  ( cd "$SRC" && find . -type f \( -name '*.json' -o -name '*.log' -o -name '*.txt' -o -name '*.md' \) -print0 \
      | xargs -0 -r -I{} cp --parents -f {} "$REPO/results/repro_h20/" 2>/dev/null )
  cd "$REPO" || return 1
  git add -f results/repro_h20 2>/dev/null
  if git diff --cached --quiet 2>/dev/null; then
    log "no new artifacts"
    return 0
  fi
  git commit -q -m "h20: reproduction checkpoint $(date +%F_%H:%M) [wip]" 2>/dev/null || true
  local out
  out=$(timeout 300 git push origin h20 2>&1 | tail -2)
  log "push: $out"
}

while pgrep -f 'autostart_h20_queue.sh' >/dev/null 2>&1; do
  sync_results
  sleep 1200
done

log "autostart finished; final sync"
sync_results
log "checkpoint pusher exiting"
