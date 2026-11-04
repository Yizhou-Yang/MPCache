#!/usr/bin/env bash
# Push results FROM the machine that produces them, on a timer.
#
# Why: any timer on the laptop dies when the laptop sleeps, and a `sleep` job in the
# agent session never wakes it. The only party that is guaranteed to be awake while an
# experiment runs is the GPU box itself, so it pushes.
#
# Usage (on the GPU box, with the repo cloned next to the results):
#   GH_TOK=<token> RESULTS_DIR=/root/viewdit/results REPO_DIR=/root/MPCache \
#     nohup bash sync_results.sh > /root/sync.log 2>&1 &
#
# The token is read from the environment at launch and used through a temporary
# askpass helper; it is never written to the repository or to the remote URL.
set -uo pipefail
: "${GH_TOK:?export GH_TOK with a repo-scoped token}"
RESULTS_DIR=${RESULTS_DIR:?path to the results directory}
REPO_DIR=${REPO_DIR:?path to the git checkout that holds the results}
BRANCH=${BRANCH:-results}
INTERVAL=${INTERVAL:-600}          # seconds between pushes
DEST=${DEST:-MPCache/experiments/results/raw}

ASK=$(mktemp); printf '#!/bin/sh\necho "$GH_TOK"\n' > "$ASK"; chmod 700 "$ASK"
trap 'rm -f "$ASK"' EXIT

cd "$REPO_DIR" || exit 1
git checkout -B "$BRANCH" >/dev/null 2>&1 || true

while true; do
  mkdir -p "$DEST"
  # copy only the small ledger files, never the videos
  find "$RESULTS_DIR" -name "metrics.jsonl" -o -name "*.json" -maxdepth 3 2>/dev/null |
    while read -r f; do
      rel=${f#$RESULTS_DIR/}
      mkdir -p "$DEST/$(dirname "$rel")"
      cp -f "$f" "$DEST/$rel"
    done
  if ! git diff --quiet || [ -n "$(git ls-files --others --exclude-standard)" ]; then
    git add "$DEST" >/dev/null
    git -c user.name=results -c user.email=results@local commit -q -m "results: $(date -Is) from $(hostname)" || true
    GIT_ASKPASS=$ASK GIT_TERMINAL_PROMPT=0 git -c credential.helper= push -q origin "$BRANCH" \
      && echo "[$(date -Is)] pushed" || echo "[$(date -Is)] push failed, will retry"
  fi
  sleep "$INTERVAL"
done
