#!/usr/bin/env bash
# Evaluate one arm against its reference and append the number to the ledger.
#   bash evaluate.sh ref_dir arm_dir out_dir [tag]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); SDK=$(cd "$HERE/../../sdk" && pwd)
REF=${1:?ref}; ARM=${2:?arm}; OUT=${3:-$(dirname "$ARM")}; TAG=${4:-$(basename "$ARM")}
PYTHONPATH=$SDK python - "$REF" "$ARM" "$OUT/metrics.jsonl" "$TAG" <<'PY'
import sys, json
from mpcache.metrics import compare
from mpcache.ledger import record
ref, arm, ledger, tag = sys.argv[1:5]
r = compare(ref, arm)
record(ledger, tag, **r)
print(tag, r)
PY
