#!/usr/bin/env python3
"""Turn a ledger JSON (from repro/ledger.py) into run_repro.sh chunk specs.

Emits one line per (model, missing-range):
    <model>:<start>:<end>:<method,method,...>

The range is the smallest one covering every missing id of those methods, because
run_repro.sh works on [start,end) chunks and refuses to skip ids inside a chunk it decides
to rerun.  Rows listed in --exclude (model:method) are never planned, which is how the
deferred Wan TaylorSeer row stays out of the overnight queue.

usage:
    python3 repro/plan_missing.py logs/ledger.json --n 100 --exclude wan:taylorseer
"""
import argparse
import json
from collections import defaultdict


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ledger")
    ap.add_argument("--n", type=int, default=100, help="only ids in [0,n) are required")
    ap.add_argument("--exclude", action="append", default=[],
                    help="model:method to leave alone (repeatable)")
    a = ap.parse_args()
    led = json.loads(open(a.ledger).read())
    exclude = {tuple(x.split(":", 1)) for x in a.exclude}

    groups: dict[tuple[str, int, int], list[str]] = defaultdict(list)
    for row in led["rows"]:
        key = (row["model"], row["method"])
        if key in exclude or row["status"] == "complete":
            continue
        missing = [i for i in row["missing"] if 0 <= i < a.n]
        if not missing:
            continue
        groups[(row["model"], missing[0], missing[-1] + 1)].append(row["method"])

    for (model, start, end), methods in sorted(groups.items()):
        print(f"{model}:{start}:{end}:{','.join(sorted(methods))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
