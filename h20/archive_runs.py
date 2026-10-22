#!/usr/bin/env python3
"""Scan every run under results/video and write a markdown ablation archive.

For each directory that has closed.jsonl, report per budget N:
  n, median reuse, median candidate, median reduction vs reuse, win rate.
Runs that contain only one operator are reported by absolute median.
Negative results are kept: they are ablations, not failures to hide.
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/viewdit/results/video")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "/root/viewdit/repro/ABLATIONS.md")


def stats(rows):
    by = defaultdict(dict)
    for r in rows:
        by[(r["N"], r["op"])][(r["prompt"], r["seed"])] = r["rel"]
    out = []
    for N in sorted({k[0] for k in by}):
        base = by.get((N, "reuse"), {})
        for op in sorted({k[1] for k in by if k[0] == N}):
            cur = by[(N, op)]
            if not cur:
                continue
            if op != "reuse" and base:
                keys = sorted(set(base) & set(cur))
                if len(keys) < 3:
                    continue
                a = np.array([cur[k] for k in keys])
                b = np.array([base[k] for k in keys])
                ratio = a / b
                lr = np.log(ratio)
                sem = lr.std(ddof=1) / math.sqrt(len(lr))
                out.append(dict(N=N, op=op, n=len(keys), base=np.median(b), med=np.median(a),
                                red=1 - np.median(ratio), win=np.mean(ratio < 1),
                                sig=bool(-lr.mean() > 2 * sem)))
            elif op == "reuse":
                out.append(dict(N=N, op=op, n=len(cur), base=np.nan, med=np.median(list(cur.values())),
                                red=np.nan, win=np.nan, sig=False))
    return out


def load_base(path):
    by = defaultdict(dict)
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["op"] == "reuse":
            by[r["N"]][(r["prompt"], r["seed"])] = r["rel"]
    return by


def main():
    lines = ["# Ablation archive", "",
             "Smoke directories (names starting with `_`) are excluded; partial runs are kept.", "",
             "Every run we produced, including the negative ones. Reduction is the median of",
             "`candidate / reuse` subtracted from one; win is the share of samples where the",
             "candidate beats reuse at the same budget. `2SEM` is the paired log-ratio test.", ""]
    base_path = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    ref = load_base(base_path) if base_path else {}
    runs = sorted([d for d in ROOT.iterdir()
                   if (d / "closed.jsonl").exists() and not d.name.startswith("_")])
    for d in runs:
        rows = [json.loads(l) for l in (d / "closed.jsonl").read_text().splitlines() if l.strip()]
        if not rows:
            continue
        st = stats(rows)
        if not st and ref:
            by = defaultdict(dict)
            for r in rows:
                by[(r["N"], r["op"])][(r["prompt"], r["seed"])] = r["rel"]
            st = []
            for (N, op), cur in sorted(by.items()):
                base = ref.get(N, {})
                keys = sorted(set(base) & set(cur))
                if len(keys) < 3:
                    continue
                a = np.array([cur[k] for k in keys]); b = np.array([base[k] for k in keys])
                ratio = a / b; lr = np.log(ratio)
                sem = lr.std(ddof=1) / math.sqrt(len(lr))
                st.append(dict(N=N, op=op, n=len(keys), base=np.median(b), med=np.median(a),
                               red=1 - np.median(ratio), win=np.mean(ratio < 1),
                               sig=bool(-lr.mean() > 2 * sem)))
        if not st:
            continue
        lines.append(f"## `{d.name}`")
        lines.append("")
        lines.append("| N | op | n | reuse | candidate | reduction | win | 2SEM |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for s in st:
            red = "" if math.isnan(s["red"]) else f"{s['red']:+.3f}"
            win = "" if math.isnan(s["win"]) else f"{s['win']:.2f}"
            base = "" if math.isnan(s["base"]) else f"{s['base']:.4f}"
            lines.append(f"| {s['N']} | {s['op']} | {s['n']} | {base} | {s['med']:.4f} | {red} | {win} | {'yes' if s['sig'] else ''} |")
        lines.append("")
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT} with {len(runs)} runs")


if __name__ == "__main__":
    main()
