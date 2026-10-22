#!/usr/bin/env python3
"""Analyse the gap probe: does the local signal keep ranking skips as the gap grows,
and does the model price take over?

Reads results/video/g0a_gap_20261020/runs.jsonl rows:
  prompt, seed, i, gap, local_end, local_sum, prior, online, true

Prints
 1. Spearman(score, true) per gap for local_end, local_sum, prior, online, mixed.
 2. Within-run (per prompt/seed/i) ranking per gap.
 3. Where the best score switches with the gap.
"""
import json
import sys
from pathlib import Path

import numpy as np


def rankdata(x):
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(1, len(x) + 1)
    return r


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra, rb = rankdata(a) - (len(a) + 1) / 2, rankdata(b) - (len(b) + 1) / 2
    den = np.linalg.norm(ra) * np.linalg.norm(rb)
    return float(ra @ rb / den) if den else float("nan")


def main(path):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    if not rows:
        print("no rows yet")
        return
    gaps = sorted({r["gap"] for r in rows})
    scores = ["local_end", "local_sum", "prior", "online"]
    print(f"rows={len(rows)} samples={len({(r['prompt'], r['seed']) for r in rows})} gaps={gaps}")
    print(f"\n{'gap':>4s} " + " ".join(f"{s:>10s}" for s in scores) + "   best")
    for g in gaps:
        sub = [r for r in rows if r["gap"] == g]
        if len(sub) < 4:
            continue
        true = [r["true"] for r in sub]
        vals = {s: spearman([r[s] for r in sub], true) for s in scores}
        best = max(vals, key=vals.get)
        print(f"{g:4d} " + " ".join(f"{vals[s]:10.3f}" for s in scores) + f"   {best}")

    print("\nwithin-run ranking (one sample, ranked across anchors and gaps)")
    by = {}
    for r in rows:
        by.setdefault((r["prompt"], r["seed"]), []).append(r)
    print(f"{'gap':>4s} " + " ".join(f"{s:>10s}" for s in scores))
    for g in gaps:
        out = []
        for s in scores:
            vs = []
            for key, rs in by.items():
                sub = [r for r in rs if r["gap"] == g]
                if len(sub) > 3:
                    vs.append(spearman([r[s] for r in sub], [r["true"] for r in sub]))
            out.append(np.nanmean(vs) if vs else float("nan"))
        print(f"{g:4d} " + " ".join(f"{v:10.3f}" for v in out))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/runs_gap.jsonl")
