#!/usr/bin/env python3
"""D-route measurement: when does an online model beat an offline prior?

Input: the g0a probe (one row per skipped step) with
  c0    propagated terminal error under the solver-only prior (no denoiser model)
  c     propagated terminal error under the online local model
  true  measured terminal error of that skip
  local ||forecast error|| at the step (the greedy signal)
  errc  the online model's own relative error on this step (reliability proxy)

Outputs
 1. Spearman(score, true) for prior, online, mixed, and the greedy local signal,
    overall and by horizon bucket.
 2. Identifiability: does errc predict where online beats prior?
    (AUC of -errc for the event |c-true| < |c0-true|.)
 3. Selector: a threshold on errc that switches between prior and online,
    and its Spearman(score, true) against each fixed choice.
"""
import json
import sys
from pathlib import Path

import numpy as np


def rankdata(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1)
    # average ties
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    for k in np.where(counts > 1)[0]:
        idx = np.where(inv == k)[0]
        ranks[idx] = ranks[idx].mean()
    return ranks


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = np.linalg.norm(ra) * np.linalg.norm(rb)
    return float(ra @ rb / den) if den else float("nan")


def auc(score, label):
    """P(score of a positive > score of a negative), ties counted 0.5."""
    s = np.asarray(score, float)
    y = np.asarray(label, bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)


def main(path):
    rows = [json.loads(l) for l in open(path)]
    i = np.array([r["i"] for r in rows])
    c0 = np.array([r["c0"] for r in rows])
    c = np.array([r["c"] for r in rows])
    true = np.array([r["true"] for r in rows])
    local = np.array([r["local"] for r in rows])
    errc = np.array([r.get("errc", np.nan) for r in rows])
    mixed = np.sqrt(np.abs(c0 * c))

    print(f"rows={len(rows)}  steps={sorted(set(i.tolist()))}")
    print("\n== 1. rank correlation with the true terminal error ==")
    print(f"{'score':10s} {'all':>7s} {'early':>7s} {'mid':>7s} {'late':>7s}")
    buckets = {"early": i < 10, "mid": (i >= 10) & (i < 20), "late": i >= 20}
    for name, s in [("local", local), ("prior", c0), ("online", c), ("mixed", mixed)]:
        vals = [spearman(s, true)]
        for b in ("early", "mid", "late"):
            m = buckets[b]
            vals.append(spearman(s[m], true[m]) if m.sum() > 2 else float("nan"))
        print(f"{name:10s} " + " ".join(f"{v:7.3f}" for v in vals))

    print("\n== 2. does the model's self-error predict where online wins? ==")
    err_prior = np.abs(c0 - true)
    err_online = np.abs(c - true)
    win = err_online < err_prior
    print(f"online beats prior on {win.mean():.1%} of steps")
    print(f"AUC(-errc -> online wins) = {auc(-errc, win):.3f}")
    print(f"AUC(local  -> online wins) = {auc(local, win):.3f}   (greedy signal as selector)")
    print(f"corr(errc, err_online-err_prior) = {spearman(errc, err_online - err_prior):.3f}")

    print("\n== 3. selector: switch on errc quantiles ==")
    print(f"{'threshold':>10s} {'share online':>13s} {'spearman':>9s}")
    for q in (0.2, 0.4, 0.6, 0.8):
        thr = np.quantile(errc[~np.isnan(errc)], q)
        pick_online = errc <= thr
        sel = np.where(pick_online, c, c0)
        print(f"{q:10.1f} {pick_online.mean():13.2f} {spearman(sel, true):9.3f}")
    print(f"{'always prior':>10s} {0.0:13.2f} {spearman(c0, true):9.3f}")
    print(f"{'always online':>10s} {1.0:13.2f} {spearman(c, true):9.3f}")

    print("\n== 1b. within-run ranking (the planner chooses steps inside one run) ==")
    keys = sorted({(r["prompt"], r["seed"]) for r in rows})
    idx = {(r["prompt"], r["seed"]): [] for r in rows}
    for n, r in enumerate(rows):
        idx[(r["prompt"], r["seed"])].append(n)
    print(f"{'score':10s} {'mean within-run spearman':>24s}")
    for name, sc in [("local", local), ("prior", c0), ("online", c), ("mixed", mixed)]:
        vals = []
        for k in keys:
            ii = idx[k]
            if len(ii) > 3:
                vals.append(spearman(sc[ii], true[ii]))
        print(f"{name:10s} {np.nanmean(vals):24.3f}")

    print("\n== oracle upper bound ==")
    oracle = np.minimum(err_prior, err_online)
    print(f"median |score-true| : prior={np.median(err_prior):.4f} "
          f"online={np.median(err_online):.4f} oracle={np.median(oracle):.4f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/runs_g0a.jsonl")
