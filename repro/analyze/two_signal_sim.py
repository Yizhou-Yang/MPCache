#!/usr/bin/env python3
"""Two-signal simulation on the g0a probe.

Task: with a skip budget of k steps out of 12 candidate steps, choose which k steps to
skip.  The cost of a policy is the measured terminal error of the skipped steps
(sum of `true`, normalised by the sum over all candidates).

Policies
  local    skip the k steps with the smallest forecast error        (greedy signal)
  prior    skip the k steps with the smallest solver-only price     (offline prior)
  online   skip the k steps with the smallest model price           (online model)
  combo    order by local, but demote a step whose model price is in the top half
           (model used as a veto, local used as the ordering)
  oracle   skip the k steps with the smallest *measured* error      (upper bound)

Also reports a budget-flexible variant: skip every step whose score is below the
score's own median, i.e. the controller decides the count from its own price.
"""
import json
import sys

import numpy as np


def main(path):
    rows = [json.loads(l) for l in open(path)]
    by = {}
    for r in rows:
        by.setdefault((r["prompt"], r["seed"]), []).append(r)
    keys = sorted(by)
    print(f"samples={len(keys)} steps/sample={len(by[keys[0]])}")

    for k in (2, 4, 6, 8):
        res = {"local": [], "prior": [], "online": [], "combo": [], "oracle": []}
        for key in keys:
            rs = by[key]
            true = np.array([r["true"] for r in rs])
            local = np.array([r["local"] for r in rs])
            c0 = np.array([r["c0"] for r in rs])
            c = np.array([r["c"] for r in rs])
            n = len(rs)
            tot = true.sum()

            def cost(order):
                return true[order[:k]].sum() / tot

            res["local"].append(cost(np.argsort(local)))
            res["prior"].append(cost(np.argsort(c0)))
            res["online"].append(cost(np.argsort(c)))
            res["oracle"].append(cost(np.argsort(true)))
            # combo: order by local, but push steps with high model price to the back
            veto = c > np.median(c)
            key_order = np.lexsort((local, veto))  # vetoed steps last, then by local
            res["combo"].append(cost(key_order))

        print(f"\nk={k} (share of terminal error from the skipped steps, lower is better)")
        for name in ("local", "prior", "online", "combo", "oracle"):
            v = np.array(res[name])
            print(f"  {name:7s} mean={v.mean():.3f} median={np.median(v):.3f}")

    print("\nbudget-flexible: skip every step below the median of each score")
    res = {"local": [], "prior": [], "online": [], "oracle": []}
    ns = {"local": [], "prior": [], "online": []}
    for key in keys:
        rs = by[key]
        true = np.array([r["true"] for r in rs])
        n = len(rs)
        tot = true.sum()
        for name, key_ in (("local", "local"), ("prior", "c0"), ("online", "c")):
            sc = np.array([r[key_] for r in rs])
            skip = sc <= np.median(sc)
            res[name].append(true[skip].sum() / tot)
            ns[name].append(int(skip.sum()))
        res["oracle"].append(true[np.argsort(true)[: n // 2]].sum() / tot)
    for name in ("local", "prior", "online", "oracle"):
        v = np.array(res[name])
        extra = f" skips={np.mean(ns[name]):.1f}" if name in ns else ""
        print(f"  {name:7s} mean={v.mean():.3f}{extra}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/runs_g0a.jsonl")
