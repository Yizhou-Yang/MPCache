#!/usr/bin/env python3
"""Analysis for G0a (cost-model fidelity), revised 2026-10-15 after reading RA-CFGCache (NeurIPS'26).

RA-CFGCache already prices a skip by  local_error x g_hat(timestep),  where g_hat is a
timestep-only propagation gain fitted OFFLINE from single-step perturbations -- exactly our
suffix re-run T_i / L_i. So the claim "propagation-aware cost beats greedy local error" is no
longer ours. What is left for MPCache's online Jacobian is the PER-PROMPT part of the gain.

Signals compared (all predict T_i = true terminal error of skipping step i alone):
  local : L_i                         greedy (TeaCache-style)
  c0    : solver-only propagation     (= ERTACache's  sum dt*eps  view, no denoiser Jacobian)
  prior : L_i * g_hat(i)              RA-CFGCache-style; g_hat(i) = median_{other prompts} T/L at step i
                                      (leave-one-prompt-out, so it is a fair offline calibration)
  c     : online-J propagation        ours

Metrics
  pooled     : Spearman over all (prompt, seed, i) rows
  within-run : Spearman across i inside one (prompt, seed), averaged   -> "which step to skip"
  within-step: Spearman across (prompt, seed) at fixed i, averaged     -> per-prompt adaptivity
  gain-resid : Spearman over rows of log(T/L) - log g_hat(i)  vs  log(C/L) - mean_i log(C/L)
               -> does online J predict the part of the gain a timestep prior cannot?

Revised gate (all must hold, else component 3 collapses onto RA-CFGCache):
  G0a-1  pooled Spearman(c) >= 0.7
  G0a-2  pooled Spearman(c) > pooled Spearman(prior) + 0.05
  G0a-3  gain-resid Spearman >= 0.4   (online J explains per-prompt gain variation)
Also reported: share of log-gain variance that is per-prompt (if tiny, nothing to adapt to).

Usage: python g0a_analyze.py [runs.jsonl]
"""
import json, math, sys
from collections import defaultdict
from statistics import median, mean, pvariance

DEFAULT = '/root/viewdit/results/video/g0a_costmodel_20261015/runs.jsonl'


def rank(xs):
    o = sorted(range(len(xs)), key=lambda k: xs[k])
    r = [0.0] * len(xs)
    k = 0
    while k < len(o):
        j = k
        while j + 1 < len(o) and xs[o[j + 1]] == xs[o[k]]:
            j += 1
        for m in range(k, j + 1):
            r[o[m]] = (k + j) / 2.0
        k = j + 1
    return r


def spearman(a, b):
    if len(a) < 3:
        return float('nan')
    ra, rb = rank(a), rank(b)
    ma, mb = mean(ra), mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den > 0 else float('nan')


def nanmean(xs):
    xs = [x for x in xs if x == x]
    return mean(xs) if xs else float('nan')


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows = [r for r in rows if r['local'] > 0 and r['true'] > 0 and r['c'] > 0]
    prompts = sorted({r['prompt'] for r in rows})
    print(f'rows={len(rows)}  runs={len({(r["prompt"], r["seed"]) for r in rows})}  prompts={len(prompts)}')

    # leave-one-prompt-out timestep prior (RA-CFGCache-style)
    for r in rows:
        others = [q['true'] / q['local'] for q in rows if q['i'] == r['i'] and q['prompt'] != r['prompt']]
        r['ghat'] = median(others) if others else float('nan')
        r['prior'] = r['local'] * r['ghat']

    sig = ['local', 'c0', 'prior', 'c'] + (['cs'] if all('cs' in r for r in rows) else [])
    if all('cosc' in r for r in rows):
        print(f'[direction] median cos(pred,true): solver-only={median(r["cos0"] for r in rows):+.3f}  '
              f'online-J={median(r["cosc"] for r in rows):+.3f}  | median rel-err(online-J)='
              f'{median(r["errc"] for r in rows):.3f}')
    T = [r['true'] for r in rows]
    print('\n[pooled Spearman vs true]')
    pooled = {s: spearman([r[s] for r in rows], T) for s in sig}
    for s in sig:
        print(f'  {s:6s} {pooled[s]:+.3f}')

    runs = defaultdict(list)
    steps = defaultdict(list)
    for r in rows:
        runs[(r['prompt'], r['seed'])].append(r)
        steps[r['i']].append(r)
    print('\n[within-run Spearman across i]  (which step to skip)')
    for s in sig:
        print(f'  {s:6s} {nanmean([spearman([r[s] for r in g], [r["true"] for r in g]) for g in runs.values()]):+.3f}')
    print('\n[within-step Spearman across prompts/seeds]  (per-prompt adaptivity; prior == local here)')
    for s in sig:
        print(f'  {s:6s} {nanmean([spearman([r[s] for r in g], [r["true"] for r in g]) for g in steps.values()]):+.3f}')

    # per-prompt share of log-gain variance
    lg = {id(r): math.log(r['true'] / r['local']) for r in rows}
    step_mean = {i: mean(lg[id(r)] for r in g) for i, g in steps.items()}
    tot = pvariance([lg[id(r)] for r in rows])
    within = mean(pvariance([lg[id(r)] for r in g]) if len(g) > 1 else 0.0 for g in steps.values())
    print(f'\n[log-gain variance] total={tot:.4f}  within-step(per-prompt)={within:.4f}  share={within / tot if tot else float("nan"):.2%}')

    # gain residual test
    lc = {id(r): math.log(r['c'] / r['local']) for r in rows}
    cmean = {i: mean(lc[id(r)] for r in g) for i, g in steps.items()}
    res_true = [lg[id(r)] - step_mean[r['i']] for r in rows]
    res_c = [lc[id(r)] - cmean[r['i']] for r in rows]
    gres = spearman(res_c, res_true)
    print(f'[gain-resid Spearman] online J vs per-prompt gain deviation: {gres:+.3f}')

    g1 = pooled['c'] >= 0.7
    g2 = pooled['c'] > pooled['prior'] + 0.05
    g3 = gres >= 0.4
    print('\n[gate]  G0a-1 pooled(c)>=0.7: %s | G0a-2 c beats prior by 0.05: %s | G0a-3 gain-resid>=0.4: %s'
          % (g1, g2, g3))
    print('VERDICT:', 'PASS (component 3 is ours)' if (g1 and g2 and g3) else
          'FAIL -> online pricing does not beat RA-CFGCache-style timestep prior')


if __name__ == '__main__':
    main()
