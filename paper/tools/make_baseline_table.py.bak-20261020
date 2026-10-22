#!/usr/bin/env python3
"""Generate sections/tab_baselines.tex from the audited aggregate JSON (no hand-copied numbers).

Usage: python tools/make_baseline_table.py data/cogvideox_table_n32.json data/wan_table_n32.json
Rows whose status is not 'evaluated' (pending / partial-n / partial-timing) are printed as
"--" with the status, never as numbers.
"""
import json, math, sys
from pathlib import Path

LABEL = {'vanilla50': r'Full ($T{=}50$)', 'vanilla25': r'Fewer steps ($T{=}25$)', 'teacache': 'TeaCache',
         'taylorseer': r'TaylorSeer ($N{=}3,O{=}1$)', 'racfg_slow': 'RA-CFGCache-Slow',
         'racfg_fast': 'RA-CFGCache-Fast'}
ORDER = ['vanilla50', 'vanilla25', 'teacache', 'taylorseer', 'racfg_slow', 'racfg_fast']
NAME = {'cogvideox': 'CogVideoX-2B', 'wan': 'Wan2.1-T2V-1.3B'}


def num(x, fmt):
    if x is None or x == '' or (isinstance(x, str) and x.lower() in ('infinity', 'inf')):
        return r'$\infty$' if isinstance(x, str) else '--'
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return r'$\infty$' if math.isinf(x) else '--'
    return fmt.format(x)


def pair(ours, paper, fmt):
    return f"{num(ours, fmt)} / {num(paper, fmt)}"


lines = [r'\begin{table*}[t]', r'\centering', r'\small', r'\setlength{\tabcolsep}{4.2pt}',
         r'\caption{Reproduction of the RA-CFGCache Table~1 protocol with the pinned public harness '
         r'(first $n$ VBench prompts, seed $=$ prompt index, fidelity vs.\ the same-seed 50-step output). '
         r"``paper'' is the published H200/bf16 value; ours uses four H20-96GB in bf16 under the identical "
         r"protocol, with no CPU offload and no VAE tiling. "
         r'$^\dagger$Speedup is a chunk-averaged ratio; H20 and H200 are different silicon, so only its trend is comparable. '
         r'Every row has 100 paired samples; non-numeric entries would mark incomplete rows and are never filled with zeros. '
         r'TaylorSeer on CogVideoX has no published reference value in that table.}',
         r'\label{tab:baselines}',
         r'\begin{tabular}{llccccc}', r'\toprule',
         r'Model & Method & $n$ & LPIPS$\downarrow$ & SSIM$\uparrow$ & PSNR$\uparrow$ & Speedup$^\dagger$ \\',
         r' & & & ours / paper & ours / paper & ours / paper & ours / paper \\', r'\midrule']
ns = set()
for path in sys.argv[1:]:
    d = json.loads(Path(path).read_text())
    rows = {r['method']: r for r in d['rows']}
    ns.add(d['requested_n'])
    first = True
    for m in ORDER:
        if m not in rows:
            continue
        r = rows[m]; p = r.get('paper') or [None] * 5
        s = r.get('summary') or {}
        model = NAME[d['model']] if first else ''
        first = False
        if r['status'] in ('evaluated', 'reference-self'):
            n = r['quality_pair_n']
            cells = [pair(s.get('lpips_mean'), p[0], '{:.4f}'), pair(s.get('ssim_mean'), p[1], '{:.4f}'),
                     pair(s.get('psnr_mean'), p[2], '{:.2f}'), pair(r.get('speedup'), p[3], r'{:.2f}$\times$')]
        else:
            n = r.get('quality_pair_n', 0)
            pp = [num(p[0], '{:.4f}'), num(p[1], '{:.4f}'), num(p[2], '{:.2f}'), num(p[3], r'{:.2f}$\times$')]
            cells = [f"\\emph{{{r['status']}}} / {pp[0]}", f"-- / {pp[1]}", f"-- / {pp[2]}", f"-- / {pp[3]}"]
        lines.append(f"{model} & {LABEL[m]} & {n} & " + ' & '.join(cells) + r' \\')
    lines.append(r'\midrule')
lines[-1] = r'\bottomrule'
lines += [r'\end{tabular}', r'\end{table*}']
out = Path(__file__).resolve().parents[1] / 'sections' / 'tab_baselines.tex'
out.write_text('\n'.join(lines) + '\n')
print(f'wrote {out} (n={sorted(ns)})')
