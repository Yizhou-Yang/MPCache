#!/usr/bin/env python3
"""Aggregate closed.jsonl shards into a markdown table.

usage:
  python summarize.py results/video/wan81_n13_p*/closed.jsonl
  python summarize.py --csv out.csv results/video/*/closed.jsonl

Reports, per (N, op): n, mean/median rel, and mean improvement vs `reuse`
computed prompt-by-prompt (paired), which is the number that matters.
"""
import argparse, glob, json, math
import statistics as st


def load(paths):
    rows = []
    for p in paths:
        for ln in open(p):
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            if not math.isfinite(r.get('rel', float('nan'))):
                continue
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='+')
    ap.add_argument('--csv', default='')
    ap.add_argument('--base', default='reuse')
    a = ap.parse_args()

    files = []
    for pat in a.paths:
        files.extend(glob.glob(pat))
    rows = load(sorted(set(files)))
    if not rows:
        raise SystemExit('no rows')

    by = {}
    for r in rows:
        by.setdefault((r['N'], r['op']), []).append(r['rel'])
    prompts = {(r['seed'], r['prompt']) for r in rows}

    base = {}
    for r in rows:
        if r['op'] == a.base:
            base[(r['seed'], r['prompt'], r['N'])] = r['rel']

    print(f'files={len(set(files))} rows={len(rows)} seeds/prompts={len(prompts)}')
    print()
    print('| N | op | n | mean rel | median | vs %s |' % a.base)
    print('|---|---|---|---|---|---|')
    for (N, op) in sorted(by):
        v = by[(N, op)]
        diffs = []
        for r in rows:
            if r['N'] != N or r['op'] != op:
                continue
            b = base.get((r['seed'], r['prompt'], N))
            if b and b > 0:
                diffs.append((r['rel'] - b) / b * 100.0)
        d = f'{st.mean(diffs):+.1f}%' if diffs else '-'
        print(f'| {N} | {op} | {len(v)} | {st.mean(v):.4f} | {st.median(v):.4f} | {d} |')

    if a.csv:
        with open(a.csv, 'w') as f:
            f.write('prompt,seed,N,op,rel,nfe\n')
            for r in sorted(rows, key=lambda x: (x['prompt'], x['N'], x['op'])):
                f.write(f"{r['prompt']},{r['seed']},{r['N']},{r['op']},{r['rel']:.6f},{r['nfe']}\n")
        print(f'\nwrote {a.csv}')


if __name__ == '__main__':
    main()
