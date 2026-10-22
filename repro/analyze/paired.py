#!/usr/bin/env python3
"""Paired comparison of two arms in one or two result directories.

Each directory must contain closed.jsonl with rows
  {prompt, seed, N, op, rel, nfe, steps}.

usage:
  python paired.py --cand <dir>:<op> --base <dir>:<op> [--cand2 <dir>:<op> ...]
"""
import argparse, json, math
from collections import defaultdict
import numpy as np


def load(spec):
    d = {}
    for item in spec:
        path, op = item.rsplit(':', 1)
        try:
            lines = open(f'{path}/closed.jsonl').read().splitlines()
        except FileNotFoundError:
            raise SystemExit(f'missing {path}/closed.jsonl')
        for line in lines:
            if not line.strip():
                continue
            r = json.loads(line)
            if r['op'] == op:
                d[(r['N'], r['prompt'], r['seed'])] = r['rel']
    return d


def report(name, cand, base):
    print(f'\n{name}')
    print('  N   n   base_med  cand_med  gain   win   2SEM')
    for N in sorted({k[0] for k in cand}):
        keys = sorted(k for k in cand if k[0] == N and k in base)
        if len(keys) < 3:
            continue
        a = np.array([cand[k] for k in keys])
        b = np.array([base[k] for k in keys])
        ratio = a / b
        lr = np.log(ratio)
        sem = lr.std(ddof=1) / math.sqrt(len(lr))
        print('%3d %3d %9.4f %9.4f %+6.3f %5.2f %6s'
              % (N, len(keys), np.median(b), np.median(a), 1 - np.median(ratio),
                 np.mean(ratio < 1), str(-lr.mean() > 2 * sem)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cand', nargs='+', required=True)
    ap.add_argument('--base', nargs='+', required=True)
    ap.add_argument('--name', default='paired comparison')
    a = ap.parse_args()
    report(a.name, load(a.cand), load(a.base))
