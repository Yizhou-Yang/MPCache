#!/usr/bin/env python3
"""Merge H20 shard outputs into repro-style run directories + manifest.

Shards live in /root/viewdit/results/video/<shard>/closed.jsonl and are merged by
(prompt, seed, N, op) into /root/MPCache/repro/results/video/<run>/closed.jsonl.
Idempotent: safe to call every 20 minutes while the jobs are still running.
"""
import json
from pathlib import Path

V = Path('/root/viewdit/results/video')
OUT = Path('/root/MPCache/repro/results/video')

GROUPS = {
    'wan33_20261022a': dict(glob='wan33_p*', model='Wan2.1-T2V-1.3B-Diffusers',
                            frames=33, height=480, width=832, steps=50,
                            solver='flowmatch-euler', sched='uniform', dtype='fp16',
                            ckpt='Wan-AI/Wan2.1-T2V-1.3B-Diffusers'),
    'hy49_pilot_20261022a': dict(glob='hy49_s480*', model='HunyuanVideo',
                           frames=49, height=480, width=832, steps=50,
                           solver='flowmatch-euler', sched='uniform', dtype='fp16',
                           ckpt='hunyuanvideo-community/HunyuanVideo'),
    'hy49_20261022a': dict(glob='hy49_p*', model='HunyuanVideo',
                           frames=49, height=480, width=832, steps=50,
                           solver='flowmatch-euler', sched='uniform', dtype='fp16',
                           ckpt='hunyuanvideo-community/HunyuanVideo'),
}


def main():
    total = 0
    for name, meta in GROUPS.items():
        files = sorted(V.glob(meta['glob'] + '/closed.jsonl'))
        rows = {}
        for f in files:
            for ln in f.read_text().splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    r = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                rows[(r['prompt'], r['seed'], r['N'], r['op'])] = r
        if not rows:
            continue
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        with open(d / 'closed.jsonl', 'w') as f:
            for k in sorted(rows):
                f.write(json.dumps(rows[k]) + '\n')
        man = {k: meta[k] for k in ('model', 'ckpt', 'frames', 'height', 'width',
                                    'steps', 'solver', 'sched', 'dtype')}
        man.update(rows=len(rows),
                   shards=[str(Path(f).parent.name) for f in files],
                   prompts=sorted({r['prompt'] for r in rows.values()}),
                   budgets=sorted({r['N'] for r in rows.values()}),
                   ops=sorted({r['op'] for r in rows.values()}),
                   note='H20 driver 535.247.01 / CUDA 12.2 / torch 2.5.1+cu121')
        (d / 'manifest.json').write_text(json.dumps(man, indent=2))
        print(f'{name}: {len(rows)} rows from {len(files)} shards')
        total += len(rows)
    print('total', total)


if __name__ == '__main__':
    main()
