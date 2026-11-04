"""The one runner every benchmark uses.

    python -m mpcache.runner --backend cogvideox --model /w --schedule 0,1,2,3 \
        --prompts prompts.txt --frames 17 --steps 50 --out out/ours --ledger metrics.jsonl

Everything benchmark-specific comes from the backend; everything else - the
controller, the instrumentation, the animation of skipped steps, the ledger - is
shared. A new benchmark is a new file in `backends/`, never a new runner.
"""

from __future__ import annotations
import argparse, json, os

from . import schedule as sched_mod
from .controller import ScheduleController
from .instrument import wrap_forward
from .ledger import record


def build_schedule(spec: str | None, budget: int | None, steps: int) -> list[int]:
    if spec:
        return sorted({int(x) for x in spec.split(",") if x.strip()})
    if budget is None:
        raise SystemExit("need --schedule or --budget")
    return sched_mod.geometric(budget, T=steps)


def save(frames, path: str, fps: int):
    import imageio.v2 as iio
    import numpy as np
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    arr = np.stack([np.asarray(f).astype("uint8") for f in frames])
    iio.mimsave(path, arr, fps=fps, codec="libx264", macro_block_size=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--frames", type=int, default=17)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out", required=True)
    ap.add_argument("--schedule", default=None, help="comma separated exact steps")
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--ledger", default=None)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks"))
    from loader import get
    be = get(a.backend)
    steps_in = build_schedule(a.schedule, a.budget, a.steps)
    if len(steps_in) > a.steps:
        raise SystemExit(f"schedule {len(steps_in)} longer than {a.steps} steps")
    prompts = [l.strip() for l in open(a.prompts) if l.strip()]
    keep = prompts[a.offset:(a.offset + a.limit) if a.limit else None]

    pipe = be.load(a.model)
    restore = None
    for i, prompt in enumerate(keep):
        ctrl = ScheduleController(steps_in)
        restore = wrap_forward(getattr(pipe, be.transformer), ctrl)
        try:
            frames = be.generate(pipe, prompt, a.frames, a.steps, seed=a.offset + i,
                                 **be.defaults)
        finally:
            restore()
        ctrl.verify()
        path = os.path.join(a.out, "video_%06d.mp4" % (a.offset + i))
        save(frames, path, be.defaults.get("fps", 8))
        print("[done] %d calls=%d %s" % (a.offset + i, ctrl.calls, path), flush=True)
        if a.ledger:
            record(a.ledger, a.tag or os.path.basename(a.out), prompt=i + a.offset,
                   calls=ctrl.calls, path=path, backend=a.backend, frames=a.frames, steps=a.steps)
    print("RUNNER_DONE", flush=True)


if __name__ == "__main__":
    main()
