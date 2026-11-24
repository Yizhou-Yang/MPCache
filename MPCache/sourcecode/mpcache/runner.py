"""The unified runner for the MPCache bilevel horizon-control and chord-shrinkage framework.

    python -m mpcache.runner --backend cogvideox --model /w --schedule 0..15,18,22,28,35,42,49 \
        --policy damped --alpha 0.5 --prompts prompts.txt --frames 49 --steps 50 --out out/ours

Everything benchmark-specific comes from the backend; everything else - the
horizon schedule solver (`geometric` or exact Bellman `dp`), the chord-shrinkage
controller, the instrumentation, and the ledger - is shared.
"""

from __future__ import annotations
import argparse, json, os

from . import schedule as sched_mod
from .controller import ScheduleController
from .instrument import wrap_forward
from .ledger import record
from .shrinkage import VALID_POLICIES


def build_schedule(
    spec: str | None,
    budget: int | None,
    steps: int,
    solver: str = "geometric",
    profile_path: str | None = None,
    beta: float = 0.5,
) -> list[int]:
    if spec:
        out: list[int] = []
        for tok in spec.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if ".." in tok:
                a, b = tok.split("..")
                out.extend(range(int(a), int(b) + 1))
            else:
                out.append(int(tok))
        return sorted(set(out))
    if budget is None:
        raise SystemExit("need --schedule or --budget")
    if solver == "dp":
        if profile_path is None:
            raise SystemExit("--profile JSON required when --solver dp")
        with open(profile_path) as f:
            data = json.load(f)
        prof = data["profile"] if isinstance(data, dict) and "profile" in data else data
        return sched_mod.solve_dp(prof, budget=budget, beta=beta)
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
    ap.add_argument("--schedule", default=None, help="comma/range exact steps, e.g. 0..15,18,22,28,35,42,49")
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--solver", choices=("geometric", "dp"), default="geometric")
    ap.add_argument("--profile", default=None, help="JSON single-skip impulse profile for --solver dp")
    ap.add_argument("--beta", type=float, default=0.5, help="Age-compounding exponent for Bellman DP")
    ap.add_argument("--policy", choices=VALID_POLICIES, default="damped",
                    help="Inner-level reconstruction operator (reuse, damped/midpoint, chord_rate, scaled)")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="Midpoint chord-shrinkage gain alpha in [0, 1]")
    ap.add_argument("--regime-head", type=int, default=0,
                    help="Bifurcation boundary H_crit below which skipped steps gate alpha -> 0")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--ledger", default=None)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks"))
    from loader import get
    be = get(a.backend)
    steps_in = build_schedule(
        a.schedule, a.budget, a.steps, solver=a.solver, profile_path=a.profile, beta=a.beta
    )
    if len(steps_in) > a.steps:
        raise SystemExit(f"schedule {len(steps_in)} longer than {a.steps} steps")
    prompts = [l.strip() for l in open(a.prompts) if l.strip()]
    keep = prompts[a.offset:(a.offset + a.limit) if a.limit else None]

    eff_alpha = 0.0 if a.policy == "reuse" or len(steps_in) == a.steps else a.alpha
    pipe = be.load(a.model)
    for i, prompt in enumerate(keep):
        ctrl = ScheduleController(
            steps_in, alpha=eff_alpha, policy=a.policy, regime_head=a.regime_head
        )
        restore = wrap_forward(getattr(pipe, be.transformer), ctrl)
        try:
            frames = be.generate(pipe, prompt, a.frames, a.steps, seed=a.offset + i,
                                 **be.defaults)
        finally:
            restore()
        ctrl.verify()
        path = os.path.join(a.out, "video_%06d.mp4" % (a.offset + i))
        save(frames, path, be.defaults.get("fps", 8))
        print("[done] %d calls=%d policy=%s alpha=%.2f %s" % (a.offset + i, ctrl.calls, a.policy, eff_alpha, path), flush=True)
        if a.ledger:
            record(a.ledger, a.tag or os.path.basename(a.out), prompt=i + a.offset,
                   calls=ctrl.calls, policy=a.policy, alpha=eff_alpha,
                   path=path, backend=a.backend, frames=a.frames, steps=a.steps)
    print("RUNNER_DONE", flush=True)


if __name__ == "__main__":
    main()
