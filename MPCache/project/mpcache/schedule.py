"""Open-loop solution of the caching allocation problem.

The cache decides, once, which steps to evaluate.  Treat the trajectory as a
horizon of T steps, a per-step sensitivity profile w (the terminal loss caused
by skipping that step alone), and a budget B of transformer calls.  The
allocation problem is

    min_{S, |S| = B, T-1 in S}  sum_{i not in S} w_i ,

whose exact solution is greedy: keep the last step, then the B-1 steps with the
largest w.  A model-predictive controller would re-solve this every step; because
w is a property of the sampler rather than of the trajectory, re-solving returns
the same set, so the closed loop collapses to this open-loop schedule.

`geometric` builds the parametric schedule we ship: an exact head, then gaps that
grow by a fixed ratio, then the last step.  It needs only the head length and the
ratio, which are read off the profile.
"""

from __future__ import annotations
from typing import Sequence


def solve(profile: Sequence[float], budget: int) -> list[int]:
    """Exact greedy allocation for a known profile."""
    T = len(profile)
    if budget >= T:
        return list(range(T))
    last = T - 1
    head_budget = max(budget - 1, 0)
    order = sorted(range(T - 1), key=lambda i: (-profile[i], i))
    chosen = sorted(order[:head_budget])
    return sorted(set(chosen + [last]))


def head_length(profile: Sequence[float], frac: float = 0.5) -> int:
    """Number of leading steps whose sensitivity is above `frac` of the maximum."""
    if not profile:
        return 0
    hi = max(profile)
    h = 0
    for i, w in enumerate(profile):
        if i == 0:
            continue          # step 0 cannot be skipped, it has no cached output
        if w >= frac * hi:
            h = i + 1
        else:
            break
    return h


def geometric(budget: int, T: int = 50, head: int | None = None, ratio: float = 1.7) -> list[int]:
    """Head exact, then gaps growing geometrically, ending on the last step."""
    head = head if head is not None else min(12, max(4, budget // 2))
    tail = max(budget - head, 1)
    steps = list(range(head))
    for k in range(1, tail + 1):
        frac = (k / tail) ** ratio
        steps.append(min(int(round(head + frac * ((T - 1) - head))), T - 1))
    return sorted(set(steps + [T - 1]))


def loss(profile: Sequence[float], chosen: Sequence[int]) -> float:
    """Terminal cost of a schedule under the additive model."""
    chosen = set(chosen)
    return float(sum(w for i, w in enumerate(profile) if i not in chosen))


def relative_gap(profile: Sequence[float], budget: int, sched: Sequence[int]) -> float:
    """How much worse `sched` is than the exact allocation, relative to it."""
    best = loss(profile, solve(profile, budget))
    ours = loss(profile, sched)
    if best <= 0:
        return 0.0
    return (ours - best) / best


def check(profile: Sequence[float], budget: int, sched: Sequence[int]) -> None:
    """Invariants every schedule must satisfy."""
    assert len(set(sched)) == len(sched), "duplicate steps"
    assert budget <= len(profile) + 1
    assert len(sched) == min(budget, len(profile)), f"budget {budget} != {len(sched)} calls"
    assert len(profile) - 1 in sched, "last step must be exact"
    assert min(sched) >= 0 and max(sched) < len(profile), "step out of range"
