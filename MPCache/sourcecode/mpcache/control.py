"""Caching as finite-horizon control, and the two ways to solve it.

Formulation. Let the horizon be the remaining denoising steps, the state be the
cache anchor (the last exact step) and the drift observed since it, the action be
evaluate-or-reuse, and the cost of a skip of length k starting at step s be

    cost(s, k) = rate * k * g_s

where `rate` is the per-step relative drift and `g_s` is the terminal-effect weight
of a skip at step s, measured once by single-step ablation. The controller solves

    min  sum over skips of cost(s, k)   s.t.  number of exact steps = budget,
                                             last step exact,

on the remaining horizon, applies only its first decision, then re-solves on the
next observation. That is the receding-horizon (MPC) version.

The **open-loop** version solves the same program once, with `rate` fixed to a
calibrated mean, and executes the schedule without feedback. The two differ in
nothing but the feedback, which is what makes them the right pair to compare.
"""

from __future__ import annotations
from dataclasses import dataclass, field


def _skips_to_cost(rate: float, g: list[float], i: int, T: int, budget: int) -> tuple[float, list[int]]:
    """DP over the remaining horizon: cheapest way to spend `budget` exact calls.

    Returns (cost, decisions) where decisions[j] is True when step i+j is evaluated.
    """
    n = T - i
    if budget >= n:
        return 0.0, [True] * n
    if budget <= 0:
        return float("inf"), [False] * n
    INF = float("inf")
    # best[t][b]: min cost for steps i+t .. T-1 with b calls left to spend
    best = [[INF] * (budget + 1) for _ in range(n + 1)]
    choice = [[0] * (budget + 1) for _ in range(n + 1)]
    best[n][0] = 0.0
    for t in range(n - 1, -1, -1):
        step = i + t
        last = (step == T - 1)
        for b in range(budget + 1):
            if last:
                if b >= 1 and best[t + 1][b - 1] < INF:
                    best[t][b] = best[t + 1][b - 1]
                    choice[t][b] = 1
                continue
            bestv, bestk = INF, 0
            # evaluate here (cost 0), then continue with b-1
            if b >= 1 and best[t + 1][b - 1] < INF:
                bestv, bestk = best[t + 1][b - 1], 1
            # or skip k steps (k>=1) and pay
            for k in range(1, n - t):
                if b > budget:
                    break
                c = rate * k * g[min(step, len(g) - 1)]
                rest = best[t + k][b]
                if rest < INF and bestv > c + rest:
                    bestv, bestk = c + rest, -k
            best[t][b], choice[t][b] = bestv, bestk
    # reconstruct the first decision only (MPC) or the whole path (open loop)
    decisions = []
    t, b = 0, budget
    while t < n:
        k = choice[t][b]
        if k == 1:
            decisions.append(True); t += 1; b -= 1
        elif k < 0:
            decisions.extend([False] * (-k)); t += -k
        else:
            decisions.extend([False] * (n - t)); t = n
    return best[0][budget], decisions


@dataclass
class Controller:
    profile: list[float]          # terminal-effect weights, normalised, length T
    budget: int
    T: int
    rate: float                   # per-step relative drift, updated when closed_loop
    closed_loop: bool = True
    rate_ema: float = 0.5
    anchor: int = -1
    anchor_value: object | None = None
    prev_value: object | None = None
    history: list[tuple[int, float]] = field(default_factory=list)

    def decide(self, step: int, exact_left: int) -> bool:
        _, dec = _skips_to_cost(self.rate, self.profile, step, self.T, exact_left)
        return bool(dec[0]) if dec else True

    def observe(self, step: int, value) -> None:
        """Called after an exact evaluation: update the drift rate from what we saw."""
        v = value.detach().float() if hasattr(value, "detach") else None
        if v is not None and self.anchor_value is not None:
            gap = max(1, step - self.anchor)
            rel = (v - self.anchor_value).norm().item() / (v.norm().item() + 1e-8)
            r = rel / gap
            if self.closed_loop:
                self.rate = (1 - self.rate_ema) * self.rate + self.rate_ema * r
            self.history.append((step, r))
        self.anchor, self.anchor_value = step, v
