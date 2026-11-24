"""Caching as finite-horizon Bellman control and Theorem 2 scale-invariance collapse.

Models remaining-horizon allocation at step ``i`` with ``budget`` exact calls left via
age-weighted interval compounding costs:

    C_beta(s, s') = rate * sum_{j = s + 1}^{s' - 1} (j - s)^beta * g[j]

Theorem 2 (Scale-Invariance Collapse): Under any strictly positive online trajectory
drift estimate ``rate_t > 0`` at a fixed hard call budget ``B``, the receding-horizon
Bellman minimizer is strictly invariant to ``rate_t`` at every step and executes the
exact same schedule ``S*`` as open-loop Bellman DP (`open_loop_schedule`).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Sequence
import numpy as np


def _vec_norm(x: Any) -> float:
    if hasattr(x, "detach") and hasattr(x, "float"):
        return float(x.detach().float().norm().item())
    arr = np.asarray(x, dtype=np.float64)
    return float(np.linalg.norm(arr))


def _skips_to_cost(
    rate: float,
    g: Sequence[float],
    i: int,
    T: int,
    budget: int,
    beta: float | None = None,
) -> tuple[float, list[bool]]:
    """DP over remaining horizon ``i..T-1`` spending ``budget`` exact calls.

    When ``beta is None``, preserves the original linear-gap cost ``rate * k * g[step]``.
    When ``beta >= 0``, uses the exact age-weighted interval compounding cost
    ``rate * sum_{m=1}^{k} m^beta * g[step + m]`` across a jump from ``step`` to
    ``step + k + 1`` (where step ``step`` is evaluated and ``k`` steps are skipped).
    """
    n = T - i
    if budget >= n:
        return 0.0, [True] * n
    if budget <= 0:
        return float("inf"), [False] * n
    INF = float("inf")

    if beta is not None:
        # Exact interval Bellman DP on indices {i, ..., T - 1}
        assert i < T - 1 and budget >= 2 or (i == T - 1 and budget >= 1)
        if i == T - 1:
            return 0.0, [True]
        V = [[INF] * (budget + 1) for _ in range(T)]
        nxt = [[0] * (budget + 1) for _ in range(T)]
        V[T - 1][1] = 0.0
        for b in range(2, budget + 1):
            for s in range(T - 2, i - 1, -1):
                best_v, best_j = INF, s + 1
                acc = 0.0
                for j in range(s + 1, T - (b - 2)):
                    if j >= s + 2:
                        m = j - 1
                        age = m - s
                        w_m = float(g[min(m, len(g) - 1)])
                        acc += rate * (w_m if beta == 0.0 else (age ** beta) * w_m)
                    cand = acc + V[j][b - 1]
                    if cand < best_v:
                        best_v, best_j = cand, j
                V[s][b], nxt[s][b] = best_v, best_j
        exact_set = {i}
        cur, b = i, budget
        while b > 1:
            cur = nxt[cur][b]
            b -= 1
            exact_set.add(cur)
        return V[i][budget], [(i + offset) in exact_set for offset in range(n)]

    best = [[INF] * (budget + 1) for _ in range(n + 1)]
    choice = [[0] * (budget + 1) for _ in range(n + 1)]
    best[n][0] = 0.0
    for t in range(n - 1, -1, -1):
        step = i + t
        last = step == T - 1
        for b in range(budget + 1):
            if last:
                if b >= 1 and best[t + 1][b - 1] < INF:
                    best[t][b] = best[t + 1][b - 1]
                    choice[t][b] = 1
                continue
            bestv, bestk = INF, 0
            if b >= 1 and best[t + 1][b - 1] < INF:
                bestv, bestk = best[t + 1][b - 1], 1
            for k in range(1, n - t):
                if b > budget:
                    break
                c = rate * k * float(g[min(step, len(g) - 1)])
                rest = best[t + k][b]
                if rest < INF and bestv > c + rest:
                    bestv, bestk = c + rest, -k
            best[t][b], choice[t][b] = bestv, bestk
    decisions: list[bool] = []
    t, b = 0, budget
    while t < n:
        k = choice[t][b]
        if k == 1:
            decisions.append(True)
            t += 1
            b -= 1
        elif k < 0:
            decisions.extend([False] * (-k))
            t += -k
        else:
            decisions.extend([False] * (n - t))
            t = n
    return best[0][budget], decisions


def open_loop_schedule(
    T: int,
    budget: int,
    g: Sequence[float],
    rate: float = 1.0,
    beta: float = 0.5,
) -> list[int]:
    """Open-loop Bellman DP schedule (identical to closed-loop under Theorem 2)."""
    _, dec = _skips_to_cost(rate, g, 0, T, budget, beta=beta)
    return [t for t, keep in enumerate(dec) if keep]


def verify_scale_invariance(
    T: int,
    budget: int,
    g: Sequence[float],
    scale_trajectories: Sequence[Sequence[float]],
    beta: float = 0.5,
) -> bool:
    """Verify Theorem 2: arbitrary positive online scale trajectories yield identical ``S*``."""
    ref = open_loop_schedule(T, budget, g, rate=1.0, beta=beta)
    for scales in scale_trajectories:
        assert len(scales) >= budget - 1
        chosen = [0]
        cur = 0
        b_left = budget
        for step_idx in range(budget - 1):
            sigma_k = float(scales[step_idx])
            assert sigma_k > 0.0, "drift scale must be strictly positive"
            _, dec = _skips_to_cost(sigma_k, g, cur, T, b_left, beta=beta)
            sub_steps = [cur + off for off, keep in enumerate(dec) if keep]
            cur = sub_steps[1]
            chosen.append(cur)
            b_left -= 1
        if chosen != ref:
            return False
    return True


@dataclass
class Controller:
    profile: list[float]
    budget: int
    T: int
    rate: float = 1.0
    closed_loop: bool = True
    rate_ema: float = 0.5
    beta: float | None = 0.5
    anchor: int = -1
    anchor_value: Any = None
    prev_value: Any = None
    history: list[tuple[int, float]] = field(default_factory=list)

    def decide(self, step: int, exact_left: int) -> bool:
        if step == 0 or self.anchor < 0:
            return True
        if step == self.T - 1:
            return exact_left >= 1
        # Check whether `step` is selected when solving from the last exact anchor
        # or from the current candidate step under `self.rate`.
        _, dec = _skips_to_cost(self.rate, self.profile, self.anchor, self.T, exact_left + 1, beta=self.beta)
        offset = step - self.anchor
        return bool(dec[offset]) if 0 <= offset < len(dec) else True

    def observe(self, step: int, value: Any) -> None:
        """Called after an exact evaluation: update scalar drift rate from observed change."""
        if self.anchor_value is not None:
            gap = max(1, step - self.anchor)
            denom = _vec_norm(value) + 1e-8
            diff_val = value - self.anchor_value
            rel = _vec_norm(diff_val) / denom
            r = max(rel / gap, 1e-9)
            if self.closed_loop:
                self.rate = (1.0 - self.rate_ema) * self.rate + self.rate_ema * r
            self.history.append((step, r))
        self.prev_value = self.anchor_value
        self.anchor = step
        self.anchor_value = value.detach().float() if hasattr(value, "detach") else np.array(value, copy=True)
