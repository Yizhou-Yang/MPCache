"""Horizon-optimal cache scheduling via age-weighted interval Bellman DP and KKT analysis.

Given a single-skip terminal impulse profile ``w[0..T-1]`` measured once per
``(backbone, sampler, T)`` configuration, MPCache models the terminal distortion of
partitioning ``[0, T-1]`` at exact anchor steps ``S = (s_0=0 < s_1 < ... < s_{B-1}=T-1)``
via the age-weighted interval compounding functional:

    C_beta(S) = sum_{m=0}^{B-2} sum_{j = s_m + 1}^{s_{m+1} - 1} (j - s_m)^beta * w_tilde[j]

where ``j - s_m >= 1`` is the anchor age (number of solver steps elapsed since the last
exact evaluation) and ``beta > 0`` (default ``beta = 0.5``, diffusive manifold drift)
captures super-additive multi-step error compounding.

- When ``beta = 0``, ``C_0(S) = sum_{j not in S} w[j]`` recovers the naive additive model
  solved greedily by :func:`solve`.
- When ``beta > 0``, :func:`solve_dp` finds the exact global optimum in ``O(B * T^2)``
  time via Bellman dynamic programming, and :func:`kkt_density` solves the continuous KKT
  relaxation, analytically yielding a budget-dependent saturated exact head ``0..H*(B)-1``
  followed by monotonically growing tail gaps (:func:`geometric`).
"""

from __future__ import annotations
from typing import Sequence


def debiased_profile(
    profile: Sequence[float],
    floor: float | None = None,
    eps: float = 1.5e-3,
) -> list[float]:
    """Subtract the numerical decode/quantization floor from a single-skip profile."""
    T = len(profile)
    if T == 0:
        return []
    interior = [float(profile[i]) for i in range(1, max(2, T - 1)) if float(profile[i]) > 0.0]
    w_floor = (min(interior) if interior else 0.0) if floor is None else float(floor)
    out = [max(float(w) - w_floor, eps) for w in profile]
    out[0] = max(out[0], max(out) * 1.05)
    return out


def interval_cost(
    profile: Sequence[float],
    s: int,
    s_next: int,
    beta: float = 0.5,
) -> float:
    """Age-weighted compounding cost across a skipped interval ``(s, s_next)``."""
    if s_next <= s + 1:
        return 0.0
    total = 0.0
    for j in range(s + 1, s_next):
        age = j - s
        w_j = float(profile[j])
        total += w_j if beta == 0.0 else (age ** beta) * w_j
    return total


def schedule_cost(
    profile: Sequence[float],
    schedule: Sequence[int],
    beta: float = 0.5,
) -> float:
    """Total age-weighted horizon cost ``C_beta(S)`` for an ordered schedule ``S``."""
    s = sorted(int(i) for i in schedule)
    if len(s) < 2:
        return 0.0
    return sum(interval_cost(profile, s[m], s[m + 1], beta=beta) for m in range(len(s) - 1))


def solve_dp(
    profile: Sequence[float],
    budget: int,
    beta: float = 0.5,
    debias: bool = True,
    floor: float | None = None,
    eps: float = 1.5e-3,
) -> list[int]:
    """Exact ``O(B * T^2)`` Bellman dynamic programming solver for horizon allocation.

    Minimizes ``C_beta(S)`` subject to ``|S| = budget`` and ``{0, T - 1} <= S``.
    """
    T = len(profile)
    if budget >= T:
        return list(range(T))
    assert 2 <= budget <= T, f"budget {budget} outside [2, {T}]"

    w = debiased_profile(profile, floor=floor, eps=eps) if debias else [float(x) for x in profile]
    inf = float("inf")
    dp = [[inf] * T for _ in range(budget + 1)]
    parent = [[-1] * T for _ in range(budget + 1)]
    dp[1][0] = 0.0

    trans = [[0.0] * T for _ in range(T)]
    for s_prev in range(T - 1):
        acc = 0.0
        for s_next in range(s_prev + 1, T):
            if s_next >= s_prev + 2:
                j = s_next - 1
                age = j - s_prev
                acc += w[j] if beta == 0.0 else (age ** beta) * w[j]
            trans[s_prev][s_next] = acc

    for b in range(2, budget + 1):
        max_s_next = T - (budget - b)
        for s_next in range(b - 1, max_s_next):
            best_val = inf
            best_prev = -1
            for s_prev in range(b - 2, s_next):
                cand = dp[b - 1][s_prev] + trans[s_prev][s_next]
                if cand < best_val:
                    best_val = cand
                    best_prev = s_prev
            dp[b][s_next] = best_val
            parent[b][s_next] = best_prev

    sched = [0] * budget
    cur = T - 1
    for b in range(budget, 0, -1):
        sched[b - 1] = cur
        cur = parent[b][cur]
    return sched


def kkt_density(
    profile: Sequence[float],
    budget: float,
    beta: float = 0.5,
    debias: bool = True,
    floor: float | None = None,
    eps: float = 1.5e-3,
) -> tuple[list[float], float, int]:
    """Continuous KKT call-density relaxation ``rho*(t) = min(1, (w(t)/lambda)^(1/(1+beta)))``.

    Returns ``(rho, lambda_star, discrete_head_len)`` satisfying ``sum_t rho*(t) == budget``,
    where ``discrete_head_len`` counts the leading prefix where optimal quantized step gap
    ``round(1 / rho*(t)) == 1`` (i.e., ``rho*(t) >= 2/3``).
    """
    T = len(profile)
    assert 2 <= budget <= T, f"budget {budget} outside [2, {T}]"
    if budget >= T:
        return [1.0] * T, 0.0, T

    w = debiased_profile(profile, floor=floor, eps=eps) if debias else [max(float(x), eps) for x in profile]
    exponent = 1.0 / (1.0 + max(beta, 1e-6))

    def density_for(lam: float) -> list[float]:
        out = [min(1.0, (wt / lam) ** exponent) for wt in w]
        out[0] = 1.0
        out[-1] = 1.0
        return out

    lo, hi = 1e-12, max(w) * (T ** (1.0 + beta))
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if sum(density_for(mid)) > budget:
            lo = mid
        else:
            hi = mid

    lam_star = 0.5 * (lo + hi)
    rho = density_for(lam_star)
    h_disc = 0
    while h_disc < T and round(1.0 / max(rho[h_disc], 1e-9)) == 1:
        h_disc += 1
    return rho, lam_star, h_disc


def solve(profile: Sequence[float], budget: int) -> list[int]:
    """Exact greedy allocation under the ``beta = 0`` additive model."""
    T = len(profile)
    if budget >= T:
        return list(range(T))
    last = T - 1
    head_budget = max(budget - 1, 0)
    order = sorted(range(T - 1), key=lambda i: (-profile[i], i))
    chosen = sorted(order[:head_budget])
    return sorted(set(chosen + [last]))


def head_length(profile: Sequence[float], frac: float = 0.5) -> int:
    """Number of leading steps whose sensitivity is above ``frac`` of the maximum."""
    if not profile:
        return 0
    hi = max(profile)
    h = 0
    for i, w in enumerate(profile):
        if i == 0:
            continue
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
    """Terminal cost of a schedule under the ``beta = 0`` additive model."""
    chosen_set = set(chosen)
    return float(sum(w for i, w in enumerate(profile) if i not in chosen_set))


def relative_gap(profile: Sequence[float], budget: int, sched: Sequence[int]) -> float:
    """Relative optimality gap under the ``beta = 0`` additive baseline."""
    best = loss(profile, solve(profile, budget))
    ours = loss(profile, sched)
    if best <= 0:
        return 0.0
    return (ours - best) / best


def relative_dp_gap(
    profile: Sequence[float],
    budget: int,
    sched: Sequence[int],
    beta: float = 0.5,
    debias: bool = True,
) -> float:
    """Relative optimality gap of ``sched`` against the exact Bellman DP optimum."""
    w = debiased_profile(profile) if debias else [float(x) for x in profile]
    opt_sched = solve_dp(w, budget, beta=beta, debias=False)
    opt = schedule_cost(w, opt_sched, beta=beta)
    if opt <= 0.0:
        return 0.0
    return (schedule_cost(w, sched, beta=beta) - opt) / opt


def check(profile: Sequence[float], budget: int, sched: Sequence[int]) -> None:
    """Invariants every schedule must satisfy."""
    assert len(set(sched)) == len(sched), "duplicate steps"
    assert budget <= len(profile) + 1
    assert len(sched) == min(budget, len(profile)), f"budget {budget} != {len(sched)} calls"
    assert len(profile) - 1 in sched, "last step must be exact"
    assert min(sched) >= 0 and max(sched) < len(profile), "step out of range"
