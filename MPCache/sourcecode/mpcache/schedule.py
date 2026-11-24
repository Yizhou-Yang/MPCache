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


def solve_dp_table(
    profile: Sequence[float],
    beta: float = 0.5,
    debias: bool = True,
    floor: float | None = None,
    eps: float = 1.5e-3,
) -> dict[int, tuple[list[int], float]]:
    """Compute the full exact Bellman DP Pareto frontier for all ``B in {2, ..., T}``.

    In a single ``O(T^3)`` pass (< 1.5 ms at ``T = 50``), returns a mapping
    ``budget -> (optimal_schedule, optimal_cost)`` across every feasible call budget.
    """
    T = len(profile)
    assert T >= 2, "profile must contain at least 2 steps"
    w = debiased_profile(profile, floor=floor, eps=eps) if debias else [float(x) for x in profile]
    inf = float("inf")
    dp = [[inf] * T for _ in range(T + 1)]
    parent = [[-1] * T for _ in range(T + 1)]
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

    frontier: dict[int, tuple[list[int], float]] = {}
    for b in range(2, T + 1):
        for s_next in range(b - 1, T):
            best_val = inf
            best_prev = -1
            for s_prev in range(b - 2, s_next):
                cand = dp[b - 1][s_prev] + trans[s_prev][s_next]
                if cand < best_val:
                    best_val = cand
                    best_prev = s_prev
            dp[b][s_next] = best_val
            parent[b][s_next] = best_prev

        sched = [0] * b
        cur = T - 1
        for k in range(b, 0, -1):
            sched[k - 1] = cur
            cur = parent[k][cur]
        frontier[b] = (sched, float(dp[b][T - 1]))
    return frontier


def detect_bifurcation_step(
    profile: Sequence[float],
    tail_factor: float = 2.0,
) -> int:
    """Detect the intrinsic dynamical bifurcation boundary ``H_crit`` from ``profile``.

    Identifies the earliest step after which single-skip terminal sensitivity settles
    within ``tail_factor`` of the median contracted-tail floor, separating Regime I
    (chaotic symmetry-breaking manifold locking) from Regime II (coherent contracted flow).
    """
    T = len(profile)
    if T <= 4:
        return T
    half = sorted(float(x) for x in profile[T // 2 : max(T // 2 + 1, T - 1)])
    tail_med = half[len(half) // 2] if half else float(profile[-1])
    thresh = tail_factor * max(tail_med, 1e-8)
    for i in range(1, T - 1):
        if float(profile[i]) <= thresh:
            return i
    return T - 1


def _exact_budget_tail(budget: int, T: int, head: int, end_to_start_ratio: float) -> list[int]:
    """Construct an exact ``budget``-call schedule with ``head`` prefix and tail gap ratio ``r = g_end / g_start``."""
    steps = list(range(head))
    m_gaps = budget - head
    span = (T - 1) - head
    if m_gaps <= 1 or span <= 0:
        return sorted(set(steps + [T - 1]))[:budget]
    weights = [
        1.0 + (end_to_start_ratio - 1.0) * (k / max(1, m_gaps - 1))
        for k in range(m_gaps)
    ]
    w_sum = sum(weights)
    cum = 0.0
    used = set(steps) | {T - 1}
    for k in range(m_gaps - 1):
        cum += span * (weights[k] / w_sum)
        cand = min(max(int(round(head + cum)), head), T - 2)
        while cand in used and cand < T - 2:
            cand += 1
        while cand in used and cand > head:
            cand -= 1
        used.add(cand)
    return sorted(used)


def stiff_sloppy_sensitivity(
    profile: Sequence[float],
    budget: int = 20,
    head_ref: int = 12,
    beta: float = 0.5,
) -> dict[str, float]:
    """Quantify the Stiff-Sloppy Parameter Hierarchy on ``C_beta(S)``.

    Compares relative horizon-cost degradation at matched exact call count ``budget`` from:
      1. Violating head saturation (shrinking ``head_ref`` to ``head_ref // 2``),
      2. Violating terminal horizon pinning (exhausting calls by ``0.56 * T`` as in unpinned greedy MPC),
      3. Perturbing the interior tail end-to-start gap ratio across ``r in [1.4, 3.0]`` at fixed ``(budget, head_ref)``.
    """
    T = len(profile)
    w = debiased_profile(profile)
    base = _exact_budget_tail(budget, T=T, head=head_ref, end_to_start_ratio=2.0)
    c_base = schedule_cost(w, base, beta=beta)

    # 1. Stiff head violation (e.g., H=6 vs H=12 at identical 20 calls)
    short_head = _exact_budget_tail(budget, T=T, head=max(2, head_ref // 2), end_to_start_ratio=2.0)
    c_head_viol = schedule_cost(w, short_head, beta=beta)

    # 2. Stiff terminal blackout (exhaust all calls by ~0.56*T, leaving terminal [t_cut..T-1] unpinned)
    t_cut = max(budget + 1, int(round(0.56 * T)))
    unpinned_prefix = _exact_budget_tail(budget, T=t_cut, head=min(head_ref, budget - 2), end_to_start_ratio=1.4)
    last_anchor = unpinned_prefix[-1]
    c_unpinned = schedule_cost(w, unpinned_prefix, beta=beta) + interval_cost(
        w, last_anchor, T, beta=beta
    )

    # 3. Sloppy interior tail ratio variation r in [1.4, 2.0, 3.0] at exact matched budget
    tail_costs = [
        schedule_cost(w, _exact_budget_tail(budget, T=T, head=head_ref, end_to_start_ratio=r), beta=beta)
        for r in (1.4, 2.0, 3.0)
    ]
    sloppy_spread = (max(tail_costs) - min(tail_costs)) / max(c_base, 1e-12)
    head_stiffness = (c_head_viol - c_base) / max(c_base, 1e-12)
    terminal_stiffness = (c_unpinned - c_base) / max(c_base, 1e-12)
    return {
        "base_cost": c_base,
        "head_violation_rel_increase": head_stiffness,
        "terminal_blackout_rel_increase": terminal_stiffness,
        "tail_ratio_14_to_30_rel_spread": sloppy_spread,
        "stiff_to_sloppy_anisotropy": head_stiffness / max(sloppy_spread, 1e-9),
    }


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
