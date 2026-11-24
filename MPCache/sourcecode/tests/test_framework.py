"""Unit tests for the MPCache bilevel horizon-control and chord-shrinkage framework."""

import itertools
import numpy as np

from mpcache import (
    ChordShrinkagePredictor,
    Controller,
    ScheduleController,
    calibrate_certificate,
    certify_budget,
    debiased_profile,
    detect_bifurcation_step,
    fit_hilbert_quadratic,
    gap_amplification_factor,
    geometric,
    interval_cost,
    kkt_density,
    open_loop_schedule,
    schedule_cost,
    solve,
    solve_dp,
    solve_dp_table,
    stiff_sloppy_sensitivity,
    theoretical_midpoint_alpha,
    verify_scale_invariance,
    wrap_forward,
)


PROFILE_COG30 = [
    0.569, 0.452, 0.313, 0.192, 0.118, 0.094, 0.063, 0.061, 0.054, 0.053,
    0.047, 0.045, 0.043, 0.043, 0.042, 0.041, 0.040, 0.041, 0.041, 0.041,
    0.041, 0.041, 0.041, 0.042, 0.042, 0.042, 0.043, 0.050,
]


def _interp_50(raw30):
    x30 = np.linspace(1, 49, len(raw30))
    prof = list(np.interp(np.arange(50), x30, raw30))
    prof[0] = raw30[0] * 1.15
    return prof


def test_interval_cost_matches_single_skip_for_all_beta():
    prof = [0.6, 0.5, 0.3, 0.1, 0.05]
    for beta in (0.0, 0.5, 1.0, 1.5):
        assert interval_cost(prof, 1, 2, beta=beta) == 0.0
        assert abs(interval_cost(prof, 1, 3, beta=beta) - prof[2]) < 1e-12


def test_solve_dp_matches_brute_force_global_optimum():
    T, B = 13, 6
    prof = [0.8 * (0.72 ** i) + 0.02 for i in range(T)]
    dp_sched = solve_dp(prof, budget=B, beta=0.5, debias=False)
    dp_val = schedule_cost(prof, dp_sched, beta=0.5)

    best_bf = float("inf")
    for interior in itertools.combinations(range(1, T - 1), B - 2):
        cand = [0, *interior, T - 1]
        best_bf = min(best_bf, schedule_cost(prof, cand, beta=0.5))
    assert abs(dp_val - best_bf) < 1e-12


def test_theorem1_kkt_head_monotonicity_and_tail_regularization():
    prof50 = _interp_50(PROFILE_COG30)
    # Naive additive (beta=0) leaves a massive unrefreshed tail blackout (>20 steps).
    add_sched = solve(debiased_profile(prof50), budget=22)
    max_gap_add = max(add_sched[k + 1] - add_sched[k] for k in range(len(add_sched) - 1))
    assert max_gap_add >= 20

    # Age-weighted Bellman DP (beta=0.5) bounds maximum tail gap and expands head with B.
    heads_dp = []
    heads_kkt = []
    for B in (10, 14, 17, 20, 22):
        s_dp = solve_dp(prof50, budget=B, beta=0.5, debias=True)
        h_dp = 0
        while h_dp < len(s_dp) and s_dp[h_dp] == h_dp:
            h_dp += 1
        heads_dp.append(h_dp)
        _, _, h_kkt = kkt_density(prof50, budget=B, beta=0.5, debias=True)
        heads_kkt.append(h_kkt)

    assert heads_dp == sorted(heads_dp) and heads_dp[-1] >= 14
    assert heads_kkt == sorted(heads_kkt) and heads_kkt[-1] >= 14

    s22 = solve_dp(prof50, budget=22, beta=0.5, debias=True)
    max_gap_dp = max(s22[k + 1] - s22[k] for k in range(len(s22) - 1))
    assert max_gap_dp <= 10


def test_theorem2_receding_horizon_scale_invariance():
    prof50 = debiased_profile(_interp_50(PROFILE_COG30))
    rng = np.random.default_rng(42)
    trajectories = [
        rng.uniform(0.05, 25.0, size=25).tolist()
        for _ in range(10)
    ]
    assert verify_scale_invariance(50, 22, prof50, trajectories, beta=0.5)

    # Also verify runtime Controller closed_loop vs open_loop produces identical executed steps.
    ctrl_cl = Controller(profile=prof50[:20], budget=9, T=20, rate=0.05, closed_loop=True, beta=0.5)
    ol = open_loop_schedule(20, 9, prof50[:20], beta=0.5)
    executed = []
    for t in range(20):
        left = 9 - len(executed)
        if ctrl_cl.decide(t, left):
            val = np.array([1.0 + 0.3 * np.sin(t), 0.5 * t])
            ctrl_cl.observe(t, val)
            executed.append(t)
    assert executed == ol


def test_theorem3_midpoint_quadrature_and_hilbert_fit():
    h, g = 4, 6
    s_prev, s_curr = 10, 10 + h
    v = lambda t: np.array([2.0 - 0.15 * t, -1.0 + 0.08 * t])
    alpha_mid = g / (2.0 * h)
    pred = ChordShrinkagePredictor(policy="damped", alpha=alpha_mid)
    pred.observe_exact(s_prev, v(s_prev))
    pred.observe_exact(s_curr, v(s_curr))
    drift_sum = sum(pred.predict(s_curr + k) - v(s_curr + k) for k in range(1, g))
    assert np.allclose(drift_sum, 0.0, atol=1e-12)

    sched = list(range(16)) + [18, 22, 28, 35, 42, 49]
    alpha_th = theoretical_midpoint_alpha(sched, coherence=0.95)
    assert 0.50 <= alpha_th <= 0.68
    assert gap_amplification_factor(sched) > 3.4

    fit49 = fit_hilbert_quadratic([0.0, 0.25, 0.5, 1.0], [0.1549, 0.1214, 0.1041, 0.1176])
    assert fit49["c2"] > 0
    assert 0.50 <= fit49["alpha_star"] <= 0.70
    assert fit49["r2_lpips"] > 0.99

    class VecModule:
        def __init__(self, vals):
            self.vals = vals
            self.i = 0

        def forward(self, step):
            out = (self.vals[step],)
            self.i += 1
            return out

    vals = [np.array([0.0]), np.array([2.0]), np.array([3.5]), np.array([4.2]), np.array([4.5])]
    mod = VecModule(vals)
    ctrl = ScheduleController(schedule=[0, 1, 4], alpha=0.5, policy="damped")
    restore = wrap_forward(mod, ctrl)
    outs = [mod.forward(t)[0] for t in range(5)]
    restore()
    ctrl.verify()
    assert ctrl.calls == 3 and mod.i == 3
    assert np.allclose(outs[2], [3.0]) and np.allclose(outs[3], [3.0])


def test_discovery1_phase_gated_operator_coupling():
    prof50 = _interp_50(PROFILE_COG30)
    h_crit = detect_bifurcation_step(prof50, tail_factor=2.0)
    assert 10 <= h_crit <= 18

    # Under a budget-starved schedule that skips inside Regime I (e.g. step 2) and Regime II (step 16),
    # regime_head=12 gates early steps to zero-order reuse while applying alpha=0.5 in the tail.
    ctrl = ScheduleController(
        schedule=[0, 1, 14, 15, 20],
        alpha=0.5,
        policy="phase_gated",
        regime_head=12,
    )
    vals = {
        0: np.array([10.0]),
        1: np.array([6.0]),   # huge orthogonal jump (-4.0) in Regime I
        14: np.array([2.0]),
        15: np.array([1.8]),  # smooth coherent chord (-0.2) in Regime II
        20: np.array([1.0]),
    }
    outs = []
    for t in range(21):
        outs.append(ctrl(lambda t=t: vals[t]))
    ctrl.verify()
    # Step 2 (inside Regime I, < 12) must NOT extrapolate -4.0*0.5; it stays safely at 6.0!
    assert np.allclose(outs[2], [6.0])
    # Step 16 (inside Regime II, >= 12) applies midpoint shrinkage: 1.8 + 0.5*(1.8 - 2.0) = 1.7!
    assert np.allclose(outs[16], [1.7])


def test_discovery2_stiff_sloppy_anisotropy():
    prof50 = _interp_50(PROFILE_COG30)
    diag = stiff_sloppy_sensitivity(prof50, budget=20, head_ref=12, beta=0.5)
    assert diag["head_violation_rel_increase"] > 1.50
    assert diag["terminal_blackout_rel_increase"] > 0.20
    assert diag["tail_ratio_14_to_30_rel_spread"] < 0.16
    assert diag["stiff_to_sloppy_anisotropy"] > 10.0


def test_discovery3_risk_certified_budget_synthesis():
    prof50 = _interp_50(PROFILE_COG30)
    frontier = solve_dp_table(prof50, beta=0.5, debias=True)
    assert len(frontier) == 49
    sched22, cost22 = frontier[22]
    assert sched22 == solve_dp(prof50, budget=22, beta=0.5, debias=True)
    assert cost22 > frontier[28][1] > frontier[50][1] == 0.0

    calib_scheds = [geometric(b, T=50) for b in (16, 22, 28)]
    cert = calibrate_certificate(
        prof50,
        calib_scheds,
        realised_means=[0.253, 0.158, 0.093],
        realised_worsts=[0.414, 0.259, 0.149],
        beta=0.5,
        shrinkage_multiplier=0.1041 / 0.1549,
    )
    assert cert.r2_mean > 0.98
    assert cert.r2_worst > 0.98
    assert cert.quantile_ratio_cv < 0.02

    alloc_mean = certify_budget(prof50, cert, target_lpips=0.15, risk="mean", use_shrinkage=True)
    alloc_worst = certify_budget(prof50, cert, target_lpips=0.15, risk="worst", use_shrinkage=True)
    assert alloc_mean.predicted_lpips <= 0.15
    assert alloc_mean.calls_saved_by_shrinkage >= 4
    assert alloc_worst.predicted_lpips <= 0.15
    assert alloc_worst.budget > alloc_mean.budget

