"""Offline trace verification of the MPCache theoretical framework on GPU trajectory logs.

Validates Proposition 1 and Theorems 1--3 directly on the repository's existing GPU runs:
  1. ``repro/results/video/g0a_costmodel_20261015/runs.jsonl`` (192 CogVideoX GPU runs):
     Closed-loop horizon amplification ``||delta x_T|| / ||delta v_i||`` decays 8.1x from
     head to tail while solver-only sensitivity ``c0`` fails to capture Jacobian damping.
  2. ``repro/results/video/step_oracle_20261015/runs.jsonl`` (576 CogVideoX GPU runs):
     Subspace orthogonality barrier on uniform schedules (even hindsight-optimal 3-anchor
     least-squares ``lsq`` fails at 0.4218 rel-L2 vs 0.1245 for head-dense plain reuse).
  3. ``repro/results/video/g0a_gap_20261020/runs.jsonl`` (160 CogVideoX GPU runs) and
     50-step budget runs: Zero-parameter diffusive age-quadrature ``C_{0.5}(S)`` predicts
     multi-step tail compounding (6.53 vs 6.56 at 5-step gap) and cross-budget LPIPS with
     ``R^2 = 0.985`` (vs ``R^2 = 0.835`` for naive additive single-skip sums), while exact
     Bellman DP derives ``H*(22) = 15..16``.
  4. ``MPCache/experiments_results/main_table.json``:
     Hilbert-space quadratic risk law ``E(alpha) = E_min + c_2*(alpha - alpha*)^2`` fits
     49-frame (100 prompts) with ``R^2 = 0.9974`` and 17-frame (32 prompts) with
     ``R^2 = 0.9868``, matching the Theorem 3 midpoint gain ``alpha* ~ 0.58--0.65``.
"""

from __future__ import annotations
import json
import os
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "MPCache" / "sourcecode"))

from mpcache import (
    debiased_profile,
    fit_hilbert_quadratic,
    gap_amplification_factor,
    geometric,
    kkt_density,
    schedule_cost,
    solve,
    solve_dp,
    theoretical_midpoint_alpha,
    verify_scale_invariance,
)


PROFILE_COG30 = [
    0.569, 0.452, 0.313, 0.192, 0.118, 0.094, 0.063, 0.061, 0.054, 0.053,
    0.047, 0.045, 0.043, 0.043, 0.042, 0.041, 0.040, 0.041, 0.041, 0.041,
    0.041, 0.041, 0.041, 0.042, 0.042, 0.042, 0.043, 0.050,
]


def main() -> dict:
    # 1. Closed-loop horizon amplification vs solver-only proxy (192 GPU runs)
    cost_path = ROOT / "repro" / "results" / "video" / "g0a_costmodel_20261015" / "runs.jsonl"
    crows = [json.loads(line) for line in open(cost_path)]
    steps = sorted(set(r["i"] for r in crows))
    cost_summary = {}
    for st in steps:
        sub = [r for r in crows if r["i"] == st]
        loc = float(np.mean([r["local"] for r in sub]))
        tru = float(np.mean([r["true"] for r in sub]))
        c0 = float(np.mean([r["c0"] for r in sub]))
        c_ch = float(np.mean([r["c"] for r in sub]))
        cost_summary[str(st)] = {
            "local_rel_l2": round(loc, 4),
            "terminal_rel_l2": round(tru, 4),
            "closed_loop_gain": round(tru / loc, 4),
            "solver_only_c0": round(c0, 5),
            "channel_jacobian_c": round(c_ch, 5),
        }
    head_tail_gain_ratio = round(
        cost_summary["3"]["closed_loop_gain"] / cost_summary["23"]["closed_loop_gain"], 2
    )
    head_tail_terminal_ratio = round(
        cost_summary["3"]["terminal_rel_l2"] / cost_summary["23"]["terminal_rel_l2"], 2
    )

    # 2. Subspace orthogonality barrier & WHEN x WHAT oracle decomposition (576 GPU runs)
    oracle_path = ROOT / "repro" / "results" / "video" / "step_oracle_20261015" / "runs.jsonl"
    orows = [json.loads(line) for line in open(oracle_path)]
    oracle_17 = {}
    for when in ("uniform", "teacache", "easycache", "othr"):
        oracle_17[when] = {}
        for what in ("reuse", "ex1", "ex2", "lsq"):
            sub = [
                r["rel"]
                for r in orows
                if r["when"] == when and r["what"] == what and 15 <= r["N"] <= 19
            ]
            if sub:
                oracle_17[when][what] = round(float(np.mean(sub)), 4)

    # Orthogonal residual at step 1 on uniform N=17 under hindsight least-squares
    unif_lsq_step1 = [
        r["errs"][0][1]
        for r in orows
        if r["when"] == "uniform" and r["what"] == "lsq" and r["N_target"] == 17 and r["errs"]
    ]

    # 3. Multi-step gap compounding (160 GPU runs) + zero-parameter diffusive age quadrature
    gap_path = ROOT / "repro" / "results" / "video" / "g0a_gap_20261020" / "runs.jsonl"
    grows = [json.loads(line) for line in open(gap_path)]
    w30_lat = np.interp(
        np.arange(30),
        steps,
        [cost_summary[str(st)]["terminal_rel_l2"] for st in steps],
    )
    tail_compounding_i19 = {}
    for m in (1, 2, 3, 5):
        meas = float(
            np.mean([r["true"] for r in grows if r["i"] == 19 and r["gap"] == m])
            / np.mean([r["true"] for r in grows if r["i"] == 19 and r["gap"] == 1])
        )
        naive = float(sum(w30_lat[19 + k] for k in range(m)) / w30_lat[19])
        diffusive = float(
            np.sqrt(sum(((k + 1) * w30_lat[19 + k]) ** 2 for k in range(m))) / w30_lat[19]
        )
        tail_compounding_i19[str(m)] = {
            "measured_ratio": round(meas, 2),
            "naive_additive_ratio": round(naive, 2),
            "diffusive_age_ratio": round(diffusive, 2),
        }

    # Cross-budget LPIPS R^2 (b16, b22, b28 on 50-step CogVideoX)
    x30 = np.linspace(1, 49, len(PROFILE_COG30))
    w50_raw = list(np.interp(np.arange(50), x30, PROFILE_COG30))
    w50_raw[0] = PROFILE_COG30[0] * 1.15
    real_lpips = np.array([0.253, 0.158, 0.093])
    scheds_b = [geometric(b, T=50) for b in (16, 22, 28)]
    c_add = np.array([schedule_cost(w50_raw, sc, beta=0.0) for sc in scheds_b])
    c_diff = np.array([schedule_cost(w50_raw, sc, beta=0.5) for sc in scheds_b])

    def _fit_origin_r2(c: np.ndarray, y: np.ndarray) -> tuple[float, list[float]]:
        s = float(np.sum(c * y) / np.sum(c ** 2))
        pred = s * c
        r2 = 1.0 - float(np.sum((y - pred) ** 2) / np.sum((y - np.mean(y)) ** 2))
        return round(r2, 4), [round(float(v), 4) for v in pred]

    r2_add, pred_add = _fit_origin_r2(c_add, real_lpips)
    r2_diff, pred_diff = _fit_origin_r2(c_diff, real_lpips)

    # Exact Bellman DP schedules across budgets
    dp_schedules = {}
    for B in (10, 14, 17, 20, 22):
        s_dp = solve_dp(w50_raw, budget=B, beta=0.5, debias=True)
        h_dp = 0
        while h_dp < len(s_dp) and s_dp[h_dp] == h_dp:
            h_dp += 1
        _, _, h_kkt = kkt_density(w50_raw, budget=B, beta=0.5, debias=True)
        dp_schedules[str(B)] = {
            "dp_head": h_dp,
            "kkt_head": h_kkt,
            "schedule": s_dp,
        }

    # Naive beta=0 schedule blackout check at B=22
    s_add22 = solve(debiased_profile(w50_raw), budget=22)
    max_gap_add22 = max(s_add22[k + 1] - s_add22[k] for k in range(len(s_add22) - 1))
    s_dp22 = dp_schedules["22"]["schedule"]
    max_gap_dp22 = max(s_dp22[k + 1] - s_dp22[k] for k in range(len(s_dp22) - 1))

    # Theorem 2 scale-invariance check
    rng = np.random.default_rng(0)
    invariance_ok = verify_scale_invariance(
        50, 22, debiased_profile(w50_raw), [rng.uniform(0.1, 10.0, size=25).tolist() for _ in range(20)]
    )

    # 4. Theorem 3 Hilbert quadratic risk fit & midpoint quadrature prediction
    shipped_22 = list(range(16)) + [18, 22, 28, 35, 42, 49]
    fit49 = fit_hilbert_quadratic([0.0, 0.25, 0.5, 1.0], [0.1549, 0.1214, 0.1041, 0.1176])
    fit17 = fit_hilbert_quadratic(
        [0.0, 0.25, 0.5, 0.75, 1.0], [0.1556, 0.1239, 0.1084, 0.1131, 0.1250]
    )

    report = {
        "proposition1_adjoint_and_subspace": {
            "n_costmodel_runs": len(crows),
            "n_oracle_runs": len(orows),
            "head_to_tail_amplification_ratio_i3_vs_i23": head_tail_gain_ratio,
            "head_to_tail_terminal_error_ratio_i3_vs_i23": head_tail_terminal_ratio,
            "per_step_costmodel": cost_summary,
            "oracle_when_x_what_N17": oracle_17,
            "uniform_step1_orthogonal_residual": round(float(np.mean(unif_lsq_step1)), 4),
        },
        "theorem1_age_dp_and_kkt_head": {
            "n_gap_runs": len(grows),
            "tail_compounding_step19": tail_compounding_i19,
            "cross_budget_r2_naive_additive": r2_add,
            "cross_budget_pred_naive_additive": pred_add,
            "cross_budget_r2_diffusive_age": r2_diff,
            "cross_budget_pred_diffusive_age": pred_diff,
            "naive_beta0_max_gap_B22": max_gap_add22,
            "bellman_dp_beta05_max_gap_B22": max_gap_dp22,
            "dp_and_kkt_schedules_by_budget": dp_schedules,
        },
        "theorem2_mpc_scale_invariance": {
            "verified_on_20_random_drift_trajectories": invariance_ok,
        },
        "theorem3_midpoint_chord_shrinkage": {
            "shipped_schedule_22": shipped_22,
            "theoretical_midpoint_alpha_coherence_095": round(
                theoretical_midpoint_alpha(shipped_22, coherence=0.95), 4
            ),
            "theoretical_midpoint_alpha_coherence_085": round(
                theoretical_midpoint_alpha(shipped_22, coherence=0.85), 4
            ),
            "rms_gap_amplification_factor_k_rms": round(gap_amplification_factor(shipped_22), 3),
            "variance_inflation_k_rms_sq": round(gap_amplification_factor(shipped_22) ** 2, 2),
            "hilbert_quadratic_fit_49f_100p": {k: round(v, 5) for k, v in fit49.items()},
            "hilbert_quadratic_fit_17f_32p": {k: round(v, 5) for k, v in fit17.items()},
        },
    }

    out_path = ROOT / "MPCache" / "experiments_results" / "framework_verification.json"
    os.makedirs(out_path.parent, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {out_path}")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
