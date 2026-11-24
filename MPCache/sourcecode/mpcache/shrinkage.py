"""Causal Midpoint Chord-Shrinkage operator and Hilbert-space quadratic risk analysis.

Given the two most recent exact denoiser outputs ``(s', v_{s'})`` and ``(s, v_s)`` with
backward chord ``Delta_s = v_s - v_{s'}`` (span ``h = s - s' >= 1``), a skipped solver
step ``j = s + k`` (``1 <= k < g``, interval gap ``g = s_next - s``) returns:

- ``reuse`` (``alpha = 0``):               ``v_s``
- ``damped`` / ``midpoint`` (Theorem 3):   ``v_s + alpha * (v_s - v_{s'})``
- ``chord_rate``:                          ``v_s + alpha * (v_s - v_{s'}) / (s - s')``
- ``scaled`` (gap-scaled Taylor):          ``v_s + (j - s) * (v_s - v_{s'})``

Theorem 3 shows that in any perceptual Hilbert space ``H`` (e.g., VGG feature space
where ``LPIPS^2 = ||Phi(y_hat) - Phi(y*)||_2^2``), expected squared risk over ``alpha``
is an exact convex parabola ``E(alpha) = E_min + c_2 * (alpha - alpha*)^2`` whose
minimizer ``alpha* ~= (g_bar / (2 * h_bar)) * rho_bar`` cancels first-order solver
quadrature bias without amplifying unmeasured trajectory curvature by ``O(g^3)``.
"""

from __future__ import annotations
import math
from typing import Any, Sequence

import numpy as np


VALID_POLICIES = ("reuse", "damped", "midpoint", "chord_rate", "scaled")


def _clone_value(x: Any) -> Any:
    if hasattr(x, "detach") and hasattr(x, "clone"):
        return x.detach().clone()
    if isinstance(x, tuple):
        return tuple(_clone_value(e) for e in x)
    if isinstance(x, list):
        return [_clone_value(e) for e in x]
    if isinstance(x, np.ndarray):
        return x.copy()
    return x


def _add_scaled_diff(curr: Any, prev: Any, weight: float) -> Any:
    if weight == 0.0 or prev is None:
        return _clone_value(curr)
    if hasattr(curr, "dtype") and hasattr(curr, "shape"):
        # Works identically for PyTorch tensors and NumPy arrays.
        delta = curr - prev
        return curr + weight * delta
    if isinstance(curr, tuple):
        return tuple(_add_scaled_diff(c, p, weight) for c, p in zip(curr, prev))
    if isinstance(curr, list):
        return [_add_scaled_diff(c, p, weight) for c, p in zip(curr, prev)]
    return curr + weight * (curr - prev)


class ChordShrinkagePredictor:
    """Two-anchor causal predictor implementing midpoint chord shrinkage (Theorem 3)."""

    def __init__(self, policy: str = "damped", alpha: float = 0.5):
        if policy not in VALID_POLICIES:
            raise ValueError(f"Unknown policy {policy!r}; expected one of {VALID_POLICIES}")
        self.policy = policy
        self.alpha = float(alpha)
        self.prev_step: int | None = None
        self.prev_val: Any = None
        self.anchor_step: int | None = None
        self.anchor_val: Any = None

    def reset(self) -> None:
        self.prev_step = None
        self.prev_val = None
        self.anchor_step = None
        self.anchor_val = None

    def observe_exact(self, step: int, value: Any) -> None:
        self.prev_step = self.anchor_step
        self.prev_val = self.anchor_val
        self.anchor_step = int(step)
        self.anchor_val = _clone_value(value)

    def effective_gain(self, step: int) -> float:
        assert self.anchor_step is not None, "cache empty before first exact evaluation"
        if self.policy == "reuse" or self.prev_val is None or self.prev_step is None:
            return 0.0
        if self.policy in ("damped", "midpoint"):
            return self.alpha
        if self.policy == "chord_rate":
            span = max(1, self.anchor_step - self.prev_step)
            return self.alpha / span
        if self.policy == "scaled":
            return float(step - self.anchor_step)
        return 0.0

    def predict(self, step: int) -> Any:
        assert self.anchor_val is not None, f"step {step} skipped before any exact evaluation"
        gain = self.effective_gain(step)
        return _add_scaled_diff(self.anchor_val, self.prev_val, gain)


def theoretical_midpoint_alpha(
    schedule: Sequence[int],
    coherence: float = 0.90,
) -> float:
    """Closed-form Theorem 3 midpoint quadrature gain ``alpha*`` from schedule geometry.

    Across each skipped interval ``[s_m, s_{m+1})`` of length ``g_m = s_{m+1} - s_m > 1``
    preceded by backward chord span ``h_m = s_m - s_{m-1}``, first-order solver drift
    cancels at ``alpha_m = coherence * (g_m / (2 * h_m))``. Weighting by the number of
    skipped steps ``(g_m - 1)`` yields the global constant shrinkage gain.
    """
    s = sorted(int(x) for x in schedule)
    num, den = 0.0, 0.0
    for m in range(1, len(s) - 1):
        g_m = s[m + 1] - s[m]
        if g_m <= 1:
            continue
        h_m = max(1, s[m] - s[m - 1])
        skips = g_m - 1
        num += skips * (g_m / (2.0 * h_m))
        den += skips
    if den == 0.0:
        return 0.0
    return float(coherence * (num / den))


def gap_amplification_factor(schedule: Sequence[int]) -> float:
    """RMS gap-scaling inflation factor ``k_rms(S) = sqrt(mean_{j not in S} (j - s_m)^2)``.

    Quantifies how much un-damped first-order extrapolation ``v_s + (j - s) * Delta_s``
    inflates step-level curvature noise relative to unit-step extrapolation.
    """
    s = sorted(int(x) for x in schedule)
    sq_ages: list[float] = []
    for m in range(len(s) - 1):
        for j in range(s[m] + 1, s[m + 1]):
            sq_ages.append(float((j - s[m]) ** 2))
    if not sq_ages:
        return 1.0
    return float(math.sqrt(sum(sq_ages) / len(sq_ages)))


def fit_hilbert_quadratic(
    alphas: Sequence[float],
    lpips_values: Sequence[float],
) -> dict[str, float]:
    """Fit Theorem 3 Hilbert-space quadratic risk ``E(alpha) = LPIPS(alpha)^2``.

    In VGG perceptual feature space ``LPIPS(y_hat, y*)^2 = ||Phi(y_hat) - Phi(y*)||_2^2``.
    Under linear terminal perturbation ``delta_Phi(alpha) = u_0 + alpha * u_1``,
    ``E(alpha) = c_0 + c_1 * alpha + c_2 * alpha^2 = E_min + c_2 * (alpha - alpha*)^2``
    is an exact convex quadratic with ``c_2 > 0``.
    """
    a = np.asarray(alphas, dtype=np.float64)
    l = np.asarray(lpips_values, dtype=np.float64)
    e = l ** 2
    c2, c1, c0 = np.polyfit(a, e, deg=2)
    e_pred = c0 + c1 * a + c2 * (a ** 2)
    l_pred = np.sqrt(np.maximum(e_pred, 0.0))

    ss_tot_e = float(np.sum((e - np.mean(e)) ** 2))
    ss_res_e = float(np.sum((e - e_pred) ** 2))
    r2_e = 1.0 - ss_res_e / max(ss_tot_e, 1e-15)

    ss_tot_l = float(np.sum((l - np.mean(l)) ** 2))
    ss_res_l = float(np.sum((l - l_pred) ** 2))
    r2_l = 1.0 - ss_res_l / max(ss_tot_l, 1e-15)

    alpha_star = float(-c1 / (2.0 * c2)) if c2 > 0 else float("nan")
    e_min = float(c0 - (c1 ** 2) / (4.0 * c2)) if c2 > 0 else float(np.min(e))
    return {
        "c0": float(c0),
        "c1": float(c1),
        "c2": float(c2),
        "alpha_star": alpha_star,
        "e_min": max(e_min, 0.0),
        "lpips_min": float(math.sqrt(max(e_min, 0.0))),
        "r2_squared_risk": r2_e,
        "r2_lpips": r2_l,
    }
