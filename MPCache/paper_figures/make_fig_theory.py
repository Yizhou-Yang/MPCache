"""Generate Figure 4 (fig_theory.pdf): Four Sharp Structural Laws & Risk Certification on GPU Traces."""

import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "MPCache" / "sourcecode"))
from mpcache import calibrate_certificate, geometric, schedule_cost, solve_dp_table

VERIF = json.loads((ROOT / "MPCache" / "experiments_results" / "framework_verification.json").read_text())

OUT_DIRS = [
    Path(__file__).resolve().parent,
    ROOT / "paper" / "figures",
]

plt.rcParams.update({
    "font.size": 6.4,
    "axes.labelsize": 6.6,
    "axes.titlesize": 6.8,
    "xtick.labelsize": 5.9,
    "ytick.labelsize": 6.0,
    "legend.fontsize": 5.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, axes = plt.subplots(1, 4, figsize=(7.05, 1.96), constrained_layout=True)
ax1, ax2, ax3, ax4 = axes

# -------- Panel (a): Dynamical Phase Bifurcation (192 GPU runs) --------
cm = VERIF["discovery1_phase_bifurcation_and_sign_inversion"]["per_step_costmodel"]
steps = sorted(int(k) for k in cm.keys() if int(k) >= 3)
gains = np.array([cm[str(s)]["closed_loop_gain_mean_of_ratios"] for s in steps])
cos0 = np.array([cm[str(s)]["directional_cosine_cos0"] for s in steps])
c0 = np.array([cm[str(s)]["solver_only_c0"] for s in steps])
c0_scaled = c0 / c0[0] * gains[0]

ax1.axvspan(2.5, 10.5, color="#fdf0ee", alpha=0.85, zorder=0)
ax1.plot(steps, gains, "o-", color="#c0504d", lw=1.15, ms=2.8, label="Closed-loop $\\|\\Psi_{i\\to T}\\|$ ($8.1\\times\\downarrow$)")
ax1.plot(steps, cos0, "^-", color="#2f6fb3", lw=1.1, ms=2.8, label="Chord align. $\\cos\\theta_i$ ($5.2\\times\\uparrow$)")
ax1.plot(steps, c0_scaled, "s--", color="#888888", lw=0.9, ms=2.4, label="Open-loop $c_0$ (wrong)")
ax1.set_title("(a) Phase Bifurcation", pad=3.5, weight="bold")
ax1.set_xlabel("Step $i$ (30-step grid)")
ax1.set_ylabel("Gain / Cosine")
ax1.set_ylim(0.0, 1.38)
ax1.set_xticks([3, 7, 11, 15, 19, 23])
ax1.text(6.5, 1.24, "Regime I", fontsize=5.3, color="#a83835", ha="center", weight="bold")
ax1.text(17.5, 1.24, "Regime II", fontsize=5.3, color="#2f6fb3", ha="center", weight="bold")
ax1.legend(loc="upper right", frameon=False, handlelength=1.2, bbox_to_anchor=(1.02, 0.94))

# -------- Panel (b): Predictor Sign-Inversion Paradox (576 GPU runs) --------
d1 = VERIF["discovery1_phase_bifurcation_and_sign_inversion"]
orc = d1["oracle_when_x_what_N17"]
tc10 = d1["sign_inversion_without_head_teacache_N10"]
oth16 = d1["sign_inversion_with_head_othr"]

w = 0.27
ax2.bar(0 - w / 2, orc["uniform"]["reuse"], width=w, color="#cfd6dd", label="Reuse ($\\alpha{=}0$)")
ax2.bar(0 + w / 2, orc["uniform"]["lsq"], width=w, color="#5a6b7c", label="3-anchor LSQ")
ax2.bar(1 - w / 2, tc10["reuse_rel_l2"], width=w, color="#cfd6dd")
ax2.bar(1 + w / 2, tc10["ex1_rel_l2"], width=w, color="#d9822b", label="1st-ord (no head)")
ax2.bar(2 - w / 2, oth16["tau016_reuse_rel_l2"], width=w, color="#cfd6dd")
ax2.bar(2 + w / 2, oth16["tau016_ex1_rel_l2"], width=w, color="#c0504d", label="1st-ord (locked)")

ax2.text(1 + w / 2, tc10["ex1_rel_l2"] + 0.015, "+31%", fontsize=5.4, color="#b86214", ha="center", weight="bold")
ax2.text(2 + w / 2, oth16["tau016_ex1_rel_l2"] + 0.015, "-42%", fontsize=5.4, color="#c0504d", ha="center", weight="bold")
ax2.set_title("(b) Predictor Sign-Flip", pad=3.5, weight="bold")
ax2.set_xticks([0, 1, 2])
ax2.set_xticklabels(["Uniform\n($H{=}0$)", "Skipped\nHead", "Locked\nHead"])
ax2.set_ylabel("Terminal latent rel-$L_2$")
ax2.set_ylim(0, 0.58)
ax2.legend(loc="upper right", frameon=False, handlelength=1.0, labelspacing=0.22)

# -------- Panel (c): Gap Compounding & Rank Inversion (160 GPU runs) --------
tc = VERIF["discovery2_observability_trap_and_stiff_sloppy_hierarchy"]["tail_compounding_step19"]
ms = [1, 2, 3, 5]
meas = [tc[str(m)]["measured_ratio"] for m in ms]
diff = [tc[str(m)]["diffusive_age_ratio"] for m in ms]
naiv = [tc[str(m)]["naive_additive_ratio"] for m in ms]

ax3.plot(ms, meas, "o", color="#2f6fb3", ms=3.6, zorder=4, label="Measured trace")
ax3.plot(ms, diff, "-", color="#c0504d", lw=1.25, zorder=3, label="Diffusive ($\\beta{=}0.5$)")
ax3.plot(ms, naiv, "--", color="#7f7f7f", lw=1.0, zorder=2, label="Additive ($R^2{=}0.83$)")
ax3.set_title("(c) Non-Additive Drift", pad=3.5, weight="bold")
ax3.set_xlabel("Skipped steps  $g{-}1$")
ax3.set_ylabel("Error / 1-skip error")
ax3.set_xticks([1, 2, 3, 4, 5])
ax3.set_ylim(0.5, 7.6)
ax3.legend(loc="upper left", frameon=False, handlelength=1.2, labelspacing=0.22)
ax3.text(4.95, 5.25, "6.56 vs 6.50\nprior $\\rho{=}{-}0.15$", fontsize=5.1, color="#c0504d", ha="right")

# -------- Panel (d): Conformal Risk-Certified Frontier (Thm. 3) --------
PROFILE_COG30 = [
    0.569, 0.452, 0.313, 0.192, 0.118, 0.094, 0.063, 0.061, 0.054, 0.053,
    0.047, 0.045, 0.043, 0.043, 0.042, 0.041, 0.040, 0.041, 0.041, 0.041,
    0.041, 0.041, 0.041, 0.042, 0.042, 0.042, 0.043, 0.050,
]
x30 = np.linspace(1, 49, len(PROFILE_COG30))
w50 = list(np.interp(np.arange(50), x30, PROFILE_COG30))
w50[0] = PROFILE_COG30[0] * 1.15
calib_bs = [16, 22, 28]
real_m = [0.253, 0.158, 0.093]
real_w = [0.414, 0.259, 0.149]
cert = calibrate_certificate(
    w50,
    [geometric(b, T=50) for b in calib_bs],
    real_m,
    real_w,
    beta=0.5,
    shrinkage_multiplier=0.1041 / 0.1549,
)
frontier = solve_dp_table(w50, beta=0.5, debias=True)
bs = list(range(14, 33))
pred_m_reuse = [cert.predict_lpips(schedule_cost(w50, frontier[b][0], 0.5), "mean", False) for b in bs]
pred_w_reuse = [cert.predict_lpips(schedule_cost(w50, frontier[b][0], 0.5), "worst", False) for b in bs]
pred_m_mid = [cert.predict_lpips(schedule_cost(w50, frontier[b][0], 0.5), "mean", True) for b in bs]

ax4.plot(bs, pred_w_reuse, ":", color="#7f7f7f", lw=1.0, label="Worst ($\\alpha{=}0$, $R^2{=}0.98$)")
ax4.plot(bs, pred_m_reuse, "--", color="#2f6fb3", lw=1.05, label="Mean ($\\alpha{=}0$, $R^2{=}0.985$)")
ax4.plot(bs, pred_m_mid, "-", color="#c0504d", lw=1.25, label="MPCache ($\\alpha{=}0.5$)")
ax4.scatter(calib_bs, real_m, color="#2f6fb3", s=12, zorder=5)
ax4.scatter(calib_bs, real_w, color="#7f7f7f", marker="s", s=11, zorder=5)
ax4.annotate(
    "$-5$ calls",
    xy=(21, 0.142),
    xytext=(25.8, 0.185),
    fontsize=5.3,
    color="#c0504d",
    weight="bold",
    arrowprops=dict(arrowstyle="-|>", lw=0.6, color="#c0504d"),
)
ax4.set_title("(d) Certified Budget", pad=3.5, weight="bold")
ax4.set_xlabel("Exact DiT calls  $B$")
ax4.set_ylabel("VGG-LPIPS")
ax4.set_xlim(13.5, 32.5)
ax4.set_ylim(0.05, 0.48)
ax4.legend(loc="upper right", frameon=False, handlelength=1.15, labelspacing=0.22)

for out_dir in OUT_DIRS:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig_theory.{ext}", dpi=300, bbox_inches="tight")
print("wrote 4-panel fig_theory.pdf and fig_theory.png")
