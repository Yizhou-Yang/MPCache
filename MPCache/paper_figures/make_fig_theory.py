"""Generate Figure 4 (fig_theory.pdf): Empirical Verification of Prop. 1 & Thm. 1 on 928 GPU Traces."""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
VERIF = json.loads((ROOT / "MPCache" / "experiments_results" / "framework_verification.json").read_text())

OUT_DIRS = [
    Path(__file__).resolve().parent,
    ROOT / "paper" / "figures",
]

plt.rcParams.update({
    "font.size": 7.0,
    "axes.labelsize": 7.1,
    "axes.titlesize": 7.2,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.0, 2.05), constrained_layout=True)

# -------- Panel (a): Closed-Loop Adjoint Gain vs Solver-Only Proxy (192 GPU runs) --------
cm = VERIF["proposition1_adjoint_and_subspace"]["per_step_costmodel"]
steps = sorted(int(k) for k in cm.keys() if int(k) >= 3)
gains = np.array([cm[str(s)]["closed_loop_gain"] for s in steps])
c0 = np.array([cm[str(s)]["solver_only_c0"] for s in steps])
# Normalize both to 1.0 at step 3 to compare horizon trend
gains_norm = gains / gains[0]
c0_norm = c0 / c0[0]

ax1.plot(steps, gains_norm, "o-", color="#c0504d", lw=1.25, ms=3.4, label="Closed-loop $\\|\\Psi_{i\\to T}\\|$ (true)")
ax1.plot(steps, c0_norm, "s--", color="#7f7f7f", lw=1.05, ms=3.1, label="Solver-only prior $c_0$ (no $\\nabla_x f_\\omega$)")
ax1.set_title("(a) Adjoint Horizon Damping", pad=4, weight="bold")
ax1.set_xlabel("Denoising step $i$ (30-step grid)")
ax1.set_ylabel("Normalized gain (step 3 $= 1.0$)")
ax1.set_ylim(0.0, 1.32)
ax1.set_xticks([3, 7, 11, 15, 19, 23])
ax1.legend(loc="upper right", frameon=False, handlelength=1.6)
ax1.annotate(
    "$6.4\\times$ decay",
    xy=(23, gains_norm[-1]),
    xytext=(15.0, 0.45),
    fontsize=6.0,
    color="#c0504d",
    arrowprops=dict(arrowstyle="-|>", lw=0.6, color="#c0504d"),
)

# -------- Panel (b): Subspace Orthogonality Barrier (576 GPU runs, N ~ 17) --------
orc = VERIF["proposition1_adjoint_and_subspace"]["oracle_when_x_what_N17"]
x = np.arange(3)
w = 0.25
unif_vals = [orc["uniform"]["reuse"], orc["uniform"]["ex1"], orc["uniform"]["lsq"]]
# Compare Uniform vs Head-Dense (TeaCache/Oracle) across Reuse / 1st-Order / Oracle
ax2.bar(0 - w, orc["uniform"]["reuse"], width=w, color="#cfd6dd", label="Reuse ($\\alpha=0$)")
ax2.bar(0, orc["uniform"]["ex1"], width=w, color="#7f7f7f", label="1st-order extrap.")
ax2.bar(0 + w, orc["uniform"]["lsq"], width=w, color="#2f6fb3", label="Hindsight LSQ oracle")

ax2.bar(1 - w / 2, orc["teacache"]["reuse"], width=w, color="#cfd6dd")
ax2.bar(1 + w / 2, orc["teacache"]["ex1"], width=w, color="#7f7f7f")

ax2.bar(2 - w / 2, orc["othr"]["reuse"], width=w, color="#cfd6dd")
ax2.bar(2 + w / 2, orc["othr"]["ex1"], width=w, color="#c0504d")

ax2.axhline(orc["uniform"]["lsq"], color="#2f6fb3", lw=0.65, ls=":")
ax2.text(1.95, 0.44, "subspace floor (0.422)", fontsize=5.7, color="#2f6fb3", ha="right")
ax2.text(2 + w / 2, orc["othr"]["ex1"] + 0.015, "0.092", fontsize=5.8, color="#c0504d", ha="center", weight="bold")
ax2.set_title("(b) Subspace Barrier ($N{\\approx}17$)", pad=4, weight="bold")
ax2.set_xticks([0, 1, 2])
ax2.set_xticklabels(["Uniform\n($H{=}0$)", "Local\nHead-Dense", "Horizon\nHead-Dense"])
ax2.set_ylabel("Terminal latent rel-$L_2$")
ax2.set_ylim(0, 0.50)
ax2.legend(loc="center right", frameon=False, handlelength=1.1, bbox_to_anchor=(1.02, 0.66))

# -------- Panel (c): Super-Additive Age Compounding (160 GPU runs) --------
tc = VERIF["theorem1_age_dp_and_kkt_head"]["tail_compounding_step19"]
ms = [1, 2, 3, 5]
meas = [tc[str(m)]["measured_ratio"] for m in ms]
diff = [tc[str(m)]["diffusive_age_ratio"] for m in ms]
naiv = [tc[str(m)]["naive_additive_ratio"] for m in ms]

ax3.plot(ms, meas, "o", color="#2f6fb3", ms=4.3, zorder=4, label="Measured GPU trace")
ax3.plot(ms, diff, "-", color="#c0504d", lw=1.3, zorder=3, label="Age-quadrature ($\\beta{=}0.5$)")
ax3.plot(ms, naiv, "--", color="#7f7f7f", lw=1.05, zorder=2, label="Naive additive ($\\beta{=}0$)")
ax3.set_title("(c) Tail Gap Compounding", pad=4, weight="bold")
ax3.set_xlabel("Contiguous skipped steps  $g{-}1$")
ax3.set_ylabel("Terminal error / 1-skip error")
ax3.set_xticks([1, 2, 3, 4, 5])
ax3.set_ylim(0.5, 7.4)
ax3.legend(loc="upper left", frameon=False, handlelength=1.5)
ax3.text(4.92, 6.72, "6.56 vs 6.50", fontsize=5.8, color="#c0504d", ha="right")

for out_dir in OUT_DIRS:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig_theory.{ext}", dpi=300, bbox_inches="tight")
print("wrote fig_theory.pdf and fig_theory.png")
