"""Generate Figure 5 (fig_damping.pdf): Midpoint Chord-Shrinkage & Hilbert Risk Law."""

from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "MPCache" / "sourcecode"))
from mpcache.shrinkage import fit_hilbert_quadratic

OUT_DIRS = [
    Path(__file__).resolve().parent,
    ROOT / "paper" / "figures",
]

plt.rcParams.update({
    "font.size": 7.4,
    "axes.labelsize": 7.4,
    "axes.titlesize": 7.4,
    "xtick.labelsize": 6.9,
    "ytick.labelsize": 6.9,
    "legend.fontsize": 6.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.25,
    "lines.markersize": 3.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

a49 = np.array([0.0, 0.25, 0.5, 1.0])
l49 = np.array([0.1549, 0.1214, 0.1041, 0.1176])
a17 = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
l17 = np.array([0.1556, 0.1239, 0.1084, 0.1131, 0.1250])
gap49 = 0.4417

fit49 = fit_hilbert_quadratic(a49, l49)
fit17 = fit_hilbert_quadratic(a17, l17)
grid = np.linspace(0.0, 1.02, 200)
curve49 = np.sqrt(fit49["c0"] + fit49["c1"] * grid + fit49["c2"] * grid ** 2)
curve17 = np.sqrt(fit17["c0"] + fit17["c1"] * grid + fit17["c2"] * grid ** 2)

fig, (ax, bx) = plt.subplots(
    1,
    2,
    figsize=(6.75, 2.05),
    gridspec_kw={"width_ratios": [2.55, 1.0], "wspace": 0.30},
    constrained_layout=True,
)

# Shaded theoretical midpoint quadrature basin [0.50, 0.65]
ax.axvspan(0.48, 0.66, color="#fdf3f1", zorder=0)
ax.text(0.57, 0.1635, "Thm. 3 midpoint\nbasin", fontsize=5.7, color="#c0504d", ha="center", va="top", linespacing=1.12)

ax.plot(grid, curve17, "-", color="#2f6fb3", lw=1.05, alpha=0.85, label="17f Hilbert law ($R^2{=}0.987$)")
ax.plot(a17, l17, "o", color="#2f6fb3", ms=3.8)

ax.plot(grid, curve49, "--", color="#c0504d", lw=1.2, label="49f Hilbert law ($R^2{=}0.997$)")
ax.plot(a49, l49, "s", color="#c0504d", ms=4.0)

ax.axhline(0.1549, color="#cfd6dd", lw=0.75, ls=":", zorder=0)
ax.text(1.01, 0.1565, "plain reuse ($\\alpha=0$)", fontsize=5.9, color="#7f7f7f", ha="right")

ax.set_xlabel("chord-shrinkage gain  $\\alpha$  in  $\\tilde{v}_j = v_s + \\alpha\\,(v_s - v_{s'})$")
ax.set_ylabel("VGG-LPIPS vs. 50-step full run")
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xlim(-0.04, 1.05)
ax.set_ylim(0.092, 0.168)
ax.legend(loc="lower left", frameon=False, handlelength=1.8, borderpad=0.1, labelspacing=0.2)

ax.annotate(
    "",
    xy=(0.5, 0.1045),
    xytext=(0.5, 0.1535),
    arrowprops=dict(arrowstyle="-|>", lw=0.75, color="#333333"),
)
ax.text(0.47, 0.132, "$-32.8\\%$\nerror", fontsize=6.0, color="#333333", ha="right", va="center", linespacing=1.12)

labels = ["midpoint\n$\\alpha=0.5$", "gap-scaled\n$(j{-}s)\\Delta_s$"]
vals = [0.1041, gap49]
cols = ["#c0504d", "#7f7f7f"]
bx.bar([0, 1], vals, color=cols, width=0.54, linewidth=0)
for i, v in enumerate(vals):
    bx.text(i, v + 0.013, f"{v:.3f}", ha="center", fontsize=6.5, weight="bold" if i == 0 else "normal")
bx.set_xticks([0, 1])
bx.set_xticklabels(labels, fontsize=6.3)
bx.set_ylim(0, 0.51)
bx.set_ylabel("VGG-LPIPS (49f, $n{=}100$)")
bx.text(0.5, 0.445, "$4.24\\times$ worse", ha="center", fontsize=6.6, color="#333333")
bx.annotate("", xy=(1, 0.405), xytext=(0, 0.405), arrowprops=dict(arrowstyle="<|-|>", lw=0.65, color="#333333"))

for out_dir in OUT_DIRS:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig_damping.{ext}", dpi=300, bbox_inches="tight")
print("wrote fig_damping.pdf and fig_damping.png")
