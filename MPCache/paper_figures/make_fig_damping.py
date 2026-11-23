import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "lines.linewidth": 1.2, "lines.markersize": 3.2,
})

a49 = [0.0, 0.25, 0.5, 1.0]
l49 = [0.1549, 0.1214, 0.1041, 0.1176]
a17 = [0.0, 0.25, 0.5, 0.75, 1.0]
l17 = [0.1556, 0.1239, 0.1084, 0.1131, 0.1250]
gap49, gap17 = 0.4417, 0.4419

fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.6, 1.95),
                             gridspec_kw={"width_ratios": [2.55, 1.0], "wspace": 0.32},
                             constrained_layout=True)

ax.plot(a17, l17, "o-", color="#2f6fb3")
ax.plot(a49, l49, "s--", color="#c0504d")
ax.axhline(0.1549, color="#999999", lw=0.6, ls=":", zorder=0)
ax.text(0.98, 0.1572, "plain reuse", fontsize=6.2, color="#777777", ha="right")
ax.text(0.78, 0.1235, "17 frames", fontsize=6.4, color="#2f6fb3")
ax.text(0.78, 0.1043, "49 frames", fontsize=6.4, color="#c0504d")
ax.set_xlabel("damping of the one-step correction  $\\alpha$")
ax.set_ylabel("LPIPS against the full run")
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xlim(-0.03, 1.05)
ax.set_ylim(0.09, 0.17)
ax.annotate("", xy=(0.5, 0.1041), xytext=(0.5, 0.1505),
            arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#333333"))
ax.text(0.53, 0.134, "a third of\nthe error", fontsize=6.4, va="center")

ax2 = bx
labels = ["damped\n$\\alpha=0.5$", "scaled by\nthe gap"]
vals = [0.1041, gap49]
cols = ["#c0504d", "#7f7f7f"]
bars = ax2.bar([0, 1], vals, color=cols, width=0.55, linewidth=0)
for i, v in enumerate(vals):
    ax2.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=6.6)
ax2.set_xticks([0, 1]); ax2.set_xticklabels(labels)
ax2.set_ylim(0, 0.50)
ax2.set_ylabel("LPIPS, 49 frames")
ax2.text(0.5, 0.44, "$\\times 4.2$", ha="center", fontsize=7.2)
ax2.annotate("", xy=(1, 0.40), xytext=(0, 0.40),
             arrowprops=dict(arrowstyle="<|-|>", lw=0.6, color="#333333"))

for ext in ("pdf", "png"):
    fig.savefig(f"figures/fig_damping.{ext}", dpi=300, bbox_inches="tight")
print("wrote figures/fig_damping.pdf")
