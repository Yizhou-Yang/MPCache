import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.size": 6.6, "axes.linewidth": 0.0, "font.family": "DejaVu Sans"})

T = 50
PROF = [0.0,0.569,0.452,0.313,0.192,0.118,0.094,0.063,0.061,0.054,0.053,0.047,0.045,0.043,0.043,
        0.042,0.041,0.040,0.041,0.041,0.041,0.041,0.041,0.041,0.041,0.042,0.042,0.042,0.043,0.050]
prof = np.interp(np.linspace(0, len(PROF) - 1, T), np.arange(len(PROF)), PROF)
OURS = list(range(16)) + [18, 22, 28, 35, 42, 49]

fig, axes = plt.subplots(1, 4, figsize=(7.1, 1.95),
                         gridspec_kw={"wspace": 0.16, "left": 0.012, "right": 0.988,
                                      "top": 0.93, "bottom": 0.12})
for ax in axes:
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.12, 0.12), 9.76, 9.76, boxstyle="round,pad=0,rounding_size=0.5",
                                facecolor="#ffffff", edgecolor="#cfd6dd", linewidth=0.7))
caps = ["fifty steps, one hundred\nforward passes", "the steps are not equal",
        "a skip must return\nsomething", "same 22 calls,\na third less error"]
for ax, c in zip(axes, caps):
    ax.text(5, 1.0, c, ha="center", va="center", fontsize=6.7, color="#333333", linespacing=1.35)

# panel 1: 50 steps, 22 exact
ax = axes[0]
w = 8.6 / T
for i in range(T):
    x = 0.7 + i * w
    on = i in OURS
    ax.add_patch(Rectangle((x, 6.0), w * 0.82, 1.5,
                           facecolor="#c0504d" if on else "#e6ebf0", edgecolor="none"))
ax.text(0.7, 5.5, "exact", fontsize=6.0, color="#c0504d")
ax.text(3.3, 5.5, "reused", fontsize=6.0, color="#8a949e")
ax.text(0.7, 4.4, "0", fontsize=5.6, color="#8a949e")
ax.text(9.05, 4.4, "49", fontsize=5.6, color="#8a949e", ha="right")
ax.text(5, 8.35, "22 of 50 calls", ha="center", fontsize=6.6, color="#333333")

# panel 2: the measured profile
ax = axes[1]
xs = np.linspace(0.8, 9.2, T)
for i in range(T):
    h = 0.35 + 5.3 * prof[i]
    ax.add_patch(Rectangle((xs[i] - 0.085, 3.3), 0.17, h,
                           facecolor=plt.cm.Blues(0.25 + 0.7 * prof[i]), edgecolor="none"))
ax.text(1.15, 9.05, "0.57", fontsize=6.2, color="#2f6fb3")
ax.text(9.15, 4.35, "0.04", fontsize=6.2, color="#2f6fb3", ha="right")
ax.plot([0.7, 9.3], [3.3, 3.3], color="#cfd6dd", lw=0.6)
ax.text(5, 2.55, "cost of skipping one step", ha="center", fontsize=6.1, color="#555555")

# panel 3: tangent vs damped
ax = axes[2]
prev, anch = np.array([1.6, 3.25]), np.array([3.6, 3.95])
tang = anch - prev
truth = anch + 0.45 * tang + np.array([0.10, -0.05])
ax.plot([prev[0], anch[0]], [prev[1], anch[1]], color="#9aa4ad", lw=1.0, zorder=1)
ax.plot(*prev, "o", ms=2.8, color="#9aa4ad")
ax.plot(*anch, "o", ms=3.4, color="#9aa4ad")
ax.add_patch(FancyArrowPatch(anch + 0.18 * tang, anch + 1.35 * tang, arrowstyle="-|>",
                             mutation_scale=7, lw=1.0, color="#7f7f7f", zorder=2))
ax.add_patch(FancyArrowPatch(anch, truth, arrowstyle="-|>", mutation_scale=7, lw=1.3,
                             color="#c0504d", zorder=3))
ax.plot(*truth, "o", ms=3.6, mfc="#ffffff", mec="#2f6fb3", mew=1.1, zorder=4)
ax.text(*(anch + 1.35 * tang + np.array([0.35, 0.15])), "\u2717", fontsize=8.0, color="#7f7f7f", ha="left", va="center")
ax.text(*(truth + np.array([0.30, 0.30])), "\u2713", fontsize=8.0, color="#c0504d", ha="left", va="center")
ax.text(2.2, 2.4, "\u2713 damped    \u2717 scaled", fontsize=6.2, color="#555555", ha="left")


# panel 4: the result
ax = axes[3]
vals, cols, labs = [0.1549, 0.1041, 0.4417], ["#a9b2ba", "#c0504d", "#7f7f7f"], ["reuse", "damped", "scaled"]
ax.text(5, 9.05, "LPIPS, 49 frames", ha="center", fontsize=6.2, color="#555555")
for i, (v, c, l) in enumerate(zip(vals, cols, labs)):
    h = 4.6 * v / 0.45
    ax.add_patch(Rectangle((1.55 + i * 2.55, 3.3), 1.75, h, facecolor=c, edgecolor="none"))
    ax.text(2.425 + i * 2.55, 3.3 + h + 0.35, f"{v:.3f}", ha="center", fontsize=6.1, color="#333333")
    ax.text(2.425 + i * 2.55, 2.75, l, ha="center", fontsize=6.1, color="#555555")
ax.plot([0.9, 9.1], [3.3, 3.3], color="#cfd6dd", lw=0.6)

for ext in ("pdf", "png"):
    fig.savefig(f"figures/fig_teaser.{ext}", dpi=300, bbox_inches="tight")
print("wrote figures/fig_teaser.pdf")
