import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

plt.rcParams.update({"font.size": 6.9, "axes.linewidth": 0.0})

ts = np.linspace(0, 1, 300)
curve = np.stack([0.9 + 7.4 * ts, 1.2 + 5.2 * ts - 3.0 * ts ** 2], axis=1)
at = lambda t: curve[int(round(t * (len(curve) - 1)))]

fig, ax = plt.subplots(figsize=(3.45, 2.0))
ax.set_xlim(0, 10); ax.set_ylim(0, 7.4); ax.axis("off")
ax.plot(curve[:, 0], curve[:, 1], color="#b9c2cb", lw=1.2, zorder=1)

prev, anch = at(0.16), at(0.40)
d = anch - prev
full, far = anch + d, anch + 2.2 * d
damp = anch + 0.5 * d

# the secant, extended
ax.plot([prev[0], far[0]], [prev[1], far[1]], color="#d7dde3", lw=0.8, zorder=0)

ax.plot(*prev, "o", ms=2.8, color="#8a949e", zorder=3)
ax.plot(*anch, "o", ms=3.6, color="#8a949e", zorder=3)
ax.plot(*full, "o", ms=2.8, color="#9aa4ad", zorder=3)
ax.plot(*far, "o", ms=2.8, color="#9aa4ad", zorder=3)

# the truth on the curve
for t in (0.55, 0.78):
    q = at(t)
    ax.plot(*q, "o", ms=2.8, mfc="#ffffff", mec="#2f6fb3", mew=1.0, zorder=3)

ax.text(prev[0] - 0.1, prev[1] - 0.55, "$v_{s-1}$", fontsize=6.2, color="#6b7683", ha="center")
ax.text(anch[0] - 0.15, anch[1] - 0.6, "$v_s$", fontsize=6.2, color="#6b7683", ha="center")

# the damped correction
ax.add_patch(FancyArrowPatch(anch, damp, arrowstyle="-|>", mutation_scale=8, lw=1.5,
                             color="#c0504d", zorder=4))
ax.plot(*damp, "o", ms=3.0, color="#c0504d", zorder=4)
ax.text(damp[0] - 0.2, damp[1] + 0.75, "$\\alpha=0.5$", fontsize=6.4, color="#c0504d", ha="right")

# the gap-scaled variant, continuing past the full step
ax.add_patch(FancyArrowPatch(full + 0.1 * d, far, arrowstyle="-|>", mutation_scale=7,
                             lw=0.9, color="#7f7f7f", linestyle=(0, (1.6, 1.2)), zorder=2))
ax.text(far[0] + 0.22, far[1] - 0.05, "gap", fontsize=6.2, color="#7f7f7f", va="center")
ax.text(far[0] + 0.05, far[1] + 0.45, "$\\times$", fontsize=8.0, color="#7f7f7f")

for ext in ("pdf", "png"):
    fig.savefig(f"figures/fig_geom.{ext}", dpi=300, bbox_inches="tight")
print("wrote figures/fig_geom.pdf")
