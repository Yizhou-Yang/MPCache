import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch

plt.rcParams.update({
    "font.size": 7.2, "axes.labelsize": 7.2, "axes.titlesize": 7.2,
    "xtick.labelsize": 6.6, "ytick.labelsize": 6.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "lines.linewidth": 0.9,
})

T = 50
PROF = [0.0,0.569,0.452,0.313,0.192,0.118,0.094,0.063,0.061,0.054,0.053,0.047,0.045,0.043,0.043,
        0.042,0.041,0.040,0.041,0.041,0.041,0.041,0.041,0.041,0.041,0.042,0.042,0.042,0.043,0.050]
prof = np.interp(np.linspace(0, len(PROF) - 1, T), np.arange(len(PROF)), PROF)
prof = prof / prof.max()

OURS = list(range(16)) + [18, 22, 28, 35, 42, 49]
UNIFORM = [int(round(i * (T - 1) / 21)) for i in range(22)]
INTERVAL = list(range(4)) + [4 + int(round(k * 45 / 18)) for k in range(1, 18)] + [49]

fig = plt.figure(figsize=(7.0, 2.15))
gs = fig.add_gridspec(1, 2, width_ratios=[2.45, 1.0], wspace=0.22,
                      left=0.055, right=0.985, top=0.90, bottom=0.16)

ax = fig.add_subplot(gs[0, 0])
# sensitivity band on top
for t in range(T):
    ax.add_patch(Rectangle((t - 0.5, 3.15), 1.0, 0.42,
                           facecolor=plt.cm.Blues(0.12 + 0.88 * prof[t]), edgecolor="none"))
ax.text(-1.6, 3.36, "cost of\na skip", fontsize=6.0, va="center", ha="right", color="#333333")

rows = [("ours", OURS, "#c0504d"), ("fixed interval", INTERVAL, "#7f7f7f"),
        ("uniform", UNIFORM, "#999999")]
for i, (name, steps, col) in enumerate(rows):
    y = 2.3 - i
    ax.plot([0, T - 1], [y, y], color="#dddddd", lw=0.8, zorder=0)
    ax.plot(steps, [y] * len(steps), "o", ms=2.6, color=col, zorder=2)
    ax.text(-1.6, y, name, fontsize=6.6, va="center", ha="right")

ax.set_xlim(-0.8, T - 0.2); ax.set_ylim(-0.75, 3.75)
ax.set_yticks([]); ax.set_xticks([0, 10, 20, 30, 40, 49])
ax.set_xlabel("denoising step")
ax.spines["left"].set_visible(False)
ax.text(24.5, 2.62, "22 calls, sixteen exact", fontsize=6.0, ha="center", color="#c0504d")

# right panel: what a skip returns
bx = fig.add_subplot(gs[0, 1])
vals = [0.1549, 0.4417, 0.1041]
names = ["reuse the\nlast output", "scale by\nthe gap", "damped\n(ours)"]
cols = ["#999999", "#7f7f7f", "#c0504d"]
bx.bar([0, 1, 2], vals, color=cols, width=0.62, linewidth=0)
for i, v in enumerate(vals):
    bx.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=6.4)
bx.set_xticks([0, 1, 2]); bx.set_xticklabels(names, fontsize=6.2)
bx.set_ylim(0, 0.52); bx.set_yticks([0, 0.2, 0.4])
bx.set_ylabel("LPIPS, same 22 calls")
bx.set_title("what a skipped step returns", fontsize=6.8, pad=4)

for ext in ("pdf", "png"):
    fig.savefig(f"figures/fig_teaser.{ext}", dpi=300, bbox_inches="tight")
print("wrote figures/fig_teaser.pdf")
