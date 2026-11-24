"""Generate Figure 2 (fig_map.pdf): Bilevel Operator Design Plane for DiT Caching."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

plt.rcParams.update({
    "font.size": 6.3,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

OUT_DIRS = [
    Path(__file__).resolve().parent,
    Path(__file__).resolve().parents[2] / "paper" / "figures",
]

fig, ax = plt.subplots(figsize=(3.55, 2.38))
ax.set_xlim(0, 10.6)
ax.set_ylim(0, 10.4)
ax.set_xticks([])
ax.set_yticks([])
for s in ("top", "right", "left", "bottom"):
    ax.spines[s].set_visible(False)

ax.annotate("", xy=(10.3, 1.55), xytext=(1.25, 1.55), arrowprops=dict(arrowstyle="-|>", lw=0.75, color="#333333"))
ax.annotate("", xy=(1.25, 9.95), xytext=(1.25, 1.55), arrowprops=dict(arrowstyle="-|>", lw=0.75, color="#333333"))
ax.text(5.8, 0.50, "Outer Placement Operator $\\mathcal{S}$", ha="center", fontsize=6.4, weight="bold")
ax.text(0.18, 5.75, "Inner Reconstruction $\\mathcal{R}$", rotation=90, va="center", fontsize=6.4, weight="bold")
ax.text(1.42, 1.06, "local / periodic", fontsize=5.5, color="#7f7f7f")
ax.text(10.15, 1.06, "horizon Bellman DP", fontsize=5.5, color="#2f6fb3", ha="right")
ax.text(1.42, 2.00, "reuse ($\\alpha{=}0$)", fontsize=5.4, color="#7f7f7f")
ax.text(1.42, 9.35, "damped shrinkage", fontsize=5.4, color="#c0504d")


def dot(x, y, col, ms=4.5, ring=False):
    ax.plot(x, y, "o", ms=ms, color=col, zorder=3)
    if ring:
        ax.plot(x, y, "o", ms=ms + 3.6, mfc="none", mec=col, mew=0.95, zorder=3)


# Periodic + Gap-scaled Taylor
dot(2.85, 6.05, "#7f7f7f")
ax.text(2.85, 6.55, "TaylorSeer, ERTACache", fontsize=5.4, color="#555555", ha="center", va="bottom")
ax.text(2.85, 5.55, "gap-scaled ($0.44$--$0.49$)", fontsize=5.2, color="#7f7f7f", ha="center", va="top")

# Local greedy + plain reuse
dot(3.10, 3.15, "#9aa4ad")
ax.text(3.10, 2.62, "TeaCache, SenCache,\nRA-CFGCache", fontsize=5.3, color="#666666", ha="center", va="top", linespacing=1.16)

# Horizon DP + plain reuse
dot(7.95, 3.15, "#2f6fb3")
ax.text(7.95, 2.62, "DPCache, IdealCache,\nOurs ($\\alpha{=}0$)", fontsize=5.3, color="#2f6fb3", ha="center", va="top", linespacing=1.16)
ax.text(7.95, 3.66, "LPIPS 0.155", fontsize=5.4, color="#2f6fb3", ha="center", va="bottom")

# MPCache (Horizon DP + Midpoint Chord-Shrinkage)
dot(8.75, 8.15, "#c0504d", ms=5.3, ring=True)
ax.text(8.75, 8.95, "MPCache (Ours)", fontsize=5.9, color="#c0504d", ha="center", va="bottom", weight="bold")
ax.text(8.75, 7.52, "LPIPS 0.104", fontsize=5.5, color="#c0504d", ha="center", va="top", weight="bold")

ax.add_patch(
    FancyArrowPatch(
        (3.85, 3.15),
        (7.20, 3.15),
        arrowstyle="-|>",
        mutation_scale=6,
        lw=0.65,
        color="#2f6fb3",
        linestyle=(0, (2.2, 1.5)),
    )
)
ax.text(5.52, 3.38, "Thm. 1: KKT head\n+ interval DP", fontsize=5.2, color="#2f6fb3", ha="center", va="bottom", linespacing=1.14)

ax.add_patch(
    FancyArrowPatch(
        (8.10, 4.22),
        (8.62, 7.42),
        arrowstyle="-|>",
        mutation_scale=6,
        lw=0.65,
        color="#c0504d",
        linestyle=(0, (2.2, 1.5)),
    )
)
ax.text(7.95, 5.95, "Thm. 3: midpoint\nshrinkage ($-32.8\\%$)", fontsize=5.2, color="#c0504d", ha="right", va="center", linespacing=1.14)

for out_dir in OUT_DIRS:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig_map.{ext}", dpi=300, bbox_inches="tight")
print("wrote fig_map.pdf and fig_map.png")
