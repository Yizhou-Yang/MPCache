import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

plt.rcParams.update({"font.size": 6.6, "axes.linewidth": 0.6})

fig, ax = plt.subplots(figsize=(3.5, 2.25))
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_xticks([]); ax.set_yticks([])
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

ax.annotate("", xy=(9.7, 1.6), xytext=(1.2, 1.6),
            arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#333333"))
ax.annotate("", xy=(1.2, 9.6), xytext=(1.2, 1.6),
            arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#333333"))
ax.text(5.45, 0.75, "how the exact calls are placed", ha="center", fontsize=6.6)
ax.text(0.0, 5.3, "what a skip returns", rotation=90, va="center", fontsize=6.6)
ax.text(1.45, 1.12, "local rule", fontsize=5.8, color="#8a949e")
ax.text(9.6, 1.12, "planned offline", fontsize=5.8, color="#8a949e", ha="right")
ax.text(1.45, 2.05, "reuse", fontsize=5.8, color="#8a949e")
ax.text(1.45, 8.15, "corrected", fontsize=5.8, color="#8a949e")

def dot(x, y, col, ms=4.4, ring=False):
    ax.plot(x, y, "o", ms=ms, color=col, zorder=3)
    if ring:
        ax.plot(x, y, "o", ms=ms + 3.4, mfc="none", mec=col, mew=0.9, zorder=3)

dot(3.1, 3.3, "#9aa4ad")
ax.text(3.1, 2.78, "TeaCache, RA-CFGCache", fontsize=5.7, color="#7f7f7f", ha="center", va="top")

dot(4.5, 6.5, "#9aa4ad")
ax.text(4.5, 7.05, "TaylorSeer", fontsize=5.7, color="#7f7f7f", ha="center", va="bottom")
ax.text(4.5, 5.95, "feature forecast", fontsize=5.5, color="#9aa4ad", ha="center", va="top")

dot(7.3, 3.3, "#2f6fb3")
ax.text(7.3, 2.78, "our schedule,\nplain reuse", fontsize=5.7, color="#2f6fb3", ha="center", va="top", linespacing=1.25)
ax.text(7.3, 3.85, "0.155", fontsize=5.7, color="#2f6fb3", ha="center", va="bottom")

dot(8.3, 7.5, "#c0504d", ms=5.2, ring=True)
ax.text(8.3, 8.35, "ours", fontsize=6.0, color="#c0504d", ha="center", va="bottom")
ax.text(8.3, 6.95, "0.104", fontsize=5.7, color="#c0504d", ha="center", va="top")

ax.add_patch(FancyArrowPatch((4.55, 3.3), (6.55, 3.3), arrowstyle="-|>", mutation_scale=6,
                             lw=0.6, color="#2f6fb3", linestyle=(0, (2, 1.5))))
ax.text(5.55, 3.55, "plan the calls", fontsize=5.5, color="#2f6fb3", ha="center", va="bottom")
ax.add_patch(FancyArrowPatch((7.45, 3.85), (8.15, 6.7), arrowstyle="-|>", mutation_scale=6,
                             lw=0.6, color="#c0504d", linestyle=(0, (2, 1.5))))
ax.text(7.15, 4.95, "damp the\ncorrection", fontsize=5.5, color="#c0504d", ha="right", va="center", linespacing=1.25)

for ext in ("pdf", "png"):
    fig.savefig(f"figures/fig_map.{ext}", dpi=300, bbox_inches="tight")
print("wrote figures/fig_map.pdf")
