import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon

plt.rcParams.update({"font.size": 6.6})

fig, ax = plt.subplots(figsize=(7.0, 2.35))
ax.set_xlim(0, 100); ax.set_ylim(0, 34); ax.axis("off")

def box(x, y, w, h, text, fc="#ffffff", ec="#333333", fs=6.4, lw=0.7, color="#111111"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=0.7",
                                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=3, color=color)

def arrow(x0, y0, x1, y1, ls="-", col="#333333", rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6,
                                 linewidth=0.7, color=col, linestyle=ls, zorder=1,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1))

# ---------------- offline (top band) ----------------
ax.add_patch(Rectangle((0.5, 21.0), 99.0, 12.0, facecolor="#f2f6fc", edgecolor="none", zorder=0))
ax.text(1.6, 31.4, "A. once per sampler, offline", fontsize=6.8, color="#2f6fb3", weight="bold")
box(2.0, 22.6, 20.0, 6.4, "single-skip ablation\n(50 runs)")
box(25.5, 22.6, 18.0, 6.4, "cost profile\n$c(t)$")
box(47.0, 22.6, 20.0, 6.4, "allocation program\n(min calls s.t. cost)")
box(70.5, 22.6, 27.5, 6.4, "schedule $\\mathcal{S}$\nhead exact, then growing gaps")
arrow(22.3, 25.8, 25.3, 25.8); arrow(43.8, 25.8, 46.8, 25.8); arrow(67.3, 25.8, 70.3, 25.8)

# ---------------- online (bottom band) ----------------
ax.add_patch(Rectangle((0.5, 1.0), 99.0, 18.5, facecolor="#fdf3f1", edgecolor="none", zorder=0))
ax.text(1.6, 17.9, "B. once per video", fontsize=6.8, color="#c0504d", weight="bold")
box(2.0, 6.4, 17.0, 6.6, "step $t$\n(next)")
ax.add_patch(Polygon([[34.0, 13.6], [42.0, 9.7], [34.0, 5.8], [26.0, 9.7]], closed=True,
                     facecolor="#ffffff", edgecolor="#333333", linewidth=0.7, zorder=2))
ax.text(34.0, 9.7, "$t \\in \\mathcal{S}$ ?", ha="center", va="center", fontsize=6.4, zorder=3)
arrow(19.3, 9.7, 25.8, 9.7)
box(47.0, 13.2, 24.0, 5.8, "transformer call\nstore $(t, v_t)$")
box(47.0, 3.4, 24.0, 5.8, "reuse\n$v_s + \\alpha\\,(v_s - v_{s-1})$")
arrow(41.6, 11.8, 46.8, 15.4); arrow(41.6, 7.6, 46.8, 6.3)
ax.text(44.2, 15.9, "yes", fontsize=6.0, color="#666666")
ax.text(44.2, 4.0, "no", fontsize=6.0, color="#666666")
box(75.0, 8.3, 11.0, 5.6, "output\n$\\tilde v_t$", fc="#f7f7f7")
arrow(71.3, 16.1, 80.5, 14.1); arrow(71.3, 6.3, 80.5, 8.1)
# loop back: output -> step t, routed below the band
loop = dict(color="#666666", linewidth=0.6, linestyle=(0, (2.2, 1.6)))
ax.plot([80.5, 80.5], [8.3, 2.9], **loop)
ax.plot([80.5, 10.5], [2.9, 2.9], **loop)
ax.add_patch(FancyArrowPatch((10.5, 2.9), (10.5, 6.2), arrowstyle="-|>", mutation_scale=6,
                             linewidth=0.6, color="#666666", linestyle=(0, (2.2, 1.6)),
                             shrinkA=0, shrinkB=0, zorder=1))
ax.text(38.0, 1.35, "next step, until $t = T-1$", fontsize=5.8, color="#777777", ha="center")
# decode
box(89.0, 8.3, 9.5, 5.6, "decode", fc="#ffffff")
arrow(86.2, 11.1, 88.8, 11.1)

for ext in ("pdf", "png"):
    fig.savefig(f"figures/fig_arch.{ext}", dpi=300, bbox_inches="tight")
print("wrote figures/fig_arch.pdf")
