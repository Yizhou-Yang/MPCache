"""Generate Figure 2 (fig_arch.pdf): Overview of the Bilevel MPCache Framework."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon

plt.rcParams.update({"font.size": 6.5, "pdf.fonttype": 42, "ps.fonttype": 42})

OUT_DIRS = [
    Path(__file__).resolve().parent,
    Path(__file__).resolve().parents[2] / "paper" / "figures",
]

fig, ax = plt.subplots(figsize=(7.0, 2.50))
ax.set_xlim(0, 100)
ax.set_ylim(0, 36)
ax.axis("off")


def box(x, y, w, h, text, fc="#ffffff", ec="#333333", fs=6.1, lw=0.75, color="#111111"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.28,rounding_size=0.7",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            zorder=2,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=3, color=color, linespacing=1.22)


def arrow(x0, y0, x1, y1, ls="-", col="#333333", rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=6,
            linewidth=0.75,
            color=col,
            linestyle=ls,
            zorder=1,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=1,
            shrinkB=1,
        )
    )


# ---------------- Stage I: Outer-Level Horizon Synthesis (top band) ----------------
ax.add_patch(Rectangle((0.5, 22.0), 99.0, 13.2, facecolor="#f2f6fc", edgecolor="#cfd6dd", linewidth=0.6, zorder=0))
ax.text(
    1.6,
    33.4,
    "Stage I (Outer Problem, Offline Once per Sampler): Terminal Impulse Identification & Bellman DP",
    fontsize=6.6,
    color="#2f6fb3",
    weight="bold",
)
box(1.8, 23.6, 20.5, 7.6, "Terminal Impulse Pass\n($\\mathcal{O}(T)$ single-skip runs,\nProp. 1)")
box(25.1, 23.6, 18.5, 7.6, "Debiased Profile $\\tilde{w}_j$\n(closed-loop $\\Psi_{j\\to T}$\n+ VAE decode)")
box(46.4, 23.6, 22.2, 7.6, "Age-Weighted Bellman DP\n$\\min_{|S|=B} \\mathcal{C}_{\\beta}(S)$\n(Thm. 1 & Thm. 2)", ec="#2f6fb3", lw=0.9)
box(71.4, 23.6, 26.8, 7.6, "Optimal Schedule $\\mathcal{S}^{\\star}$\nExact head $0..H^{\\star}(B){-}1$\n+ growing tail gaps", fc="#ffffff", ec="#c0504d", lw=0.95)
arrow(22.5, 27.4, 24.9, 27.4)
arrow(43.8, 27.4, 46.2, 27.4)
arrow(68.8, 27.4, 71.2, 27.4)

# Schedule feed-down arrow from Stage I to Stage II decision diamond
arrow(84.8, 23.4, 34.0, 14.8, ls=(0, (2.5, 1.6)), col="#c0504d", rad=-0.12)
ax.text(63.5, 20.2, "zero-overhead offline policy $\\mathcal{S}^{\\star}$ (scale-invariant, Thm. 2)", fontsize=5.7, color="#c0504d", ha="center")

# ---------------- Stage II: Inner-Level Causal Shrinkage (bottom band) ----------------
ax.add_patch(Rectangle((0.5, 0.8), 99.0, 18.6, facecolor="#fdf3f1", edgecolor="#cfd6dd", linewidth=0.6, zorder=0))
ax.text(
    1.6,
    17.8,
    "Stage II (Inner Problem, Online per Video): Two-Anchor Causal Midpoint Chord-Shrinkage",
    fontsize=6.6,
    color="#c0504d",
    weight="bold",
)
box(1.8, 6.4, 16.4, 6.8, "Guided Solver\nStep $t$, state $x_t$")
ax.add_patch(
    Polygon(
        [[33.2, 14.1], [41.6, 9.8], [33.2, 5.5], [24.8, 9.8]],
        closed=True,
        facecolor="#ffffff",
        edgecolor="#333333",
        linewidth=0.75,
        zorder=2,
    )
)
ax.text(33.2, 9.8, "$t \\in \\mathcal{S}^{\\star}$ ?", ha="center", va="center", fontsize=6.3, zorder=3)
arrow(18.4, 9.8, 24.6, 9.8)

box(46.2, 12.6, 25.8, 5.8, "Exact DiT Evaluation $v_t$\nupdate cache $(s', v_{s'}), (s, v_s)$")
box(
    46.2,
    3.5,
    25.8,
    6.2,
    "Midpoint Shrinkage (Thm. 3)\n$\\tilde{v}_t = v_s + \\alpha\\,(v_s - v_{s'})$",
    ec="#c0504d",
    lw=0.95,
)
arrow(41.2, 11.6, 46.0, 15.2)
arrow(41.2, 8.0, 46.0, 6.6)
ax.text(43.2, 15.3, "yes", fontsize=5.8, color="#555555")
ax.text(43.2, 4.7, "no", fontsize=5.8, color="#c0504d")

box(75.0, 6.9, 12.2, 5.8, "Solver Step\n$x_{t+1}(\\tilde{v}_t)$", fc="#f7f7f7")
arrow(72.2, 15.5, 81.1, 12.9)
arrow(72.2, 6.6, 81.1, 12.9 if False else 6.9)

# Clean routing to solver box
ax.lines.clear() if False else None

loop = dict(color="#666666", linewidth=0.65, linestyle=(0, (2.2, 1.6)))
ax.plot([81.1, 81.1], [6.9, 2.2], **loop)
ax.plot([81.1, 10.0], [2.2, 2.2], **loop)
ax.add_patch(
    FancyArrowPatch(
        (10.0, 2.2),
        (10.0, 6.2),
        arrowstyle="-|>",
        mutation_scale=6,
        linewidth=0.65,
        color="#666666",
        linestyle=(0, (2.2, 1.6)),
        shrinkA=0,
        shrinkB=0,
        zorder=1,
    )
)
ax.text(45.5, 1.15, "iterate $t = 0, \\dots, T-1$", fontsize=5.7, color="#666666", ha="center")

box(89.6, 6.9, 8.8, 5.8, "VAE\nDecode", fc="#ffffff")
arrow(87.4, 9.8, 89.4, 9.8)

for out_dir in OUT_DIRS:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig_arch.{ext}", dpi=300, bbox_inches="tight")
print("wrote fig_arch.pdf and fig_arch.png")
