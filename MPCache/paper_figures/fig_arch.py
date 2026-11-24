"""Generate Figure 2 (fig_arch.pdf): Phase-Separated & Risk-Certified MPCache Architecture."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon

plt.rcParams.update({"font.size": 6.4, "pdf.fonttype": 42, "ps.fonttype": 42})

OUT_DIRS = [
    Path(__file__).resolve().parent,
    Path(__file__).resolve().parents[2] / "paper" / "figures",
]

fig, ax = plt.subplots(figsize=(7.0, 2.55))
ax.set_xlim(0, 100)
ax.set_ylim(0, 36)
ax.axis("off")


def box(x, y, w, h, text, fc="#ffffff", ec="#333333", fs=5.95, lw=0.75, color="#111111"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.25,rounding_size=0.7",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            zorder=2,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        zorder=3,
        color=color,
        linespacing=1.20,
    )


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


# ---------------- Stage I: Outer-Level Horizon Synthesis & Risk Certification ----------------
ax.add_patch(Rectangle((0.5, 22.0), 99.0, 13.3, facecolor="#f2f6fc", edgecolor="#cfd6dd", linewidth=0.6, zorder=0))
ax.text(
    1.5,
    33.5,
    "Stage I (Outer Horizon Synthesis & Risk Certification, Offline < 1.5 ms CPU)",
    fontsize=6.5,
    color="#2f6fb3",
    weight="bold",
)
box(1.5, 23.6, 21.0, 7.7, "Closed-Loop Impulse $w_j$\n($\\mathcal{O}(T)$ single-skip pass;\ncaptures $\\Psi_{j\\to T}$ & VAE)")
box(24.7, 23.6, 22.8, 7.7, "Boundary-Pinned Bellman DP\n$\\min_{0,T{-}1\\in S} \\mathcal{C}_{0.5}(S)$\nPareto table $\\{S^{\\star}_B\\}_{B=2}^T$ (Thm. 1)", ec="#2f6fb3", lw=0.9)
box(49.7, 23.6, 23.2, 7.7, "Conformal Budget Certifier\n$B^{\\star}(\\varepsilon, \\delta) = \\min\\{B:\\,\\hat{\\kappa}_{\\delta}\\rho(\\alpha) C \\leq \\varepsilon\\}$\n(Thm. 2 & Thm. 3)", ec="#2f6fb3", lw=0.9)
box(75.1, 23.6, 23.3, 7.7, "Phase-Separated $\\mathcal{S}^{\\star}$\nExact head $0..H^{\\star}{-}1$\n+ stiff-sloppy tail ($T{-}1\\in\\mathcal{S}^{\\star}$)", fc="#ffffff", ec="#c0504d", lw=0.95)
arrow(22.7, 27.45, 24.5, 27.45)
arrow(47.7, 27.45, 49.5, 27.45)
arrow(73.1, 27.45, 74.9, 27.45)

# Schedule feed-down arrow
arrow(86.7, 23.4, 34.0, 14.8, ls=(0, (2.5, 1.6)), col="#c0504d", rad=-0.11)
ax.text(63.2, 20.1, "scale-invariant schedule $\\mathcal{S}^{\\star}$ & bifurcation gate $H_{\\mathrm{crit}}$ (0 runtime overhead)", fontsize=5.65, color="#c0504d", ha="center")

# ---------------- Stage II: Inner-Level Phase-Gated Tangent Shrinkage ----------------
ax.add_patch(Rectangle((0.5, 0.8), 99.0, 18.6, facecolor="#fdf3f1", edgecolor="#cfd6dd", linewidth=0.6, zorder=0))
ax.text(
    1.5,
    17.8,
    "Stage II (Inner Output-Tangent Reconstruction, Online per Video: 2 Cache Slots, 0 Extra FLOPs)",
    fontsize=6.5,
    color="#c0504d",
    weight="bold",
)
box(1.6, 6.4, 16.6, 6.8, "Guided Solver\nStep $t$, state $x_t$\n(DDIM / UniPC)")
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
ax.text(33.2, 9.8, "$t \\in \\mathcal{S}^{\\star}$ ?", ha="center", va="center", fontsize=6.2, zorder=3)
arrow(18.4, 9.8, 24.6, 9.8)

box(46.0, 12.6, 26.2, 5.8, "Exact Guided DiT $v_t = f_\\omega(x_t, \\tau_t)$\nupdate 2-slot buffer $(s', v_{s'}), (s, v_s)$")
box(
    46.0,
    3.4,
    26.2,
    6.4,
    "Phase-Gated Midpoint Shrinkage (Thm. 4)\n$\\tilde{v}_t = v_s + \\alpha\\,\\mathbb{I}[t \\geq H_{\\mathrm{crit}}](v_s - v_{s'})$",
    ec="#c0504d",
    lw=0.95,
)
arrow(41.2, 11.6, 45.8, 15.2)
arrow(41.2, 8.0, 45.8, 6.6)
ax.text(43.1, 15.3, "yes", fontsize=5.7, color="#555555")
ax.text(43.1, 4.7, "no", fontsize=5.7, color="#c0504d")

box(75.0, 6.9, 12.2, 5.8, "Exact Solver\n$x_{t+1}(x_t, \\tilde{v}_t)$", fc="#f7f7f7")
arrow(72.4, 15.5, 81.1, 12.9)
arrow(72.4, 6.6, 81.1, 6.9)

loop = dict(color="#666666", linewidth=0.65, linestyle=(0, (2.2, 1.6)))
ax.plot([81.1, 81.1], [6.9, 2.2], **loop)
ax.plot([81.1, 9.9], [2.2, 2.2], **loop)
ax.add_patch(
    FancyArrowPatch(
        (9.9, 2.2),
        (9.9, 6.2),
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
ax.text(45.5, 1.15, "solver-bookkeeping-preserving loop $t = 0, \\dots, T-1$", fontsize=5.65, color="#666666", ha="center")

box(89.6, 6.9, 8.8, 5.8, "3D VAE\nDecode", fc="#ffffff")
arrow(87.4, 9.8, 89.4, 9.8)

for out_dir in OUT_DIRS:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig_arch.{ext}", dpi=300, bbox_inches="tight")
print("wrote fig_arch.pdf and fig_arch.png")
