"""The mechanism figure: why the correction must be damped, not scaled.

Panel (a) is the illustrative geometry sketch from ART_BRIEF.md section 2, redrawn with the
two minimax endpoints (stop / persist) made explicit instead of a single generic "long arrow".
Panel (b) is a data/derived-function panel, not a sketch: it plots the exact worst-case risk
max(alpha^2, (1-alpha)^2) from the minimax derivation in method.tex (Eq. minimax) and must not
be restyled by an illustrator beyond colour, same rule as fig_damping.pdf and fig_sensitivity.pdf.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np

RED, BLUE, GREY, RULE = "#c0504d", "#2f6fb3", "#7f7f7f", "#cfd6dd"

plt.rcParams.update({
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "lines.linewidth": 1.2, "lines.markersize": 3.2,
})

fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.6, 2.05),
                             gridspec_kw={"width_ratios": [1.2, 1.0], "wspace": 0.38},
                             constrained_layout=True)

# ---------------------------------------------------------------- panel (a)
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
ax.set_title("(a) two stored outputs fix a direction, not how far to trust it",
             fontsize=7.0, pad=3)

prev = np.array([0.3, 2.3])   # v_{s'}
anch = np.array([3.0, 4.0])   # v_s
delta = anch - prev           # Delta
normal = np.array([delta[1], -delta[0]])
normal = normal / np.linalg.norm(normal)

def true_point(k, curvature=0.55):
    """The unknown continuation at k steps' worth of Delta: a secant plus a drop that grows
    with the square of the distance, i.e. the trajectory curves away from its own tangent."""
    return anch + k * delta - curvature * k ** 2 * normal

# the unknown true continuation, drawn as a curve the method never gets to see
ks = np.linspace(0.0, 1.9, 120)
curve = np.array([true_point(k) for k in ks])
ax.plot(curve[:, 0], curve[:, 1], color=BLUE, lw=1.3, zorder=1, alpha=0.85)
ax.text(curve[-1, 0] + 0.15, curve[-1, 1] - 0.05, "unknown\ncontinuation",
        fontsize=5.8, color=BLUE, ha="left", va="center")

# the measured secant: prev -> anch
ax.plot([prev[0], anch[0]], [prev[1], anch[1]], color=GREY, lw=0.9, ls=(0, (2.5, 1.8)), zorder=1)
ax.plot(*prev, "o", ms=3.0, color=GREY, zorder=2)
ax.plot(*anch, "o", ms=3.4, color=GREY, zorder=2)
ax.text(prev[0], prev[1] - 0.55, "$v_{s'}$", fontsize=6.4, color="#555555", ha="center")
ax.text(anch[0] - 0.15, anch[1] + 0.45, "$v_s$", fontsize=6.4, color="#555555", ha="center")

# damped guess, alpha = 1/2: short red arrow, lands next to the truth
damp_tip = anch + 0.5 * delta
ax.add_patch(FancyArrowPatch(anch, damp_tip, arrowstyle="-|>", mutation_scale=7,
                             lw=1.5, color=RED, zorder=3))
truth_half = true_point(0.5)
ax.plot(*truth_half, "o", ms=3.6, mfc="#ffffff", mec=BLUE, mew=1.1, zorder=4)
ax.annotate("", xy=truth_half, xytext=damp_tip,
            arrowprops=dict(arrowstyle="-", lw=0.5, color="#999999", ls=":"), zorder=2)
ax.text(damp_tip[0] - 0.05, damp_tip[1] - 0.6, "damped\n$\\alpha=\\frac{1}{2}$  \u2713",
        fontsize=6.2, color=RED, ha="center", va="top")

# the two minimax endpoints and the gap-scaled failure, one long grey arrow with a waypoint tick
persist_tip = anch + 1.0 * delta
scaled_tip = anch + 2.0 * delta
ax.add_patch(FancyArrowPatch(anch, scaled_tip, arrowstyle="-|>", mutation_scale=7,
                             lw=1.1, color=GREY, zorder=2))
tick_dir = normal
ax.plot([persist_tip[0] - 0.12 * tick_dir[0], persist_tip[0] + 0.12 * tick_dir[0]],
        [persist_tip[1] - 0.12 * tick_dir[1], persist_tip[1] + 0.12 * tick_dir[1]],
        color="#555555", lw=1.0, zorder=3)
ax.text(persist_tip[0] - 0.1, persist_tip[1] + 0.42, "persist\n$\\alpha=1$",
        fontsize=5.8, color="#555555", ha="center", va="bottom")
ax.text(scaled_tip[0] + 0.15, scaled_tip[1] + 0.05, "\u2717 scaled by\nthe gap",
        fontsize=6.2, color=GREY, ha="left", va="center")

# the point v_s itself is the other minimax endpoint (alpha = 0, plain reuse)
ax.text(anch[0] - 0.1, anch[1] - 0.55, "reuse\n$\\alpha=0$", fontsize=5.8, color="#555555",
        ha="center", va="top")

# ---------------------------------------------------------------- panel (b)
bx.set_title("(b) worst case over an unknown stop/persist outcome", fontsize=7.0, pad=3)
alpha = np.linspace(0, 1, 200)
stop_branch = alpha ** 2            # error if the true output turns out to be v_s (stops)
persist_branch = (1 - alpha) ** 2   # error if it turns out to be v_s + Delta (persists)
worst = np.maximum(stop_branch, persist_branch)

bx.plot(alpha, stop_branch, color=GREY, lw=0.8, ls=(0, (3, 2)))
bx.plot(alpha, persist_branch, color=GREY, lw=0.8, ls=(0, (1, 1.6)))
bx.plot(alpha, worst, color=RED, lw=1.6)
bx.fill_between(alpha, worst, 1.0, color=RULE, alpha=0.5, lw=0)
bx.plot([0.5], [0.25], "o", ms=4.0, color=RED, zorder=5)
bx.axvline(0.5, color="#999999", lw=0.5, ls=":")
bx.annotate("minimax optimum\n$\\alpha=\\frac{1}{2},\\ \\ \\frac{1}{4}\\,\\Vert\\Delta\\Vert^2$",
            xy=(0.5, 0.25), xytext=(0.66, 0.84), ha="center", fontsize=6.2, color=RED,
            arrowprops=dict(arrowstyle="-", lw=0.5, color="#999999"))
bx.text(0.02, (0.02) ** 2 + 0.05, "stops\n($\\alpha^2$)", fontsize=5.8, color="#666666")
bx.text(0.78, (1 - 0.78) ** 2 + 0.05, "persists\n$((1-\\alpha)^2)$", fontsize=5.8, color="#666666")
bx.plot([0], [1], "o", ms=3.0, mfc="#ffffff", mec=GREY, mew=1.0)
bx.plot([1], [1], "o", ms=3.0, mfc="#ffffff", mec=GREY, mew=1.0)
bx.text(0.0, 1.05, "reuse", fontsize=5.8, color="#555555", ha="center")
bx.text(1.0, 1.05, "undamped", fontsize=5.8, color="#555555", ha="center")
bx.set_xlabel("damping of the one-step correction  $\\alpha$")
bx.set_ylabel("worst-case error  /  $\\Vert\\Delta\\Vert^2$")
bx.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
bx.set_xlim(-0.03, 1.03); bx.set_ylim(0, 1.12)

for ext in ("pdf", "png"):
    fig.savefig(f"figures/fig_mechanism.{ext}", dpi=300, bbox_inches="tight")
print("wrote figures/fig_mechanism.pdf")
