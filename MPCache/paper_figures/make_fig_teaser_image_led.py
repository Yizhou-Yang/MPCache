import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np, imageio.v2 as iio
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.size": 6.6, "axes.linewidth": 0.0, "font.family": "DejaVu Sans"})

SRC = ("/Users/yizhouyang/Desktop/论文和资料/ViewDiT/repro/results/"
       "20261030_cog_aligned/results/cog30/full/video_000004.mp4")
rd = iio.get_reader(SRC); N = rd.count_frames()
picks = [int(round(t * (N - 1))) for t in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)]
frames = [rd.get_data(i) for i in picks]

T = 50
PROF = [0.0,0.569,0.452,0.313,0.192,0.118,0.094,0.063,0.061,0.054,0.053,0.047,0.045,0.043,0.043,
        0.042,0.041,0.040,0.041,0.041,0.041,0.041,0.041,0.041,0.041,0.042,0.042,0.042,0.043,0.050]
prof = np.interp(np.linspace(0, len(PROF) - 1, T), np.arange(len(PROF)), PROF)
OURS = list(range(16)) + [18, 22, 28, 35, 42, 49]

fig = plt.figure(figsize=(7.1, 2.45))
gs = fig.add_gridspec(2, 2, width_ratios=[4.55, 1.85], height_ratios=[1.9, 1.0],
                      left=0.014, right=0.99, top=0.965, bottom=0.075, wspace=0.10, hspace=0.30)

# ---------- filmstrip ----------
axs = fig.add_subplot(gs[0, 0]); axs.set_xlim(0, 6); axs.set_ylim(0, 1); axs.axis("off")
for k, fr in enumerate(frames):
    axs.imshow(fr, extent=(k + 0.03, k + 0.97, 0.02, 0.98), aspect="auto", zorder=1)
    axs.add_patch(Rectangle((k + 0.03, 0.02), 0.94, 0.96, fill=False, edgecolor="#ffffff", lw=1.6, zorder=2))
axs.text(0.0, 1.02, "one clip, decoded from the full run", fontsize=6.2, color="#555555")

# ---------- timeline ----------
ax = fig.add_subplot(gs[1, 0])
ax.set_xlim(-2.0, T); ax.set_ylim(0.0, 6.2); ax.axis("off")
BAND_Y, DOT_Y = 2.55, 1.75
for t in range(T):
    ax.add_patch(Rectangle((t - 0.5, BAND_Y), 1.0, 0.42,
                           facecolor=plt.cm.Blues(0.10 + 0.85 * prof[t]), edgecolor="none"))
ax.plot([-0.5, T - 0.5], [DOT_Y, DOT_Y], color="#e3e8ee", lw=0.9, zorder=0)
ax.plot(OURS, [DOT_Y] * len(OURS), "o", ms=3.0, color="#c0504d", zorder=2)
ax.text(-1.4, BAND_Y + 0.21, "cost of a skip", fontsize=5.8, color="#555555", ha="right", va="center")
ax.text(-1.4, DOT_Y, "22 calls", fontsize=5.8, color="#c0504d", ha="right", va="center")
ax.text(0, 0.75, "step 0", fontsize=5.6, color="#8a949e", ha="center")
ax.text(T - 1, 0.75, "49", fontsize=5.6, color="#8a949e", ha="center")
ax.annotate("an error here is carried\nby every later step", xy=(3.0, BAND_Y + 0.5), xytext=(5.5, 5.35),
            fontsize=5.8, color="#333333", ha="left", va="center", linespacing=1.25,
            arrowprops=dict(arrowstyle="-", lw=0.5, color="#9aa4ad"))
ax.annotate("an error here is re-explained\nbefore the decode", xy=(33.0, BAND_Y + 0.5), xytext=(36.0, 5.35),
            fontsize=5.8, color="#333333", ha="left", va="center", linespacing=1.25,
            arrowprops=dict(arrowstyle="-", lw=0.5, color="#9aa4ad"))

# ---------- inset: profile ----------
bx = fig.add_subplot(gs[0, 1])
bx.set_xlim(0, 10); bx.set_ylim(0, 5.0); bx.axis("off")
bx.add_patch(FancyBboxPatch((0.2, 0.3), 9.6, 5.2, boxstyle="round,pad=0,rounding_size=0.4",
                            facecolor="#f7f9fc", edgecolor="#e3e8ee", lw=0.6))
xs = np.linspace(0.8, 9.2, T)
for i in range(T):
    bx.add_patch(Rectangle((xs[i] - 0.085, 0.75), 0.17, 0.3 + 2.6 * prof[i],
                           facecolor=plt.cm.Blues(0.25 + 0.7 * prof[i]), edgecolor="none"))
bx.text(0.6, 3.85, "0.57", fontsize=6.0, color="#2f6fb3")
bx.text(9.2, 1.15, "0.04", fontsize=6.0, color="#2f6fb3", ha="right")
bx.text(5.0, 0.42, "what one skip costs", fontsize=5.7, color="#555555", ha="center")

# ---------- inset: result ----------
cx = fig.add_subplot(gs[1, 1])
cx.set_xlim(0, 10); cx.set_ylim(0, 6.0); cx.axis("off")
cx.add_patch(FancyBboxPatch((0.2, 0.3), 9.6, 5.2, boxstyle="round,pad=0,rounding_size=0.4",
                            facecolor="#fdf5f4", edgecolor="#e3e8ee", lw=0.6))
vals, cols, labs = [0.1549, 0.1041, 0.4417], ["#a9b2ba", "#c0504d", "#7f7f7f"], ["reuse", "damp", "gap"]
for i, (v, c, l) in enumerate(zip(vals, cols, labs)):
    h = 2.6 * v / 0.45
    cx.add_patch(Rectangle((1.0 + i * 2.9, 1.05), 1.7, h, facecolor=c, edgecolor="none"))
    cx.text(1.85 + i * 2.9, 1.25 + h, f"{v:.3f}", ha="center", fontsize=5.6, color="#333333")
    cx.text(1.85 + i * 2.9, 0.72, l, ha="center", fontsize=5.6, color="#555555")
cx.text(5.0, 5.45, "LPIPS, same 22 calls", fontsize=5.7, color="#555555", ha="center")

for ext in ("pdf", "png"):
    fig.savefig(f"figures/fig_teaser2.{ext}", dpi=300, bbox_inches="tight")
print("wrote figures/fig_teaser2.pdf")
