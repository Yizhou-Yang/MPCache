"""Compose the qualitative grid: one row per prompt, one column per arm.

Arms are given as name=directory pairs. The frame index is a fraction of the clip, so
the same moment of the video is shown for every arm. Optional circles mark artefacts:
--circle row,col,x,y,r with x, y, r in fractions of the cell.

    python compose_qualitative.py --out fig_qual.pdf --frame 0.6 \
        --arm Original=out/full --arm TeaCache=out/tea --arm RA-CFGCache=out/ra \
        --arm Ours=out/ours --circle 0,2,0.72,0.30,0.10
"""
import argparse, os, glob
import numpy as np
import imageio.v2 as iio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--frame", type=float, default=0.6, help="fraction of the clip to sample")
ap.add_argument("--arm", action="append", default=[], help="name=dir")
ap.add_argument("--rows", type=int, default=3)
ap.add_argument("--labels", default="", help="comma separated sub-labels, e.g. speedups")
ap.add_argument("--circle", action="append", default=[], help="row,col,x,y,r in fractions")
ap.add_argument("--fontsize", type=float, default=8)
a = ap.parse_args()

arms = [tuple(x.split("=", 1)) for x in a.arm]
assert arms, "give at least one --arm name=dir"

def videos(d):
    fs = []
    for r, _, names in os.walk(d):
        fs += [os.path.join(r, n) for n in names if n.endswith(".mp4")]
    return sorted(fs)

cols = [videos(d) for _, d in arms]
n = min([len(c) for c in cols] + [a.rows])
n = min(n, a.rows)
if n == 0:
    raise SystemExit("no videos found")

fig, axes = plt.subplots(n, len(arms), figsize=(2.05 * len(arms), 1.22 * n),
                         squeeze=False, constrained_layout=True)
for i in range(n):
    for j, (name, _) in enumerate(arms):
        path = cols[j][i]
        rd = iio.get_reader(path)
        N = rd.count_frames()
        idx = min(N - 1, max(0, int(round(a.frame * (N - 1)))))
        frame = rd.get_data(idx)
        ax = axes[i][j]
        ax.imshow(frame)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.5); s.set_color("#cccccc")
        if i == 0:
            lab = name
            extra = a.labels.split(",")
            if len(extra) == len(arms) and extra[j].strip():
                lab = "%s (%s)" % (name, extra[j].strip())
            ax.set_title(lab, fontsize=a.fontsize, pad=3)
for spec in a.circle:
    r, c, x, y, rad = (float(t) for t in spec.split(","))
    ax = axes[int(r)][int(c)]
    ax.add_patch(Ellipse((x, y), rad * 2 * ax.bbox.width / max(1, ax.bbox.height), rad * 2,
                         fill=False, edgecolor="#c0392b", lw=1.0, transform=ax.transAxes, clip_on=False))
fig.savefig(a.out, dpi=300, bbox_inches="tight")
print("wrote", a.out)
