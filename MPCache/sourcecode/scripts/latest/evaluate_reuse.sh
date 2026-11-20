#!/usr/bin/env bash
# Evaluate every reuse policy in a directory tree against a reference, one row per policy.
# usage: evaluate_reuse.sh <ref_dir> <policies_root> [arms...]
set -euo pipefail
REF=${1:?ref}; ROOT=${2:?root}; shift 2
ARMS=${*:-"latest e05 e10 scaled"}
PY=${PYTHON:-python3}
$PY - "$REF" "$ROOT" $ARMS <<'PY'
import os, sys, numpy as np, torch, lpips, imageio.v2 as iio, warnings
from skimage.metrics import structural_similarity as ssim
warnings.filterwarnings("ignore")
def idx(d):
    o = {}
    for r, _, fs in os.walk(d):
        for f in fs:
            if f.endswith(".mp4"): o[os.path.basename(f)] = os.path.join(r, f)
    return o
def fr(p, n=10):
    r = iio.get_reader(p); N = r.count_frames(); ii = list(range(0, N, max(1, N // n)))[:n]
    return np.stack([r.get_data(i) for i in ii]).astype("float32") / 255.
ref, root, arms = sys.argv[1], sys.argv[2], sys.argv[3:]
L = lpips.LPIPS(net="vgg").cuda()
R = idx(ref)
print("%-8s %4s %8s %8s %8s" % ("policy", "n", "LPIPS", "SSIM", "PSNR"))
for a in arms:
    C = idx(os.path.join(root, a)); ks = sorted(set(R) & set(C))
    if not ks:
        print("%-8s %4d %8s" % (a, 0, "-")); continue
    lp, ss, ps = [], [], []
    for k in ks:
        va, vb = fr(R[k]), fr(C[k])
        ta = torch.from_numpy(va).permute(0, 3, 1, 2) * 2 - 1
        tb = torch.from_numpy(vb).permute(0, 3, 1, 2) * 2 - 1
        with torch.no_grad(): lp.append(L(ta.cuda(), tb.cuda()).mean().item())
        ss.append(np.mean([ssim(va[i], vb[i], channel_axis=2, data_range=1.) for i in range(len(va))]))
        ps.append(10 * np.log10(1. / max(float(np.mean((va - vb) ** 2)), 1e-12)))
    print("%-8s %4d %8.4f %8.4f %8.2f" % (a, len(ks), np.mean(lp), np.mean(ss), np.mean(ps)))
PY
