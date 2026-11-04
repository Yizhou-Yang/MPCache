"""Evaluate a cached run against its full reference with LPIPS-VGG.

Usage: python eval_lpips_vgg.py REF_DIR CACHE_DIR [--frames 10]
Both directories are searched recursively for *.mp4 and matched by basename.
"""
import argparse, glob, os, sys
import numpy as np, torch, imageio.v2 as iio, lpips, warnings
warnings.filterwarnings("ignore")

def index(d):
    return {os.path.basename(f): f for f in glob.glob(d + "/**/*.mp4", recursive=True)}

def frames(p, n):
    r = iio.get_reader(p); N = r.count_frames()
    idx = list(range(0, N, max(1, N // n)))[:n]
    v = np.stack([r.get_data(i) for i in idx]).astype(np.float32) / 127.5 - 1
    return torch.from_numpy(v).permute(0, 3, 1, 2).cuda()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref"); ap.add_argument("cache")
    ap.add_argument("--frames", type=int, default=10)
    a = ap.parse_args()
    L = lpips.LPIPS(net="vgg").cuda()
    ref, cur = index(a.ref), index(a.cache)
    keys = sorted(set(ref) & set(cur))
    if not keys:
        sys.exit("no overlapping videos")
    vals = []
    for k in keys:
        with torch.no_grad():
            vals.append(L(frames(ref[k], a.frames), frames(cur[k], a.frames)).mean().item())
    v = np.array(vals)
    print("n=%d mean=%.4f median=%.4f p90=%.4f" % (
        len(v), v.mean(), np.median(v), np.sort(v)[int(0.9 * (len(v) - 1))]))

if __name__ == "__main__":
    main()
