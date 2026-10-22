#!/usr/bin/env python3
"""LPIPS / SSIM / PSNR between two directories of video_%06d.mp4.

Only the three metrics the paper reports; no CLIP, no ImageReward, no offline-HF issues.

usage:
  python metrics.py --ref DIR --cmp DIR [--max-frames 16] [--out-json x.json] [--out-csv x.csv]

Frames are aligned by linspace over min(len(ref), len(cmp)) so a different frame count
does not silently compare frame 0 to frame 80.
"""
import argparse, json, re
from pathlib import Path

import numpy as np
import torch
import imageio.v3 as iio
import lpips
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn


def vids(d):
    return {int(re.search(r'(\d+)', p.stem).group(1)): p
            for p in Path(d).glob('video_*.mp4')}


def frames(p, max_frames):
    a = iio.imread(p, plugin='FFMPEG')          # (T,H,W,3) uint8
    t = a.shape[0]
    if max_frames and t > max_frames:
        idx = np.linspace(0, t - 1, max_frames).round().astype(int)
        a = a[idx]
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', required=True)
    ap.add_argument('--cmp', required=True)
    ap.add_argument('--max-frames', type=int, default=0, help='0 = all frames (matches RA eval_video.py)')
    ap.add_argument('--out-json', default='')
    ap.add_argument('--out-csv', default='')
    a = ap.parse_args()

    ref, cmp = vids(a.ref), vids(a.cmp)
    keys = sorted(set(ref) & set(cmp))
    if not keys:
        raise SystemExit(f'no matched video_XXXXXX.mp4 between {a.ref} and {a.cmp}')

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = lpips.LPIPS(net='alex').to(dev).eval()

    rows = []
    with torch.no_grad():
        for k in keys:
            x = frames(ref[k], a.max_frames)
            y = frames(cmp[k], a.max_frames)
            n = min(len(x), len(y))
            if a.max_frames and a.max_frames > 0:
                idx = np.linspace(0, n - 1, min(n, a.max_frames)).round().astype(int)
            else:
                idx = np.arange(n)
            x, y = x[idx], y[idx]

            xf = torch.from_numpy(x).permute(0, 3, 1, 2).float().to(dev) / 127.5 - 1
            yf = torch.from_numpy(y).permute(0, 3, 1, 2).float().to(dev) / 127.5 - 1
            l = float(net(xf, yf).mean())

            ss, ps = [], []
            for i in range(len(x)):
                ss.append(ssim_fn(x[i], y[i], channel_axis=2, data_range=255))
                ps.append(psnr_fn(x[i], y[i], data_range=255))
            rows.append(dict(index=k, lpips=l, ssim=float(np.mean(ss)), psnr=float(np.mean(ps))))

    def mean(key):
        return float(np.mean([r[key] for r in rows]))

    summary = dict(n=len(rows), lpips_mean=mean('lpips'), ssim_mean=mean('ssim'),
                   psnr_mean=mean('psnr'),
                   ref=a.ref, cmp=a.cmp, max_frames=a.max_frames)
    print(json.dumps(summary, indent=2))
    if a.out_json:
        Path(a.out_json).write_text(json.dumps(dict(summary=summary, rows=rows), indent=2))
    if a.out_csv:
        with open(a.out_csv, 'w') as f:
            f.write('index,lpips,ssim,psnr\n')
            for r in rows:
                f.write(f"{r['index']},{r['lpips']:.6f},{r['ssim']:.6f},{r['psnr']:.4f}\n")


if __name__ == '__main__':
    main()
