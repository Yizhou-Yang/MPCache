"""The three metrics the paper reports, computed on the same sampled frames.

Generic over backbones: it only needs two directories of videos and matches them by
basename. LPIPS uses a VGG backbone; on the same pair of videos AlexNet reports 0.041
where VGG reports 0.118, so the backbone is a parameter, not a detail.
"""

from __future__ import annotations
import glob, os


def index(directory: str) -> dict[str, str]:
    return {os.path.basename(f): f for f in glob.glob(directory + "/**/*.mp4", recursive=True)}


def frames(path: str, n: int):
    import imageio.v2 as iio
    import numpy as np
    r = iio.get_reader(path)
    total = r.count_frames()
    idx = list(range(0, total, max(1, total // n)))[:n]
    return np.stack([r.get_data(i) for i in idx]).astype("float32") / 255.0


def compare(ref_dir: str, arm_dir: str, n_frames: int = 10, backbone: str = "vgg") -> dict:
    import numpy as np, torch, lpips
    from skimage.metrics import structural_similarity as ssim
    L = lpips.LPIPS(net=backbone).cuda()
    ref, arm = index(ref_dir), index(arm_dir)
    keys = sorted(set(ref) & set(arm))
    if not keys:
        raise SystemExit("no overlapping videos")
    lp, ss, ps = [], [], []
    for k in keys:
        a, b = frames(ref[k], n_frames), frames(arm[k], n_frames)
        ta = torch.from_numpy(a).permute(0, 3, 1, 2) * 2 - 1
        tb = torch.from_numpy(b).permute(0, 3, 1, 2) * 2 - 1
        with torch.no_grad():
            lp.append(L(ta.cuda(), tb.cuda()).mean().item())
        ss.append(np.mean([ssim(a[i], b[i], channel_axis=2, data_range=1.0) for i in range(len(a))]))
        mse = float(np.mean((a - b) ** 2))
        ps.append(10 * np.log10(1.0 / max(mse, 1e-12)))
    return {"n": len(keys), "lpips": round(float(np.mean(lp)), 4),
            "ssim": round(float(np.mean(ss)), 4), "psnr": round(float(np.mean(ps)), 2)}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("ref"); ap.add_argument("arm")
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--backbone", default="vgg")
    a = ap.parse_args()
    print(compare(a.ref, a.arm, a.frames, a.backbone))


if __name__ == "__main__":
    main()
