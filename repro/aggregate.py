"""Merge chunked runs, score them with RA-CFGCache's own evaluation/eval_video.py,
and print our V100 numbers next to RA-CFGCache Table 1 (H200).

    conda activate racfg
    python repro/aggregate.py --model cogvideox [--n 32] [--skip_eval]

Fidelity = LPIPS(alex)/SSIM/PSNR of each accelerated video vs Vanilla(T=50) under the same
prompt + seed (paper protocol). Speedup = mean sampling latency of Vanilla50 / method,
computed only over prompts present for both, from the 'Avg sec / video sampling' lines.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

VIEWDIT = Path(os.environ.get("VIEWDIT", "/root/viewdit"))
RACFG = VIEWDIT / "repro/upstream/RA-CFGCache"
OUT_ROOT = Path(os.environ.get("OUT_ROOT", VIEWDIT / "results/repro"))

# RA-CFGCache Table 1 (NeurIPS'26, arXiv 2609.36433), H200 bf16, 100 VBench prompts.
# (LPIPS, SSIM, PSNR, speedup, latency_s)
PAPER = {
    "cogvideox": {
        "vanilla50": (0.0, 1.0, float("inf"), 1.00, 40.48),
        "vanilla25": (0.4751, 0.5445, 13.61, 1.86, 21.79),
        "teacache": (0.2040, 0.7538, 21.48, 2.42, 16.72),
        "taylorseer": None,  # not reported for CogVideoX in Table 1
        "racfg_slow": (0.0968, 0.8746, 26.93, 2.39, 16.96),
        "racfg_fast": (0.2052, 0.7614, 21.89, 3.13, 12.91),
    },
    "wan": {
        "vanilla50": (0.0, 1.0, float("inf"), 1.00, 93.32),
        "vanilla25": (0.4371, 0.5658, 15.87, 1.93, 48.41),
        "teacache": (0.1661, 0.7955, 23.08, 2.44, 38.24),
        "taylorseer": (0.3632, 0.6187, 16.96, 2.13, 43.79),
        "racfg_slow": (0.0993, 0.8778, 28.03, 2.56, 36.48),
        "racfg_fast": (0.1409, 0.8386, 26.12, 2.99, 31.25),
    },
}
LABEL = {
    "vanilla50": "Vanilla (T=50)", "vanilla25": "Vanilla (T=25)",
    "teacache": "TeaCache", "taylorseer": "TaylorSeer (N=3,O=1)",
    "racfg_slow": "RA-CFGCache-Slow", "racfg_fast": "RA-CFGCache-Fast",
}
TIME_RE = re.compile(r"(?:Avg sec / video sampling|Avg sampling sec/video):\s*([\d.]+)")
CHUNK_RE = re.compile(r"p(\d+)-(\d+)")


def merge(model: str, method: str, n: int):
    """Symlink chunk videos into one dir; return (videos_dir, {idx: latency_s})."""
    mdir = OUT_ROOT / model / method
    merged = OUT_ROOT / model / "_merged" / method / "videos"
    merged.mkdir(parents=True, exist_ok=True)
    lat = {}
    for chunk in sorted(mdir.glob("p*-*")):
        m = CHUNK_RE.fullmatch(chunk.name)
        if not m:
            continue
        log = OUT_ROOT / model / "_logs" / f"{method}_{chunk.name}.log"
        t = None
        if log.exists():
            hits = TIME_RE.findall(log.read_text(errors="ignore"))
            t = float(hits[-1]) if hits else None
        for v in sorted((chunk / "videos").glob("video_*.mp4")):
            idx = int(re.findall(r"\d+", v.stem)[-1])
            if idx >= n:
                continue
            dst = merged / v.name
            if not dst.exists():
                dst.symlink_to(v.resolve())
            if t is not None:
                lat[idx] = t  # chunk average (batch=1, so per-video mean of that chunk)
    return merged, lat


def evaluate(model, method, ref_dir, cmp_dir):
    out = OUT_ROOT / model / "_merged" / method / "metrics.json"
    env = dict(os.environ, HF_ENDPOINT=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        env.pop(k, None)
    cmd = [sys.executable, "evaluation/eval_video.py", "--ref_dir", str(ref_dir),
           "--cmp_dir", str(cmp_dir), "--prompt_file", "resources/prompts/prompt_video.txt",
           "--out_json", str(out), "--out_csv", str(out.with_suffix(".csv")),
           "--device", "cuda:0", "--no_imagereward"]
    subprocess.run(cmd, cwd=RACFG, env=env, check=True)
    return json.loads(out.read_text())["summary"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["cogvideox", "wan"])
    ap.add_argument("--n", type=int, default=100, help="use prompts [0, n)")
    ap.add_argument("--skip_eval", action="store_true", help="reuse existing metrics.json")
    args = ap.parse_args()

    ref_dir, ref_lat = merge(args.model, "vanilla50", args.n)
    n_ref = len(list(ref_dir.glob("*.mp4")))
    rows = []
    for method in LABEL:
        if not (OUT_ROOT / args.model / method).exists():
            continue
        cmp_dir, lat = merge(args.model, method, args.n)
        n_cmp = len(list(cmp_dir.glob("*.mp4")))
        common = sorted(set(lat) & set(ref_lat))
        sp = (sum(ref_lat[i] for i in common) / sum(lat[i] for i in common)) if common else float("nan")
        mean_lat = sum(lat[i] for i in common) / len(common) if common else float("nan")
        if method == "vanilla50":
            s = {"lpips_mean": 0.0, "ssim_mean": 1.0, "psnr_mean": float("inf"), "num_video_pairs": n_ref}
        else:
            mj = OUT_ROOT / args.model / "_merged" / method / "metrics.json"
            s = json.loads(mj.read_text())["summary"] if (args.skip_eval and mj.exists()) \
                else evaluate(args.model, method, ref_dir, cmp_dir)
        rows.append((method, s, sp, mean_lat, n_cmp))

    lines = [f"## {args.model}: V100 fp16 reproduction vs RA-CFGCache Table 1 (H200 bf16)", "",
             "| Method | n | LPIPS↓ ours / paper | SSIM↑ ours / paper | PSNR↑ ours / paper | Speedup ours / paper | Latency(s) ours / paper |",
             "|---|---|---|---|---|---|---|"]
    f = lambda x, d=4: "∞" if x == float("inf") else f"{x:.{d}f}"
    for method, s, sp, mlat, n in rows:
        p = PAPER[args.model].get(method)
        pp = (lambda i, d=4: f(p[i], d) if p else "–")
        lines.append(f"| {LABEL[method]} | {s.get('num_video_pairs', n)} | {f(s['lpips_mean'])} / {pp(0)} | "
                     f"{f(s['ssim_mean'])} / {pp(1)} | {f(s['psnr_mean'], 2)} / {pp(2, 2)} | "
                     f"{f(sp, 2)}× / {pp(3, 2)}× | {f(mlat, 1)} / {pp(4, 2)} |")
    md = "\n".join(lines)
    print(md)
    out = OUT_ROOT / args.model / "_merged" / f"table_n{args.n}.md"
    out.write_text(md + "\n", encoding="utf-8")
    print(f"\n[aggregate] wrote {out}")


if __name__ == "__main__":
    main()
