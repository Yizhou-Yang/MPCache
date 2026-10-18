"""Full-population video validation (not a sample).

Decodes the first / middle / last frame of EVERY video under results/repro_h20 (excluding the
_merged evaluation workspace) and flags near-black (mean <= 5), near-uniform (std <= 2), zero-frame,
or wrong-geometry files.

Runs in parallel; safe to leave unattended. Output: audit/VIDEO_VALIDITY_ALL.json
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path("/root/viewdit/results/repro_h20")
OUT = Path("/root/viewdit/audit/VIDEO_VALIDITY_ALL.json")
EXPECTED = {"cogvideox": (49, 720, 480), "wan": (81, 832, 480)}


def probe(path_str: str) -> dict:
    import cv2

    path = Path(path_str)
    model = path.relative_to(ROOT).parts[0]
    exp_frames, exp_w, exp_h = EXPECTED.get(model, (0, 0, 0))
    cap = cv2.VideoCapture(path_str)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    samples = []
    if frames > 0:
        for idx in (0, max(frames // 2, 0), max(frames - 1, 0)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            samples.append((float(gray.mean()), float(gray.std())))
    cap.release()
    black = any(m <= 5 or s <= 2 for m, s in samples)
    ok = frames > 0 and not black and len(samples) == 3 and width == exp_w and height == exp_h
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "frames": frames,
        "width": width,
        "height": height,
        "samples": [{"mean": round(m, 2), "std": round(s, 2)} for m, s in samples],
        "frames_ok": frames == exp_frames,
        "geometry_ok": (width, height) == (exp_w, exp_h),
        "ok": ok,
    }


def main() -> None:
    videos = [str(p) for p in sorted(ROOT.rglob("video_*.mp4")) if "_merged" not in p.parts]
    print(f"[validate-all] {len(videos)} videos", flush=True)
    workers = min(32, (os.cpu_count() or 8))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        details = list(pool.map(probe, videos, chunksize=8))

    summary, bad, frame_mismatch = {}, [], []
    for rec in details:
        key = "/".join(rec["path"].split("/")[:2])
        e = summary.setdefault(key, {"checked": 0, "ok": 0})
        e["checked"] += 1
        e["ok"] += int(rec["ok"])
        if not rec["ok"]:
            bad.append(rec)
        if not rec["frames_ok"]:
            frame_mismatch.append({"path": rec["path"], "frames": rec["frames"], "expected": EXPECTED[rec["path"].split("/")[0]][0]})

    payload = {
        "total": len(details),
        "passed": sum(1 for r in details if r["ok"]),
        "failed": len(bad),
        "frame_count_mismatch": len(frame_mismatch),
        "summary": summary,
        "failures": bad[:50],
        "frame_mismatch_sample": frame_mismatch[:50],
    }
    OUT.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(json.dumps({k: payload[k] for k in ("total", "passed", "failed", "frame_count_mismatch")}, indent=1))


if __name__ == "__main__":
    main()
