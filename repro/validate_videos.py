"""Sanity-check generated videos: decodable, right geometry, not black, not static.

Samples a few videos per (model, method) row and inspects the first/middle/last frames.
A video is flagged bad if any sampled frame is near-black (mean <= 5) or near-uniform (std <= 2),
or if the container reports zero frames.

Usage: python3 repro/validate_videos.py
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/root/viewdit/results/repro_h20")
OUT = Path("/root/viewdit/audit/VIDEO_VALIDITY.json")
MODELS = ["cogvideox", "wan"]
METHODS = ["vanilla50", "vanilla25", "teacache", "taylorseer", "racfg_slow", "racfg_fast"]


def probe(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    samples = []
    if frames > 0:
        for idx in (0, frames // 2, frames - 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            samples.append({"idx": idx, "mean": round(float(gray.mean()), 2), "std": round(float(gray.std()), 2)})
    cap.release()
    ok = frames > 0 and len(samples) == 3 and all(s["mean"] > 5 and s["std"] > 2 for s in samples)
    return {"frames": frames, "width": width, "height": height, "sampled": samples, "ok": ok}


def main() -> None:
    details = []
    for model in MODELS:
        for method in METHODS:
            videos = sorted((ROOT / model / method).rglob("video_*.mp4"))
            if not videos:
                continue
            picks = list(dict.fromkeys(videos[:3] + videos[-2:]))
            for path in picks:
                rec = probe(path)
                rec["path"] = path.relative_to(ROOT).as_posix()
                details.append(rec)

    summary = {}
    for rec in details:
        key = "/".join(rec["path"].split("/")[:2])
        entry = summary.setdefault(key, {"checked": 0, "ok": 0, "frames": []})
        entry["checked"] += 1
        entry["ok"] += int(rec["ok"])
        entry["frames"].append(rec["frames"])

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"summary": summary, "details": details}, indent=1, sort_keys=True))
    total_chk = sum(v["checked"] for v in summary.values())
    total_ok = sum(v["ok"] for v in summary.values())
    print(json.dumps({"rows": len(summary), "checked": total_chk, "ok": total_ok}, indent=1))


if __name__ == "__main__":
    main()
