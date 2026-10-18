"""Build an auditable bundle from reproduction artifacts.

Outputs under <VIEWDIT>/audit/:
  MANIFEST.json     sha256 + size for EVERY artifact, including videos that are not tracked by git
  ENVIRONMENT.json  hardware / software / pinned-source fingerprint
  TRACE_STATS.json  per-log CFGCache action counts and per-prompt wall-clock timings
  logs_clean/       logs with ANSI escapes and tqdm progress noise removed (content otherwise intact)

The manifest lets a reviewer verify downloaded videos against the exact bytes this run produced,
without having to re-execute the pipeline.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(os.environ.get("VIEWDIT", "/root/viewdit"))
RES = ROOT / "results" / "repro_h20"
OUT = ROOT / "audit"

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# tqdm progress bars are in-line noise; strip the bar prefix but KEEP whatever follows it
# (the CFGCache trace is frequently appended to the bar on the same physical line).
TQDM_BAR = re.compile(r".*?\d+%\|[^|]*\|\s*\d+/\d+\s*\[[^\]]*\]")
LOAD_NOISE = re.compile(r"^(Loading weights:|Loading checkpoint shards:|Loading pipeline components|Fetching \d+ files)")

ACTION_RE = re.compile(r"action=(\w+)")
STEP_RE = re.compile(r"CFGCache step (\d+):")
SAVE_RE = re.compile(r"Saving time:\s+([0-9.]+)")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sh(cmd: str):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def main() -> None:
    if not RES.is_dir():
        raise SystemExit(f"missing {RES}")
    OUT.mkdir(exist_ok=True)
    clean_dir = OUT / "logs_clean"
    clean_dir.mkdir(exist_ok=True)

    files = []
    for path in sorted(RES.rglob("*")):
        if not path.is_file():
            continue
        files.append({
            "path": path.relative_to(RES).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256(path),
        })
    manifest = {
        "results_root": str(RES),
        "file_count": len(files),
        "total_bytes": sum(f["size"] for f in files),
        "files": files,
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))

    stats = {}
    for log in sorted(RES.rglob("*.log")):
        rel = log.relative_to(RES).as_posix()
        kept, actions, timings = [], {}, []
        steps = 0
        for raw in log.read_text(errors="replace").splitlines():
            line = ANSI.sub("", raw).replace("\r", "").strip()
            line = TQDM_BAR.sub("", line).strip()
            if not line or LOAD_NOISE.match(line):
                continue
            kept.append(line)
            m = ACTION_RE.search(line)
            if m:
                actions[m.group(1)] = actions.get(m.group(1), 0) + 1
            if STEP_RE.search(line):
                steps += 1
            m = SAVE_RE.search(line)
            if m:
                timings.append(float(m.group(1)))
        dst = clean_dir / rel.replace("/", "__")
        dst.write_text("\n".join(kept) + "\n")
        timings_sorted = sorted(timings)
        stats[rel] = {
            "kept_lines": len(kept),
            "cfgcache_steps": steps,
            "actions": actions,
            "prompts_timed": len(timings),
            "timing_sum_s": round(sum(timings), 3) if timings else None,
            "timing_median_s": timings_sorted[len(timings_sorted) // 2] if timings_sorted else None,
            "timing_min_s": min(timings_sorted) if timings_sorted else None,
            "timing_max_s": max(timings_sorted) if timings_sorted else None,
        }
    (OUT / "TRACE_STATS.json").write_text(json.dumps(stats, indent=1, sort_keys=True))

    py = "/root/miniconda3/envs/racfg/bin/python"
    env = {
        "generated_at": sh("date -Iseconds"),
        "hostname": sh("hostname"),
        "gpus": sh("nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv,noheader"),
        "cuda_version": sh("nvidia-smi | grep -oE 'CUDA Version: [0-9.]+'"),
        "nproc": sh("nproc"),
        "mem_total_gb": sh("free -g | awk '/^Mem:/{print $2}'"),
        "disk": sh("df -h / | tail -1"),
        "python": sh(f"{py} -V"),
        "packages": sh(
            f"{py} -m pip list 2>/dev/null | grep -iE "
            "'^(torch|torchvision|diffusers|transformers|accelerate|opencv-python-headless|lpips|numpy|safetensors|huggingface-hub) '"
        ),
        "torch_cuda_arch_list": sh(f"{py} -c \"import torch;print(torch.cuda.get_arch_list())\""),
        "bf16_supported": sh(f"{py} -c \"import torch;print(torch.cuda.is_bf16_supported())\""),
        "source_revisions": {
            "RA-CFGCache": sh(f"git -C {ROOT}/repro/upstream/RA-CFGCache rev-parse HEAD"),
            "TeaCache": sh(f"git -C {ROOT}/repro/upstream/TeaCache rev-parse HEAD"),
            "TaylorSeer": sh(f"git -C {ROOT}/repro/upstream/TaylorSeer rev-parse HEAD"),
        },
        "prompt_file": {
            "path": "repro/upstream/RA-CFGCache/resources/prompts/prompt_video.txt",
            "nonempty_lines": sh(
                f"{py} -c \"from pathlib import Path;"
                f"print(sum(bool(x.strip()) for x in Path('{ROOT}/repro/upstream/RA-CFGCache/resources/prompts/prompt_video.txt').read_text().splitlines()))\""
            ),
        },
        "protocol": {
            "profile": "h20",
            "dtype": "bfloat16",
            "cpu_offload": False,
            "vae_tiling": False,
            "text_encoder_parking": False,
            "steps": 50,
            "negative_prompt": "animation",
            "batch_size": 1,
            "seed_rule": "global prompt id",
        },
    }
    (OUT / "ENVIRONMENT.json").write_text(json.dumps(env, indent=1, sort_keys=True))
    print(json.dumps({"artifacts": len(files), "logs": len(stats), "total_bytes": manifest["total_bytes"]}))


if __name__ == "__main__":
    main()
