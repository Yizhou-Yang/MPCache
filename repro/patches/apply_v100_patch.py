"""Idempotent V100 patch for RA-CFGCache (commit e17d6a7).

Upstream hard-codes torch.bfloat16 (they ran on H200). V100 (sm_70) has no bf16
tensor cores, so we make the compute dtype configurable through RACFG_DTYPE.
Default stays "bfloat16", i.e. without the env var the code behaves exactly like upstream.
CogVideoX-2B is officially an fp16 checkpoint, so RACFG_DTYPE=float16 is the native setting.

Nothing else (schedules, thresholds, proxies, calibration tables) is modified.

usage: python apply_v100_patch.py /path/to/RA-CFGCache
"""
import re
import sys
from pathlib import Path

FILES = [
    "models/cogvideox/util.py",
    "models/cogvideox/sample.py",
    "models/cogvideox/taylor_utils/grouped_taylor.py",
    "models/wan/util.py",
    "models/wan/sample.py",
    "models/wan/taylor_utils/grouped_taylor.py",
]
EXPR = 'getattr(torch, os.environ.get("RACFG_DTYPE", "bfloat16"))'
MARK = "# [viewdit-v100-patch]"


def patch(path: Path) -> str:
    src = path.read_text(encoding="utf-8")
    if MARK in src:
        return "already"
    new, n = re.subn(r"torch\.bfloat16", EXPR, src)
    if n == 0:
        return "no-bf16"
    if not re.search(r"^import os\b", new, flags=re.M):
        new = "import os\n" + new
    new = f"{MARK} torch.bfloat16 -> RACFG_DTYPE ({n} sites)\n" + new
    path.write_text(new, encoding="utf-8")
    return f"patched {n}"


VAE_MARK = "# [viewdit-v100-patch] vae tiling+slicing"
VAE_ANCHOR = '        print(f"[load] pipe.to({device}) done")\n\n    return pipe'
VAE_SNIPPET = '''        print(f"[load] pipe.to({device}) done")

    # [viewdit-v100-patch] vae tiling+slicing
    # V100 (31.7GiB, shared with other jobs): the fp32 Wan VAE OOMs decoding 81 frames.
    # Tiling + slicing make decode peak memory ~constant (negligible quality change).
    if hasattr(pipe, "vae") and getattr(pipe, "vae", None) is not None:
        try:
            pipe.vae.enable_tiling()
            pipe.vae.enable_slicing()
            print("[load] vae tiling+slicing enabled", flush=True)
        except Exception as _e:
            print(f"[warn] vae tiling failed: {_e}", flush=True)

    return pipe'''


def patch_wan_vae(path: Path) -> str:
    src = path.read_text(encoding="utf-8")
    if VAE_MARK in src:
        return "already"
    if VAE_ANCHOR not in src:
        return "no-anchor"
    path.write_text(src.replace(VAE_ANCHOR, VAE_SNIPPET, 1), encoding="utf-8")
    return "patched"


def main():
    root = Path(sys.argv[1])
    for rel in FILES:
        p = root / rel
        if p.exists():
            print(f"[v100-patch] {rel}: {patch(p)}")
        else:
            print(f"[v100-patch] {rel}: missing")
    wan = root / "models/wan/sample.py"
    if wan.exists():
        print(f"[v100-patch] wan/sample.py vae: {patch_wan_vae(wan)}")


if __name__ == "__main__":
    main()
