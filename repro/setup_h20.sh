#!/usr/bin/env bash
# Build the competitor-reproduction environment on the H20 (96 GiB Hopper) machine.
#
#   bash repro/setup_h20.sh              # preflight + clone + env + patch + weights
#   SKIP_WEIGHTS=1 bash repro/setup_h20.sh
#
# Profile "h20" applies ONLY patches that cannot change numerical behaviour:
#   * RACFG_DTYPE indirection, whose default is still "bfloat16" (identical to upstream)
#   * the CFGCache threshold binding fix (an upstream bug; see apply_repro_patch.py)
# No VAE tiling, no text-encoder release, no other adapter. On a 96 GiB card the paper's
# own bf16 configuration runs unmodified.
set -euo pipefail

VIEWDIT=${VIEWDIT:-/root/viewdit}
UP=$VIEWDIT/repro/upstream
WEIGHTS=$VIEWDIT/weights
ENV_NAME=${ENV_NAME:-racfg}
PYPI=${PYPI:-https://mirrors.tencent.com/pypi/simple}
HF=${HF_ENDPOINT:-https://hf-mirror.com}

RACFG_COMMIT=e17d6a74ac2f7ca68d9ee9b8af8bfd9bc3886d8a
CONDA_SH=${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}
CONDA_PY=${CONDA_PY:-/root/miniconda3/bin/python}

echo "================ preflight ================"
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader || {
  echo "[preflight] FATAL: nvidia-smi failed"; exit 1; }
[ -f "$CONDA_SH" ] || { echo "[preflight] FATAL: no conda at $CONDA_SH
  install miniconda or set CONDA_SH=/path/to/etc/profile.d/conda.sh"; exit 1; }
echo "[preflight] conda: $($CONDA_PY -V)"
echo "[preflight] disk free: $(df -h "$VIEWDIT" 2>/dev/null | tail -1)"
echo "[preflight] free RAM:  $(free -g | awk '/^Mem:/{print $7" GiB available"}')"
mkdir -p "$UP" "$WEIGHTS" "$VIEWDIT/logs" "$VIEWDIT/results"

echo "================ upstream (pinned) ================"
cd "$UP"
[ -d RA-CFGCache/.git ] || git clone https://github.com/yiming-l21/RA-CFGCache.git
git -C RA-CFGCache fetch -q origin || true
git -C RA-CFGCache checkout -q "$RACFG_COMMIT"
for spec in ali-vilab/TeaCache Shenyi-Z/TaylorSeer; do      # reference only -> non-fatal
  d=${spec#*/}
  for try in 1 2 3; do
    [ -d $d/.git ] && break
    rm -rf $d; git clone -q https://github.com/$spec.git && break
    echo "[setup] clone $spec failed (try $try)"; sleep 20
  done
done
for d in RA-CFGCache TeaCache TaylorSeer; do echo "  $d $(git -C $d rev-parse HEAD 2>/dev/null || echo MISSING)"; done

echo "================ patch (profile h20) ================"
"$CONDA_PY" "$VIEWDIT/repro/patches/apply_repro_patch.py" "$UP/RA-CFGCache" h20
"$CONDA_PY" - "$UP/RA-CFGCache" <<'PY'
import ast, sys
from pathlib import Path
root = Path(sys.argv[1])
for m in ("cogvideox", "wan"):
    p = root / "models" / m / "sample.py"
    src = p.read_text(); ast.parse(src)
    assert "cfgcache_thresh\": float(opts.rel_l1_thresh)" in src, f"{m}: threshold not bound"
    assert "RACFG_RELEASE_TEXT_ENCODER" not in src, f"{m}: memory adapter leaked into h20 tree"
    assert "enable_tiling" not in src, f"{m}: VAE tiling leaked into h20 tree"
    print(f"[verify] {m}/sample.py: parses, threshold bound, no adapter")
print("[verify] h20 tree differs from upstream only by the dtype indirection + threshold fix")
PY

echo "================ conda env ================"
source "$CONDA_SH"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -n "$ENV_NAME" python=3.10 -y
fi
conda activate "$ENV_NAME"
PIP="python -m pip install -i $PYPI --timeout 180"
$PIP -U pip setuptools wheel
$PIP -r "$UP/RA-CFGCache/requirements.txt"
$PIP -r "$UP/RA-CFGCache/evaluation/requirements.txt"
python -m pip uninstall -y opencv-python opencv-contrib-python >/dev/null 2>&1 || true
$PIP --force-reinstall --no-deps opencv-python-headless==4.13.0.92
(cd "$UP/RA-CFGCache" && python -m pip install -e . --no-deps)

echo "================ torch / bf16 check (sm_90) ================"
if ! python -c "import torch,sys; sys.exit(0 if 'sm_90' in torch.cuda.get_arch_list() else 1)"; then
  echo "[setup] wheel lacks sm_90 -> reinstalling torch 2.10.0 cu128"
  python -m pip install --timeout 300 --force-reinstall torch==2.10.0 torchvision==0.25.0 \
      --index-url https://download.pytorch.org/whl/cu128
fi
python - <<'PY'
import torch, diffusers, transformers
print("[setup] torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_arch_list())
print("[setup] diffusers", diffusers.__version__, "transformers", transformers.__version__)
print("[setup] device", torch.cuda.get_device_name(0), "cap", torch.cuda.get_device_capability(0))
assert torch.cuda.is_bf16_supported(), "bf16 unsupported on this device"
x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
assert torch.isfinite(x @ x).all().item()
print("[setup] bf16 matmul ok on sm_90")
PY
python -c "import lpips; lpips.LPIPS(net='alex')" >/dev/null 2>&1 && echo "[setup] lpips ok"

echo "================ weights ================"
if [ "${SKIP_WEIGHTS:-0}" != "1" ]; then
  HF_ENDPOINT=$HF HF_HUB_DISABLE_XET=1 python - <<PY
import os
from huggingface_hub import snapshot_download

def get(repo, dest, tries=30):
    from pathlib import Path
    if (Path(dest) / "model_index.json").exists():
        print(f"[setup] {dest} already present"); return
    for i in range(1, tries + 1):
        try:
            snapshot_download(repo, local_dir=dest, max_workers=4)
            print(f"[setup] {repo} done"); return
        except Exception as exc:
            print(f"[setup] {repo} attempt {i}/{tries} failed: {type(exc).__name__}: {exc}", flush=True)
    raise SystemExit(f"[setup] FATAL: could not download {repo}")

for repo in ("THUDM/CogVideoX-2b", "zai-org/CogVideoX-2b"):
    try:
        get(repo, "$WEIGHTS/CogVideoX-2b")
        break
    except SystemExit as exc:
        print(exc)
else:
    raise SystemExit("[setup] FATAL: CogVideoX-2b unavailable on the mirror")
get("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "$WEIGHTS/Wan2.1-T2V-1.3B-Diffusers")
PY
fi

echo "================ protocol inputs ================"
python - "$UP/RA-CFGCache/resources/prompts/prompt_video.txt" <<'PY'
import sys
from pathlib import Path
p = Path(sys.argv[1])
n = sum(bool(x.strip()) for x in p.read_text().splitlines())
print(f"[setup] prompt_video.txt nonempty prompts: {n}")
if n < 100:
    raise SystemExit(f"[setup] FATAL: expected 100 nonempty prompts, found {n}")
PY
for w in CogVideoX-2b Wan2.1-T2V-1.3B-Diffusers; do
  echo "[setup] $w: $(du -sh "$WEIGHTS/$w" 2>/dev/null | cut -f1) $(ls "$WEIGHTS/$w/model_index.json" 2>/dev/null >/dev/null && echo OK || echo MISSING)"
done
echo "[setup] DONE"
