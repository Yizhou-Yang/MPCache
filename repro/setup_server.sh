#!/usr/bin/env bash
# Build the competitor-reproduction environment on the V100 server.
# Does NOT touch the main /root/miniconda3 base env used by our own experiments.
#
#   bash repro/setup_server.sh            # full setup (clone + env + weights)
#   SKIP_WAN=1 bash repro/setup_server.sh # skip the Wan2.1 download
set -euo pipefail

VIEWDIT=${VIEWDIT:-/root/viewdit}
UP=$VIEWDIT/repro/upstream
WEIGHTS=$VIEWDIT/weights
ENV_NAME=${ENV_NAME:-racfg}
PYPI=${PYPI:-https://mirrors.tencent.com/pypi/simple}

# Pinned upstream commits (keep in sync with repro/upstream submodules)
RACFG_COMMIT=e17d6a74ac2f7ca68d9ee9b8af8bfd9bc3886d8a

mkdir -p "$UP" "$VIEWDIT/logs"
cd "$UP"
[ -d RA-CFGCache/.git ] || git clone https://github.com/yiming-l21/RA-CFGCache.git
git -C RA-CFGCache fetch -q origin || true
git -C RA-CFGCache checkout -q "$RACFG_COMMIT"
# Official TeaCache / TaylorSeer repos are kept for reference / cross-checks only;
# the main table uses RA-CFGCache's re-implementations (that is what its Table 1 reports).
for spec in ali-vilab/TeaCache Shenyi-Z/TaylorSeer; do      # reference only -> non-fatal
  d=${spec#*/}
  for try in 1 2 3; do
    [ -d $d/.git ] && break
    rm -rf $d; git clone -q https://github.com/$spec.git && break
    echo "[setup] clone $spec failed (try $try)"; sleep 20
  done
done
echo "[setup] upstream commits:"
for d in RA-CFGCache TeaCache TaylorSeer; do echo "  $d $(git -C $d rev-parse HEAD 2>/dev/null || echo MISSING)"; done

# V100 has no native bf16 -> make the upstream dtype configurable (default unchanged = bf16).
# Profile "v100" additionally enables the 31.7 GiB memory adapters and the threshold fix.
/root/miniconda3/bin/python "$VIEWDIT/repro/patches/apply_repro_patch.py" "$UP/RA-CFGCache" v100

source /root/miniconda3/etc/profile.d/conda.sh
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

# torch 2.10 must still ship sm_70 kernels; otherwise fall back to the cu126 build of the same version
if ! python -c "import torch,sys; sys.exit(0 if 'sm_70' in torch.cuda.get_arch_list() else 1)"; then
  echo "[setup] default torch wheel lacks sm_70 -> reinstalling torch 2.10.0 cu126"
  python -m pip install --timeout 300 --force-reinstall torch==2.10.0 torchvision==0.25.0 \
      --index-url https://download.pytorch.org/whl/cu126
fi
python - <<'PY'
import torch, diffusers, transformers
print("[setup] torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_arch_list())
print("[setup] diffusers", diffusers.__version__, "transformers", transformers.__version__)
x = torch.randn(256, 256, device="cuda", dtype=torch.float16)
print("[setup] fp16 matmul ok:", torch.isfinite(x @ x).all().item())
PY

# LPIPS(alex) backbone weights are fetched lazily by torchvision; prefetch once
python -c "import lpips; lpips.LPIPS(net='alex')" >/dev/null 2>&1 && echo "[setup] lpips ok"

if [ "${SKIP_WAN:-0}" != "1" ] && [ ! -f "$WEIGHTS/Wan2.1-T2V-1.3B-Diffusers/model_index.json" ]; then
  echo "[setup] downloading Wan2.1-T2V-1.3B-Diffusers via hf-mirror"
  HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 python - <<PY
from huggingface_hub import snapshot_download
snapshot_download("Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                  local_dir="$WEIGHTS/Wan2.1-T2V-1.3B-Diffusers", max_workers=4)
PY
fi
ls "$WEIGHTS/Wan2.1-T2V-1.3B-Diffusers" 2>/dev/null || true
echo "[setup] DONE"
