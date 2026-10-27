#!/usr/bin/env bash
# One-shot bootstrap for a fresh H20 host running the MPCache reproduction.
# Usage:  bash bootstrap_h20.sh [/root/viewdit]
# Idempotent: safe to re-run after a machine is reclaimed and re-attached.
#
# Order matters and is the whole point of this script:
#   1) conda env (fresh H20 has NO miniconda and NO accepted channel TOS)
#   2) torch 2.5.1+cu121 (driver 535 cannot load cu124/cu128 wheels, see PITFALLS P13)
#   3) everything else that pulls torch as a dependency
#   4) nvidia-cublas-cu12==12.4.5.8 LAST -- any later pip run silently downgrades it
#      back to 12.1.3.1 and every small-M bf16 GEMM SIGFPEs (P18)
#   5) weights through hf-mirror, resumable
#   6) verification: crash shapes + a real Wan video (mean/std), not just "it ran"
set -uo pipefail
ROOT="${1:-/root/viewdit}"
PY=/root/miniconda3/envs/racfg/bin/python
PIP=/root/miniconda3/envs/racfg/bin/pip
CONDA=/root/miniconda3/bin/conda
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=60

echo "== 0. miniconda =="
if [ ! -x "$CONDA" ]; then
  curl -fsSL https://mirrors.cloud.tencent.com/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/mc.sh
  bash /tmp/mc.sh -b -p /root/miniconda3
fi
# Fresh installs refuse to create envs until the channel TOS is accepted.
"$CONDA" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
"$CONDA" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true

echo "== 1. python env =="
if [ ! -x "$PY" ]; then
  echo "creating conda env racfg (python 3.10)"
  "$CONDA" create -y -n racfg python=3.10 || exit 1
fi
"$PY" -V

echo "== 2. torch 2.5.1+cu121 =="
$PIP install -q --upgrade pip
$PIP install -q "torch==2.5.1" "torchvision==0.20.1" --index-url https://download.pytorch.org/whl/cu121
$PY -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'devices',torch.cuda.device_count())"

echo "== 3. diffusers stack =="
$PIP install -q "diffusers==0.37.0" "transformers==4.57.1" accelerate safetensors "numpy<2" \
  sentencepiece protobuf imageio imageio-ffmpeg opencv-python-headless einops ftfy regex \
  huggingface_hub scikit-image pandas easydict lpips
$PY -c "import diffusers,transformers;print('diffusers',diffusers.__version__,'transformers',transformers.__version__)"

echo "== 4. cublas fix -- MUST BE THE LAST PIP RUN =="
$PIP install -q "nvidia-cublas-cu12==12.4.5.8"
$PIP list 2>/dev/null | grep -i "nvidia-cublas" || { echo "[FAIL] cublas missing"; exit 1; }
$PY - <<'EOF'
import torch
for M, K, N in [(2, 1536, 9216), (1, 768, 3072), (1, 3072, 18432)]:
    a = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
    b = torch.randn(K, N, device='cuda', dtype=torch.bfloat16)
    (a @ b); torch.cuda.synchronize()
print("[ok] SIGFPE shapes pass")
EOF

echo "== 5. weights (resumable, mirror, retries) =="
mkdir -p "$ROOT/weights" "$ROOT/logs"
"$PY" - "$ROOT" <<'EOF'
import sys, time
from pathlib import Path
from huggingface_hub import snapshot_download
root = Path(sys.argv[1]); W = root / "weights"
models = [
    ("CogVideoX-2b", ["zai-org/CogVideoX-2b", "THUDM/CogVideoX-2b"]),
    ("Wan2.1-T2V-1.3B", ["Wan-AI/Wan2.1-T2V-1.3B-Diffusers"]),
    ("HunyuanVideo", ["hunyuanvideo-community/HunyuanVideo"]),
]
for local, repos in models:
    dest = W / local
    if (dest / "model_index.json").exists():
        print(f"[skip] {local} already present")
        continue
    ok = False
    for attempt in range(1, 6):
        for repo in repos:
            try:
                print(f"[pull] {local} <- {repo} (attempt {attempt})", flush=True)
                snapshot_download(repo_id=repo, local_dir=str(dest), max_workers=4)
                print(f"[ok] {local}"); ok = True; break
            except Exception as e:
                print(f"[retry] {repo}: {type(e).__name__}: {str(e)[:160]}", flush=True)
                time.sleep(15)
        if ok:
            break
    if not ok:
        print(f"[FAIL] {local} after 5 attempts")
EOF

echo "== 6. code + prompts =="
mkdir -p "$ROOT"/{experiments/video,results,logs,repro,resources/prompts}
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cp -n "$SRC"/*.py "$ROOT/experiments/video/" 2>/dev/null || true
cp -n "$SRC"/../repro/metrics.py "$ROOT/repro/" 2>/dev/null || true
cp -n "$SRC"/../resources/prompts/prompt_video.txt "$ROOT/resources/prompts/" 2>/dev/null || true
echo "code: $(ls "$ROOT/experiments/video" | tr '\n' ' ')"

echo "== 7. validation (not just 'it ran') =="
cd "$ROOT" || exit 1
echo "-- 7a. reference videos: LPIPS needs RA Full; it is NOT in git (P21)"
find "$ROOT/results/ra_wan" -name 'video_*.mp4' 2>/dev/null | wc -l | xargs echo "   RA Full videos found:"

echo "-- 7b. one Wan sample at N=50 (all-exact); a flat colour means the text "
echo "      embeddings or the latent dtype regressed (P16/P17)"
CUDA_VISIBLE_DEVICES=0 $PY experiments/video/mpcache_wan.py \
  --seed-from-prompt --negative "animation" --solver unipc --sched sigma \
  --frames 33 --no-full --save-video --save-op ex1 --save-n 50 --Ns 50 --planners ex1 \
  --nprompt 1 --out "$ROOT/results/_bootstrap_smoke" 2>&1 | tail -2
$PY - <<'EOF'
import glob
import imageio.v3 as iio
f = glob.glob('/root/viewdit/results/_bootstrap_smoke/videos/ex1/*.mp4')
if not f:
    print('[FAIL] no video written'); raise SystemExit(1)
a = iio.imread(f[0], plugin='FFMPEG')
print('[check] %s shape=%s mean=%.1f std=%.1f' % (f[0], a.shape, a.mean(), a.std()))
if a.std() < 40:
    print('[FAIL] flat video: check P16 (zero-padding of text embeddings) and P17 (fp32 latents)')
    raise SystemExit(1)
print('[ok] real content')
EOF

echo "== done =="
echo "queues:  bash $ROOT/queue_wan.sh   (ex1 -> tea_gap -> full -> eval, all 4 GPUs)"
