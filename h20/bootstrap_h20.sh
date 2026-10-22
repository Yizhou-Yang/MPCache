#!/usr/bin/env bash
# One-shot bootstrap for a fresh H20 host running the MPCache reproduction.
# Usage:  bash bootstrap_h20.sh /root/viewdit
# Idempotent: safe to re-run after a machine is reclaimed and re-attached.
set -uo pipefail
ROOT="${1:-/root/viewdit}"
PY=/root/miniconda3/envs/racfg/bin/python
PIP=/root/miniconda3/envs/racfg/bin/pip
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=60

echo "== 1. python env =="
if [ ! -x "$PY" ]; then
  echo "creating conda env racfg (python 3.10)"
  /root/miniconda3/bin/conda create -y -n racfg python=3.10
fi
"$PY" - <<'EOF'
import sys
print("python", sys.version.split()[0])
EOF

echo "== 2. packages (pinned to the V100 reference env) =="
$PIP install -q --upgrade pip
$PIP install -q torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
$PIP install -q diffusers==0.37.0 transformers==5.2.0 accelerate safetensors numpy \
  sentencepiece protobuf imageio imageio-ffmpeg opencv-python-headless einops ftfy regex

echo "== 3. weights (resumable, mirror, retries) =="
mkdir -p "$ROOT/weights" "$ROOT/logs"
"$PY" - "$ROOT" <<'EOF'
import sys, time
from pathlib import Path
from huggingface_hub import snapshot_download
root = Path(sys.argv[1]); W = root / "weights"
models = {
    "CogVideoX-2b": "THUDM/CogVideoX-2b",
    "Wan2.1-T2V-1.3B-Diffusers": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "HunyuanVideo": "hunyuanvideo-community/HunyuanVideo",
}
for local, repo in models.items():
    dest = W / local
    if dest.exists() and any(dest.iterdir()) and (dest / "model_index.json").exists():
        print(f"[skip] {local} already present")
        continue
    for attempt in range(1, 6):
        try:
            print(f"[pull] {local} <- {repo} (attempt {attempt})", flush=True)
            snapshot_download(repo_id=repo, local_dir=str(dest),
                              resume_download=True, max_workers=4)
            print(f"[ok] {local}")
            break
        except Exception as e:
            print(f"[retry] {local}: {type(e).__name__}: {e}", flush=True)
            time.sleep(20)
    else:
        print(f"[FAIL] {local} after 5 attempts")
EOF

echo "== 4. code =="
test -f "$ROOT/experiments/video/mpcache_controller.py" \
  && echo "controller present" \
  || echo "WARNING: copy experiments/video/*.py and repro/*.py into $ROOT/experiments/video and $ROOT/repro"

echo "== 5. smoke: one CogVideoX-2B sample at N=13 =="
cd "$ROOT" || exit 1
$PY experiments/video/mpcache_controller.py --frames 17 --nprompt 1 --seeds 3000 \
    --Ns 13 --planners reuse,tea_gap --out "$ROOT/results/video/_bootstrap_smoke" 2>&1 | tail -3

echo "== done =="
echo "long runs:  setsid nohup $PY experiments/video/mpcache_controller.py ... > logs/<name>.log 2>&1 < /dev/null &"
