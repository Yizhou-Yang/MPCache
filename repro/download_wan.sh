#!/usr/bin/env bash
# Resumable Wan2.1-T2V-1.3B-Diffusers download (hf-mirror drops connections now and then).
# On success appends "[setup] DONE" to logs/repro_setup.log, which unblocks the GPU queues.
cd /root/viewdit
source /root/miniconda3/etc/profile.d/conda.sh
conda activate racfg
DST=/root/viewdit/weights/Wan2.1-T2V-1.3B-Diffusers
for try in $(seq 1 30); do
  HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HUB_ETAG_TIMEOUT=60 HF_HUB_DOWNLOAD_TIMEOUT=120 \
  python - <<PY && break
from huggingface_hub import snapshot_download
snapshot_download("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", local_dir="$DST", max_workers=4)
PY
  echo "[download_wan] try $try failed, retrying in 30s"; sleep 30
done
python - <<PY || exit 1
import json, pathlib
d = pathlib.Path("$DST")
idx = json.loads((d / "model_index.json").read_text())
missing = [k for k, v in idx.items() if isinstance(v, list) and not (d / k).is_dir()]
assert not missing, missing
print("[download_wan] ok, components:", [k for k, v in idx.items() if isinstance(v, list)])
PY
du -sh "$DST"
echo "[setup] DONE" >> logs/repro_setup.log
