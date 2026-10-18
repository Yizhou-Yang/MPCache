# Resume instructions — H20 reproduction (environment was reclaimed)

Written 2026-10-18 09:40 (+08:00) because the container is being recycled.

## 1. What is preserved and where

| asset | location |
|---|---|
| code, scripts, harness | branch `h20` of `https://github.com/Yizhou-Yang/MPCache` |
| audit bundle (clean logs, trace stats, sha256 manifest, env fingerprint) | `audit/` on branch `h20` |
| progress report | `docs/H20_PROGRESS_2026-10-18.md` |
| audit guide | `docs/AUDIT_H20.md` |
| **all videos + metadata + raw logs (952.5 MB)** | GitHub **Release** `h20-artifacts-2026-10-18`, asset `videos_repro_h20.tar` |
| environment setup log | `docs/env/setup_h20.log` |

The tar contains the whole `results/repro_h20/` tree (videos, `metadata/`, `manifest.jsonl`, `_logs/`).

## 2. State at reclaim

| row | cogvideox | wan |
|---|---:|---:|
| vanilla50 | 100 | 100 |
| vanilla25 | 100 | 100 |
| teacache | 100 | 100 |
| taylorseer | 100 (+1 smoke) | 100 |
| racfg_slow | 100 | **84** (in flight) |
| racfg_fast | 100 | **16** (in flight) |

- CogVideoX is **complete** (600/600).
- Wan is missing `racfg_slow` 16 and `racfg_fast` 84 prompts.
- No aggregation has been run yet (it requires an idle GPU and all rows present).
- `wan + taylorseer` completed 100/100 on 96 GiB **without OOM**.

## 3. Rebuild on a new H20 host

```bash
# 0) prerequisites: miniconda at /root/miniconda3, or set CONDA_SH
/root/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
/root/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# 1) code
git clone https://github.com/Yizhou-Yang/MPCache.git /root/viewdit
cd /root/viewdit && git checkout h20

# 2) environment + pinned upstream + weights (~40 GB, hf-mirror, ~20 min)
export VIEWDIT=/root/viewdit RACFG_PROFILE=h20 RACFG_DTYPE=bfloat16 CPU_OFFLOAD=0 ALLOW_GIT_PUSH=0
bash repro/setup_h20.sh        # must end with "[setup] DONE"

# 3) restore previous results so the runner skips finished videos
curl -L -o /tmp/videos.tar \
  https://github.com/Yizhou-Yang/MPCache/releases/download/h20-artifacts-2026-10-18/videos_repro_h20.tar
mkdir -p /root/viewdit/results
tar -xf /tmp/videos.tar -C /root/viewdit/results     # yields results/repro_h20/...

# 4) finish only what is missing (resumable, exact global ids)
CUDA_VISIBLE_DEVICES=0 bash repro/run_repro.sh wan 0 50 racfg_slow
CUDA_VISIBLE_DEVICES=1 bash repro/run_repro.sh wan 50 100 racfg_slow
CUDA_VISIBLE_DEVICES=2 bash repro/run_repro.sh wan 0 50 racfg_fast
CUDA_VISIBLE_DEVICES=3 bash repro/run_repro.sh wan 50 100 racfg_fast

# 5) aggregate (needs an idle GPU; LPIPS/SSIM/PSNR vs same-hardware Vanilla50)
python repro/aggregate.py --model cogvideox --n 100 --device cuda:0
python repro/aggregate.py --model wan --n 100 --device cuda:0

# 6) audit bundle + push
python3 repro/audit_pack.py
```

## 4. Pitfalls already hit (do not rediscover)

1. **conda ToS** blocks `conda create` in non-interactive mode; accept both channels first (step 0).
2. **`uv`/conda 环境不要移动目录**：本机用的是 conda env，不受 editable-path 影响；若改用 uv，先 `uv sync` 再跑，移动目录会破坏 `.pth`。
3. **权重走 `hf-mirror.com`**；容器无公网直连，只有 HTTPS 代理（`star-proxy`），因此 `download.pytorch.org` 等直连会 `Network is unreachable`。
4. **不要用 `nc -z` 判断连通**：iOA 网关先 accept 再 403，必须读 SSH banner。
5. **日志清洗陷阱**：tqdm 进度条与 `CFGCache step` trace 常在同一物理行；清洗时必须先剥前缀再判空，否则会丢 trace（第一版丢了 744 条，已在 `audit_pack.py` 修复）。
6. **不要相信 `Saving time:` 是逐条延迟**：每个进程只输出一次汇总。
7. **视频不进 git**（`.mp4` 被 ignore）；用 Release + `audit/MANIFEST.json` 的 sha256 校验。

## 5. Standing scientific caveats

- H20 ≠ H200：质量指标可比（同 bf16、同 seed、同硬件 Vanilla50 配对），**速度/加速比不可比**。
- CogVideoX 上的 TaylorSeer 是额外行，RA-CFGCache Table 1 无此行。
- CFGCache 阈值修复是真实协议偏离，必须写进论文复现节。
- 复现基线不产生 MPCache 的 SOTA 或统计门主张。
- G0a 代价模型结论仍为 **FAIL**，本次运行不改变它。
