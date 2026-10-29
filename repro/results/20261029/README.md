# 2026-10-29 主表实验：可核查数据包

本目录是**新 H20（4×H20，2026-10-29）**上主表实验的完整可核查产物。

## 目录

| 路径 | 内容 |
|---|---|
| `traces/` | 所有机器可读 trace：每个臂的 `closed.jsonl`（逐 prompt 的 rel/nfe/steps）、`metrics_*.json`（LPIPS/SSIM/PSNR 汇总与逐条）、以及 RA harness 的 `config.json` 与逐视频 `metadata/*.json`（记录了 prompt/seed/步数/缓存设置） |
| `logs/` | 每个阶段的日志尾部（400 行），含启动配置、每 prompt 的步列表、错误栈 |
| `manifests/sha256sums.txt` | **每个产物文件的 sha256**，用于核对报告数字与实际视频一一对应 |
| `manifests/COUNT.txt` / `SIZE.txt` | 文件计数与体积 |
| `videos/` | 每个臂前 3 个 prompt 的视频（`scale=480, crf 32` 压缩，仅用于人工抽查；**指标在原始视频上计算**） |

## 口径

- 所有臂：100 个 VBench prompt 的前 50 个，seed = prompt index，negative prompt `animation`；
- Wan2.1-T2V-1.3B：81 帧、832×480、UniPC-50 步、flow shift 3、CFG 5；
- ours 的加速点：N=19（2.63×），`tea_gap` 与 `fr`（隐藏层残差重建）两臂都跑；
- 参考：RA harness 的 `--mode original`（50 步全量）；
- 评测：`repro/metrics.py`，LPIPS(alex) 帧间均值。

## 核查方法

```bash
# 1) 校验文件完整性
sha256sum -c manifests/sha256sums.txt

# 2) 重算指标（需要原始视频，仓库内为压缩版，指标会略有差异）
python repro/metrics.py --ref <RA Full 视频目录> --cmp <某臂视频目录> --out-json out.json

# 3) 复现某臂
见 h20/RUNBOOK_DEPLOY.md 与 h20/TODO_MAIN_TABLE.md 中的命令行
```

> 说明：仓库内的 `videos/` 是压缩抽查版；**报告中所有数字均在未压缩原始视频上计算**，原始视频的 sha256 记录在 `manifests/sha256sums.txt` 中。
