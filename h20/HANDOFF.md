# HANDOFF — 下次从这里开始

> 给下一个接手的人 / 下一个 AI：读完这一页 + `STATUS.md` + `PITFALLS.md`，
> 5 分钟就能接着干。完整数字在 `EXPERIMENT_SUMMARY.md`。

---

## 0. 一句话

MPCache 论文（步级缓存预算规划）的实验仓库。**V100 跑 CogVideoX（只能 17 帧）**，
**H20 跑 Wan2.1 / HunyuanVideo 的 fidelity 主表**（LPIPS/SSIM/PSNR vs 50 步 Full）。

---

## 1. 仓库结构

```
repro/
  code/            控制器与探针（V100 侧，产生 trace）
  results/video/   64+ 个 JSONL trace（rel 口径，含负结果）
  analyze/         paired.py / gap_analyze.py / d_identifiability.py
  merge_runs.py    把 H20 分片合并成 run 目录 + manifest
  metrics.py       LPIPS/SSIM/PSNR（全帧，与 RA 的 eval_video.py 对齐）
  finalize.sh      23:15 定时收尾（指标 + 总结 + 推送）
  push_results.sh.example  增量推送模板（**用 GH_TOKEN 环境变量，不要提交 token**）
paper/             论文 v97 源码
h20/               本机手册与状态
  RUNBOOK.md       从零建环境 → 跑实验（含 prompt 检查点、RA 接入、ours 命令）
  PITFALLS.md      15 条坑，每条有症状/根因/修复
  STATUS.md        进度、P0–P3 TODO、决策记录
  EXPERIMENT_SUMMARY.md  当晚结果 + 6 条经验
  HANDOFF.md       本文件
```

---

## 2. 五分钟恢复

```bash
# 1) 先看状态（不要先跑命令）
cd /root/MPCache
git log --oneline -5
sed -n '1,60p' h20/STATUS.md          # 当前在跑什么、还缺什么
grep -n "P1[0-9]" h20/PITFALLS.md     # 最近踩的坑

# 2) 环境是否还在
/root/miniconda3/envs/racfg/bin/python -c "import torch,diffusers;print(torch.__version__,diffusers.__version__)"
/root/miniconda3/envs/racfg/bin/python -c "import torch;print(torch.cuda.device_count())"

# 3) 权重
ls /root/viewdit/weights/*/model_index.json

# 4) 上一个任务跑到哪
cat repro/results/H20_PROGRESS.txt 2>/dev/null
```

---

## 3. 当前状态（2026-10-22 夜）

| 项 | 状态 |
|---|---|
| H20 环境 | `racfg`：py3.10 / torch 2.5.1+cu121 / diffusers 0.37 / transformers 4.57.1 |
| cuBLAS | **必须** `nvidia-cublas-cu12==12.4.5.8`，否则小 M bf16 GEMM SIGFPE |
| Wan 主表 | Full 100 + ours 100（自洽基准）；RA TeaCache / CFG-Slow / CFG-Fast |
| Hunyuan | runner 可跑（49f 832×480，17.3 s/步），**未做 100-prompt** |
| Cog | V100 只有 17f；49f 待下一台 H20（13.6 h） |
| rel 口径 | `wan33_20261022a` 699 行、`hy49_pilot` 18 行（仅 ablation） |

---

## 4. 绝对不要做的事

| 禁忌 | 原因 |
|---|---|
| 改 seed 语义为"全局固定 0" | RA 的 `--seed 0` 是 base seed，有效 seed = prompt index；改了配对失效 |
| 用 per-token 门控进主表 | V100 `pt16` 实测 −21.7%（胜率 0.12），输出空间的修正必须整场一致 |
| 把 ours 与 RA 的视频混在同一列 | 两套 pipeline 的 Full 输出 LPIPS 0.98（P15），基准不同源 |
| 用 17 帧冒充 49 帧 | 论文正文写的是 49 帧，V100 跑不了就必须用 H20 补 |
| 把 token / 密钥提交进仓库 | `push_results.sh.example` 已改成读 `$GH_TOKEN` |
| 用 `ps` 判断任务是否完成 | `setsid` 会丢 pid；用 `closed.jsonl` 行数或 `video_*.mp4` 计数 |
| 用 17f 的步时估 49f | 差 40 倍（0.75 vs 29.9 s/步，attention 二次） |

---

## 5. 下次第一件事

**Cog 49f × 100 prompt（13.6 h）**，步骤见 `STATUS.md` 5.1：

1. 给 `mpcache_controller.py` 补 `--save-video`（`AutoencoderKLCogVideoX`，fps 8，无 latent 反归一化）；
2. Full 100（10.4 h）→ ours 100（2.7 h）→ 指标（0.3 h）；
3. 验收：100+100 个 mp4、索引对齐、LPIPS 0.05–0.30。

之后：Hunyuan 的 4 个竞品行（各 1.5 h）、Wan 的 Fewer25（1.1 h）、TaylorSeer（1.0 h）。

---

## 6. 论文口径的约定（别再重新讨论）

1. **主表用 UniPC**（flow shift 3），Euler 只作 solver 阶数消融；
2. **seed = prompt index**，来自 RA 的 `base + video_idx`；
3. **negative prompt = "animation"**，三个 backbone 统一；
4. **fidelity 对 50 步 Full**，不是对 latent；
5. **加速倍数不跨论文横比**：WorldCache 的 3.65× 是 PSNR 23.5 换来的，我们的口径是匹配 NFE 的保真度；
6. **两篇同名 WorldCache 要分开引**（见 PITFALLS P14）；
7. `rel` 口径的数据只能进附录，表注写「100 self-authored prompts, seed 3000」。

---

## 7. 凭据

仓库里的脚本不保存任何 token。推送需要向项目所有者索取 PAT，然后：

```bash
export GH_TOKEN=<pat>
git push "https://x-access-token:${GH_TOKEN}@github.com/Yizhou-Yang/MPCache.git" h20-run
```

---

## 8. 这次没做但值得记的

- 会话 trace 本身没有入库：体积大、噪声高、且含敏感信息。取而代之的是本文件 +
  `STATUS.md` + `PITFALLS.md` + `EXPERIMENT_SUMMARY.md` 四份结构化文档，
  覆盖了"看什么、为什么这么决定、踩过什么、还缺什么"。
- 「harness 差异」的定量标尺（`LPIPS(RA-TeaCache, RA-Full)` 对比 `LPIPS(ours, our-Full)`）
  跑完要写回 `EXPERIMENT_SUMMARY.md` 第 2.1 节的表注：两者量级接近 → 可加表注并列；
  差得远 → 必须分表。
