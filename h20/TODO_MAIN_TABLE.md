# H20 执行清单（2026-10-27，第二次修订）

> 上一版（`482ebf0`）有三处与现实不符，本版按实测替换：
> 1. 这台机器是**全新**的（无 conda/权重/参考视频），"10 分钟口径校验"根本无从谈起；
> 2. P0 的同源校验**已经失败**：`PITFALLS P15` 记录过 ours 与 RA 同 prompt/seed 的
>    LPIPS ≈ 0.98。本次定位到根因是我们自己的两个 bug（`P16`/`P17`），不是 RA 的 patch；
> 3. ex1 命令漏了 `--frames 81`，默认 33 帧（`P19`）。

## 0. 方法状态

| 项 | 值 |
|---|---|
| 控制器 | `h20/mpcache_controller.py` 与 `repro/code/mpcache_controller.py` 现已同为一个文件，sha256 `baeb0e61c0d5a398…`（`b623a12` 恢复的权威副本）；本机已同步到 `experiments/video/` |
| Cog 上的方法 | 对角在线模型 + 门控 `age ≥ 2`；`--auto-base 1` 在 Cog 上是空操作（48/48 逐位相同） |
| Wan 上的方法 | 均匀调度 → 纯一阶外推（ex1），不加模型修正 |
| 一句话 | Cog 跑 `tea_gap`，Wan 跑 `ex1` |

## 1. 主表状态

| Backbone | Full | TeaCache | TaylorSeer | RA-Slow | RA-Fast | ours |
|---|---|---|---|---|---|---|
| CogVideoX-2B | ✅（RA 侧） | ✅ | ✅ | ✅ | ✅ | ❌ 需 49f Full 参考 + controller 加 `--save-video` |
| Wan2.1-T2V-1.3B | ✅ 本机已重跑 | ✅ | ✅ | ✅ | ✅ | 🔄 本机重跑中（P16/P17 修复后） |
| HunyuanVideo-13B | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ NaN 未修 |

## 2. 口径校验（已完成，结论与上一版相反）

上一版要求「ours full50 vs RA Full，LPIPS < 0.01 且 PSNR > 35」。实测（81 帧、seed 0、50 步全精确）：

| 对比 | LPIPS | SSIM | PSNR | 结论 |
|---|---:|---:|---:|---|
| ours（修 P16/P17 前） vs RA Full | 0.891 | 0.207 | 9.4 | 完全无关（ours 是纯蓝屏，见 P16） |
| ours（修 P16，bf16 latent） vs RA Full | 0.089 | 0.867 | 25.5 | 同内容，但 bf16 累计漂移（P17） |
| **ours（修 P16+P17） vs RA Full** | **0.030** | 0.927 | 30.7 | 同内容，剩的是跨实现数值底噪 |
| ours（同上） vs 纯 diffusers 管线 | 0.021 | 0.936 | 31.8 | 我们的 runner 本身已对齐官方 |

结论：**两者不完全等价，但已在「同一场景、同一噪声」的范围内**。0.03 是每个 ours 数字都要背的
底噪，方法效应（0.09–0.14）仍在。主表报数时必须写这一条，不能声称 bit-level 同源。

## 3. 已排期与执行（4 卡饱和，队列串联，无空转）

阶段一（`queue_wan.sh`，自动串下一棒）：

| 时段 | 卡 | 任务 | 产出 |
|---|---|---|---|
| 11:35–13:55 | 0-3 | RA Full，100 prompt × 50 步 | `results/ra_wan/wan_original_wan_original_full/videos` 100 mp4 |
| 15:03–15:30 | 0-3 | Wan ours **ex1** N=13，81 帧 | `results/wan_arms/ex1/videos/ex1` |
| 15:30–15:55 | 0-3 | Wan ours **tea_gap** N=13（对照旧 0.1375） | `results/wan_arms/tea_gap/videos/tea_gap` |
| 15:55–18:20 | 0-3 | Wan **our full** N=50（自洽参考，100 prompt） | `results/wan_full/n50/videos/ex1` |
| 18:20–18:35 | CPU | `repro/metrics.py`：ex1/tea_gap vs our-full 与 vs RA-full | `results/metrics_*.{json,csv}` |

阶段二（`queue_stage2.sh`，等阶段一退出后自动启动，继续占满 4 卡）：

| 时段 | 卡 | 任务 | 为什么 |
|---|---|---|---|
| 18:35–19:45 | 0-3 | Wan ex1 **N=10**（100 prompt） | 质量-NFE 曲线的低预算点，直接复用已验证代码 |
| 19:45–21:30 | 0-3 | Wan ex1 **N=17**（100 prompt） | 曲线的高预算点，同时给出「N 越大越接近 full」的单调性检查 |

未排上 GPU 的两项（不是忘了，是前置条件没到，硬开只会产出不可用的数字）：

| # | 任务 | 卡它的东西 | 成本 |
|---|---|---|---|
| 1 | Cog 49f ours | controller 还缺 `--save-video/--no-full/--negative/--video-dir`；且 **49f Full 参考要 10.4 h/4 卡**，本机时段放不下，没有它 LPIPS 无从算 | 0.5 h 改码 + 10.4 h Full + 2.7 h ours |
| 2 | Hunyuan 13B | `rel=nan` 未定位；且竞品四行（TeaCache/TaylorSeer/RA-Slow/RA-Fast）全空，单跑 ours 补不齐一行 | 1–2 h 定位 + 6 h Full + 1.6 h ours + 4×1.5 h 竞品 |

实测利用率：三个阶段内 4 张卡持续 100%（采样期 8.5 GiB/97 GiB，RA Full 16.9 GiB）；
唯一的空转是评测那 15 分钟，评测本身跑在 CPU 上，不需要卡。
两阶段脚本都用「等进程退出再起下一臂」，而不是按臂拆卡——拆卡会让关键路径变长（见 STATUS 决策 7）。

**Cog 的移植注意**（下次别踩）：controller 的 latent 是 `(1, LT, C, PH*2, PW*2)` 布局，
而 `AutoencoderKLCogVideoX.decode` 要 `(B, C, T, H, W)`，且 **49f/720×480 的 latent 空间是
60×90**——`FRAMES[49]` 给的 `ph=60, pw=90` 再乘 2 就对不上，必须先写一条 17f 单样本把
shape 打到日志里确认，再谈批量。Cog 的 VAE 没有 `latents_mean/std`，帧率是 8 fps。

## 4. 验收标准

| 检查 | 期望 | 现状 |
|---|---|---|
| Wan ex1 LPIPS（vs RA Full） | ≤ 0.12 | 待评测 |
| Wan ex1 vs tea_gap | ex1 更低 | 待评测 |
| 视频非纯色 | 帧 std > 40 | ✅ 114.8/79.7 |
| 主表每格 | 有数或 `--`，不估算 | 遵守 |
| 加速比 | 实测 wall-clock，不用 NFE 比 | 遵守 |

## 5. 本轮修掉的环境/代码问题（详见 `PITFALLS`）

| 编号 | 问题 | 影响 |
|---|---|---|
| P16 | 文本嵌入没截断到真实长度、padding 位没清零 | Wan 出纯蓝视频，`rel` 却正常 |
| P17 | latents 用 bf16 累计 | 与官方 50 步全精确差 LPIPS 0.06 |
| P18 | cublas 12.4.5.8 被后续 pip 降回 12.1.3.1 | RA 四卡同时 SIGFPE (-8) |
| P19 | `--frames` 默认 33 | 与 81 帧参考不可比 |
| P20 | `scripts/eval.py` 不存在 | 实际用 `repro/metrics.py` |
| P21 | 参考视频不在 git | 换机必须重跑 RA Full（2.2 h） |
