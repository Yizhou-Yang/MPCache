# H20 实验总结（2026-10-22）

> 本文件在机器回收前最后更新。数字由 `repro/metrics.py` 与 `repro/results/` 下的 json 生成，
> 口径与踩坑见 `h20/PITFALLS.md`，进度与 TODO 见 `h20/STATUS.md`。

---

## 1. 今晚的时间线

| 时刻 | 事件 |
|---|---|
| 14:13 | 机器探测：4×H20-97GB，驱动 535 / CUDA 12.2，网络全通（代理） |
| 14:20 | 建 `miniconda3/envs/racfg`：py3.10 + torch 2.5.1+cu121 + diffusers 0.37 + transformers 4.57.1 |
| 14:30 | 发现 cuBLAS 12.1 小 M bf16 GEMM SIGFPE，升级 12.4.5.8 后修复 |
| 14:35 | 拉齐三份权重（Cog 13G / Wan 27G / Hunyuan 40G，HF 直下） |
| 14:40 | Wan runner 移植完成（`mpcache_wan.py`），33f 1.66 s/步、81f 6.40 s/步 |
| 14:50 | Hunyuan runner 移植完成（`mpcache_hunyuan.py`），49f 832×480 17.3 s/步 |
| 15:00 | 接入 RA-CFGCache（`github.com/yiming-l21/RA-CFGCache`），验证 `demo_wan.sh` 可跑 |
| 15:10 | 统一 prompt 协议：RA 的 `prompt_video.txt`（100 条 VBench） |
| 15:20 | 跑 RA Full（Wan 81f，100 prompt，4 卡） |
| 16:52 | RA Full 完成（100 视频，6.0 min/prompt） |
| 17:05 | ours 启动失败（`--height/--width` 参数不存在），修复后重启 |
| 17:25 | 发现 ours 与 RA 视频不可比（LPIPS 0.98 @ N=50），改用自洽基准 |
| 17:25 | 启动 our Full（N=50，100 prompt） |
| 18:48 | our Full 60/100 |
| 19:45 | our Full 完成 → ours (N=13) |
| 20:25 | ours 完成 → RA TeaCache |
| 21:15 | RA TeaCache 完成 → RA CFGCache-Slow |
| 22:05 | RA CFGCache-Slow 完成 → RA CFGCache-Fast（22:50 截止） |
| 22:50 | 停止新任务 |
| 23:00 | 最终推送 + 本文件定稿 |

---

## 2. 结果

### 2.1 Wan2.1-T2V-1.3B（81 帧 832×480，50 UniPC 步，100 VBench prompts，seed = prompt index）

| 方法 | 基准 | n | LPIPS↓ | SSIM↑ | PSNR↑ | Speedup↑ |
|---|---|---|---|---|---|---|
| Full (T=50) | 自比 | 100 | 0.0000 | 1.0000 | ∞ | 1.00× |
| ours (tea_gap, N=13) | our Full | 待填 | 待填 | 待填 | 待填 | 3.85× |
| RA TeaCache (0.15) | RA Full | 待填 | 待填 | 待填 | 待填 | 待填 |
| RA CFGCache-Slow (0.6) | RA Full | 待填 | 待填 | 待填 | 待填 | 待填 |
| RA CFGCache-Fast (1.4) | RA Full | 可能跳过 | | | | |

**⚠ 基准不同源**：ours 的参考视频由本仓库 runner 生成，baseline 的参考视频由 RA-CFGCache
生成。两者在 N=50 全精确下 LPIPS 0.98（见 PITFALLS P15），因此 ours 与 baseline 的
LPIPS/SSIM/PSNR **不能严格同列比较**。TeaCache 那一行同时充当"harness 差异"的标尺：
若 `LPIPS(TeaCache vs RA Full)` 与 `LPIPS(ours vs our Full)` 量级接近，说明相对保真度
对 pipeline 差异不敏感，可加表注并列；否则必须分表。

### 2.2 rel 口径（仅 ablation，不得进主表）

| run | 内容 | 行数 | 结论 |
|---|---|---|---|
| `wan33_20261022a` | Wan 33f，60 prompt × Ns 10/13/17 × 4 臂 | 699 | `tea_gap` vs `reuse`：N=17 −15%；`mpc` 在 N=10/13 显著更差（+110%/+133%） |
| `hy49_pilot_20261022a` | Hunyuan 49f，4 prompt × 2 Ns × 3 臂 | 18 | 仅冒烟级 |

### 2.3 未完成

| 项 | 原因 |
|---|---|
| Hunyuan 100-prompt / 竞品行 | 17.3 s/步，100 prompt = 70 卡时；且无自洽 baseline |
| Cog 49f 100-prompt | **已定性为下一台 H20 的第一任务**（13.6 h）；V100 只能 17f，不能冒充 49f |
| Wan Fewer25 | 1.1 h，放不进 22:50 截止 |
| TaylorSeer | 1.0 h，同上 |
| ours 与 RA 同 harness | 状态空间不匹配，见 P15 与 STATUS P1 |

---

## 3. 关键决策与理由

| 决策 | 理由 |
|---|---|
| 主表用 UniPC（不换 Euler） | 与 RA 基线同 solver，否则二阶/一阶混比不可比；Euler 只作消融 |
| seed = prompt index | RA 的 `--seed 0` 是 base seed，实际 seed = `0 + video_idx`；metadata 实测 `video_000025.json → seed 25` |
| 100 条 VBench prompts | 与 RA 逐行一致，索引对齐才能按 `video_%06d.mp4` 配对 |
| ours 用 `--no-full` | fidelity 的参考就是 50 步 Full，自己再跑一遍既慢又不产生新信息 |
| ours 与 baseline 分基准 | P15：两套 pipeline 的 Full 输出不一致 |
| 砍 Hunyuan / Cog / Fewer | 时间只够一条主线；宁缺不凑 |
| 不引入 per-token 门控 | V100 `pt16` 实测 −21.7%（胜率 0.12），输出空间的修正必须保持空间连贯 |

---

## 4. 踩坑（完整 15 条见 `h20/PITFALLS.md`）

按杀伤力排序，每条都花了时间：

1. **cuBLAS 12.1 小 M bf16 GEMM → SIGFPE**：进程无 traceback 直接消失，栈顶 `linear.py:125`。升 12.4.5.8 修复，但 Wan 的 `time_proj`（M=2,K=1536,N=9216）仍需局部 fp32。
2. **内置 8 条 prompt**：`--nprompt 100` 在 `embs[8]` 崩；改为读 RA 的 100 条。
3. **ours 与 RA 不可比**（P15）：LPIPS 0.98 @ N=50，排除噪声/dtype/VAE/solver/prompt 后仍未定位。
4. **VAE 反归一化**：Wan 解码前必须 `z*latents_std+latents_mean`，漏了会得到完全不同色彩。
5. **Wan 的 UniPC 是二阶**：跳步时历史被伪造，N=13 误差 0.65；换 Euler+uniform 降到 0.30（但主表仍用 UniPC）。
6. **`--prompt-offset` 下 sanity 读 `embs[0]`**：分片时 `KeyError: 0`。
7. **RA 输出先落 `.multi_gpu_tmp`**：merge 前回收会丢已完成视频。
8. **GitHub SSH 不可用**：只能 HTTPS + token。
9. **`zfull` 未缓存**：补一条臂要重付 50 步。
10. **Hunyuan 是 embedded guidance**：batch=1、`guidance=6*1000`，不是 CFG 双分支。

---

## 5. 经验（下次少走弯路）

1. **先对齐协议再跑量**：今晚 699 行 rel 数据因为 prompt/seed/指标口径不对，只能进附录。协议三件套（prompt 文件、seed 语义、solver）要在开跑前逐项核对，而不是跑完再对。
2. **baseline 与 ours 必须同 harness**：不同 pipeline 的 Full 输出能差到 LPIPS 0.98，比方法差异大一个量级。下一台机器第一件事是把 ours 注进 RA 的 `cal_type` 钩子。
3. **`--sanity` 只能验证"能跑"，不能验证"对"**：今晚的 VAE 反归一化和 pipeline 分叉，sanity 全过。要加一个"与已知输出对比"的冒烟（例如同一个 seed 跑官方 pipeline，比较 latent 的 L2）。
4. **每个阶段都用可观测的完成判据**：`closed.jsonl` 行数、`video_*.mp4` 计数，而不是 `ps`（`setsid` 会丢 pid）。
5. **时间预算按实测步时算**：Cog 49f 是 Wan 81f 的 4.7 倍单步成本，用 17f 的 0.75 s/步估 49f 会差 40 倍。
6. **负结果要留证据**：per-token 的 −21.7%、mpc 的 +110%，都是"为什么不做某件事"的答案，比再跑一个变体有价值。

---

## 6. 数据清单（哪些能进论文）

| 路径 | 内容 | 能否进主表 |
|---|---|---|
| `results/ours_wan/full50/` | our Full，100 mp4 | 参考基准 |
| `results/ours_wan/n13/` | ours，100 mp4 | ✓ 配表注 |
| `results/ra_wan/wan_original_full/` | RA Full，100 mp4 | 参考基准 |
| `results/ra_wan_tea/`、`ra_wan_cfg/`、`ra_wan_cfgfast/` | RA 竞品 | ✓ 与 RA Full 比 |
| `repro/results/video/wan33_20261022a/` | 699 行 rel | ✗ 附录 ablation |
| `repro/results/video/hy49_pilot_20261022a/` | 18 行 rel | ✗ 冒烟 |
| `results/ra_metadata/` | RA 各分片 manifest/config | 溯源用 |
