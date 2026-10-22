# H20 进度快照与 TODO

最后更新：2026-10-22 14:40 (CST)，机器 **9.5 h 后回收**（约 23:30），数据必须在此之前落到 GitHub。

---

## 1. 一句话状态

环境、权重、三个模型的 runner、RA 基线接入全部就绪；
Wan 的论文口径（LPIPS/SSIM/PSNR）正在跑，ours 与 Hunyuan 排队。

---

## 2. 已完成

| 项 | 说明 |
|---|---|
| 环境 | `/root/miniconda3/envs/racfg`，py3.10 + torch 2.5.1+cu121 + diffusers 0.37.0 + transformers 4.57.1 |
| cuBLAS 修复 | 升级 `nvidia-cublas-cu12==12.4.5.8`，根治 SIGFPE（见 PITFALLS P1） |
| 权重 | CogVideoX-2b 13G、Wan2.1-T2V-1.3B 27G、HunyuanVideo 40G，全部 HF 直下 |
| Cog runner | `mpcache_controller.py`，17f 0.75 s/步、49f 29.9 s/步 |
| Wan runner | `mpcache_wan.py`，33f 1.66 s/步、81f 6.40 s/步 |
| Hunyuan runner | `mpcache_hunyuan.py`，49f 544×960 27.1 s/步、832×480 17.3 s/步 |
| prompt 协议 | 统一读 RA 的 `prompt_video.txt`（100 条），三个 runner 同一套解析顺序 |
| 论文协议参数 | `--seed-from-prompt --negative "animation" --solver unipc --sched sigma --save-video` |
| RA 基线 | `https://github.com/yiming-l21/RA-CFGCache`，`RUN/demo_wan.sh` 已验证可跑 |
| 自动推送 | `repro/push_results.sh` 每 10 min 合并分片 + commit + push |
| 无人值守编排 | `after_ra_full.sh`（RA Full 结束→ours），`after_wan.sh`（Wan→Hunyuan） |

## 2.1 方法边界（2026-10-22 决定：不引入 token 级分类）

```
ours = tea_gap            # 步级预算规划 + age 门控 + 信赖域 cap
```

曾实现 `tea_tok`（逐 token 曲率二值门控）与 `tea_tokc`（曲率定 cap），冒烟可跑，
但**不采用、不进主表**：

- WorldCache 系工作做的是 **token 级特征复用 + 近似**（warp / blending / per-token 分类），
  我们做的是 **步级"何时评估"的预算规划**，两条轴正交。混进来会让贡献点变模糊，
  审稿人也会问"这到底是缓存策略还是 token 选择"；
- **V100 上 `pt16` 的 8 配对实测反证了这条路**：`tea_pt vs reuse` **−21.7%**（胜率 0.12），
  `tea_pt vs tea_gap` 中位比值 **1.293**（差 29%）。机制原因：我们的修正发生在**输出空间**（guided
  output `v`），必须保持整场空间连贯；只修一部分 patch、其余复用会让 latent 场出现局部跳变，
  下一步 attention 会放大它。WorldCache 能逐 token 是因为它在**特征空间**做缓存，
  并且用光流 warp 把缓存对齐到当前位置——我们没有对齐机制，粒度不能跨层迁移；
- 代码保留（`--planners tea_tok,tea_tokc --token-gate τ`），作为 future work 或附录探索，
  **不写入 `tab_main`**。

与 WorldCache 的关系只在 Related Work 里说清：他们解决"何处"，把"何时"留给阈值；
我们补"何时"，并给出可检验的机制（gap 探针、门控消融），而不是又一个阈值。

## 3. 已有数据（rel 口径，只能作消融）

> `rel` 是终点 latent 相对 L2，**不是**主表的 LPIPS/SSIM/PSNR；prompt 也不是 VBench 集。
> 禁止把这些数值填进 `tab_main` / `tab_baselines`，只能在附录作为 planner 消融引用，
> 表注写明「100 self-authored prompts, seed 3000」。废弃的 `wan81_n13_20261022a` 已删除。

| run | 内容 | 行数 | 口径 |
|---|---|---|---|
| `wan33_20261022a` | Wan 33f，60 prompt × Ns 10/13/17 × {reuse,ex1,tea_gap,mpc} | 699 | latent rel L2，seed 3000，**非 VBench，仅 ablation** |
| `hy49_pilot_20261022a` | Hunyuan 49f，4 prompt × 2 Ns × 3 臂 | 18 | latent rel L2，**仅 ablation** |

结论（仅作方向参考）：Wan 上 `tea_gap` 稳定优于 `reuse`（N=17 −15%），
`mpc` 在 N=10/13 显著劣于 reuse（+110%/+133%），N=17 才转正（−30%）。

## 4. 正在跑

```
RA-CFGCache Wan Full (T=50, UniPC, shift 3, CFG 5, 100 VBench prompts)
  4 卡 × 25 prompt，6.0 min/prompt，14:26 起，ETA 16:56
  → /root/viewdit/results/ra_wan/wan_original/{videos,metadata,eval_frames}
```

## 5. TODO（全部未完成项，按优先级）

### P0 本机今晚（16:55 起自动执行，不需人工）

| # | 项 | 规模 | 状态 |
|---|---|---|---|
| 1 | RA Full (T=50) × 100 | 100 mp4 | 跑中 87/100 |
| 2 | ours 三臂：`tea_gap` / `tea_tok` / `tea_tokc` | 300 mp4 | 排队 |
| 3 | RA CFGCache-Slow (`rel_l1_thresh 0.6`) | 100 mp4 | 排队 |
| 4 | RA TeaCache (`rel_l1_thresh 0.15`) | 100 mp4 | 排队 |
| 5 | Hunyuan Full + ours（12 prompt） | 24 行 | 排队 |
| 6 | 指标 `eval.sh` + 合并 + 推送 | — | 收尾 |

### P1 方法（下一台机器，优先级最高）

| # | 项 | 为什么 | 成本 |
|---|---|---|---|
| 10 | **OSI 向量投影**替换 `ex1` | 线性外推在轨迹弯曲时沿旧方向外推；最小二乘投影更稳 | 0.5 h 实现 + 0.6 h 跑 |
| 11 | **SWD 通道方差加权** | 显著性高的位置漂移应占更大权重 | 0.5 h 实现 |
| 12 | **mpc `--gamma 1.5`** | 现有 γ=0.6 预算前置，与 flow matching 尾段敏感相反；ATS 式后置放宽应能救回 | 0.6 h |

### P1-b 已否决（备查，不做）

| 项 | 否决理由 |
|---|---|
| chaotic-only β（只在高 κ token 上算误差比） | 属于 token 级改造；V100 `pt16` 已证明输出空间逐坐标掩码显著更差（−21.7%，胜率 0.12） |
| τ 敏感性 0.3/0.5/0.7 | 只在采用 `tea_tok` 时才需要，现不采用 |
| κ 的 ε 归一化 | 同上 |
| `tea_tok` / `tea_tokc` 进主表 | 代码保留，仅作 future work |

### P2 矩阵补全

| # | 项 | 说明 | 成本 |
|---|---|---|---|
| 13 | **Cog 帧数协议（二选一，见 5.1）** | V100 只能 17f；H20 49f 需 13.1 h | 决策项 |
| 14 | **Hunyuan 竞品行** | TeaCache / TaylorSeer / RA-Slow / RA-Fast 四行，全空 | 4 × 1.5 h |
| 15 | **Wan 的 RA-CFGCache-Fast + TaylorSeer** | 本机砍掉的两行 | 0.8 + 1.0 h |
| 16 | **Wan Fewer25** | 本机用三臂对比换掉了 | 1.1 h |
| 17 | **Euler ablation** | `--solver euler --sched uniform` vs UniPC，说明 solver 阶数影响 | 0.6 h |

### P3 论文写作

| # | 项 | 要点 |
|---|---|---|
| 18 | Related Work 摆位 | token 级缓存（何处）与步级控制（何时）是正交两轴；我们补后者，且把他们的二值分类连续化为 per-token 半径 |
| 19 | 两篇 WorldCache 区分 | 见 PITFALLS P14，引用别混 |
| 20 | 表注 | `rel` 口径数据只能进附录，注明 "100 self-authored prompts, seed 3000" |
| 21 | seed 约定 | 写进协议段（见 `repro/REPRODUCE.md` 1.1） |
| 22 | 加速倍数不横比 | WorldCache 3.65× 是 PSNR 23.5 换来的，我们口径是匹配 NFE 的保真度 |

### 5.1 Cog 的帧数协议（V100 上 49f 不可行）

**硬约束**：V100 只能跑 **17 帧**（49 帧在 V100 上不可行：显存/步时都不允许）。
因此 Cog 这条 backbone 目前**没有 49 帧的 100-prompt 结果**，只有 17 帧。

两条路，必须选一条并写进论文：

| 路径 | 成本 | 论文表述 | 风险 |
|---|---|---|---|
| **A. H20 跑 49f × 100 prompt** | 13.1 h（Full+ours，4 卡） | Cog 行与其他 backbone 同协议（49 帧官方默认） | 需要一整台 H20 的半天，本机今晚不够 |
| **B. 就用 V100 的 17f** | 0（已有） | Cog 单独一个 block，表注「17 frames, V100」 | 与 Wan 的 81 帧不同协议，不能声称"统一协议" |

我的建议：**A 作为下一台机器的第一个任务**（它同时解决 `--save-video` 缺口）；
在此之前，论文里 Cog 行只能用 B 并明确标注，不要与 Wan 并排比较加速比。

#### A 的实测成本（49 帧 480×720 = 29.9 s/步）

实测 49 帧 480×720 是 **29.9 s/步**（70k tokens × 30 层，GPU 无降频），按 100 prompt、4 卡：

| 配置 | GPU·h | 4 卡墙钟 |
|---|---|---|
| Full（参考，只跑一次） | 41.5 | **10.4 h** |
| ours（13 步） | 10.8 | 2.7 h |
| Full + ours | 52.3 | **13.1 h** |
| 5 臂全量 | 84.7 | **21.2 h** |

对比 Wan 81 帧（6.4 s/步）：Full 8.9 GPU·h / 2.2 h，5 臂 18.1 GPU·h / 4.5 h。
Cog 49 帧是 Wan 的 4.7 倍单步成本，**单 Full 就要 10.4 h**，这是它排不进本机的根本原因。

不要用 17 帧的 0.75 s/步去估 Cog 49 帧：那会得到"25 分钟跑完"的错误结论，
两者差 40 倍（17f 5 个 latent vs 49f 13 个 latent，attention 二次增长）。

下一台机器的**第一个**任务：

```bash
# 先补 --save-video（照搬 mpcache_wan.py 的 save_video，VAE 换 AutoencoderKLCogVideoX）
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i setsid nohup /root/miniconda3/envs/racfg/bin/python \
    /root/viewdit/experiments/video/mpcache_controller.py \
    --frames 49 --nprompt 25 --prompt-offset $((i*25)) \
    --seeds 0 --seed-from-prompt --negative "animation" \
    --Ns 13 --planners tea_tok --token-gate 0.5 \
    --out /root/viewdit/results/video/cog49_p$i > /root/viewdit/logs/cog49_p$i.log 2>&1 &
done
```

Cog 的 VAE 是 8×8 空间、4× 时间压缩，解码 49 帧 720×480 实测约 8 s / 3 GiB。

## 6. 关键决策记录

1. **主表用 UniPC**（与 RA 基线同 solver），Euler 只作消融——否则二阶/一阶混比不可比。
2. **seed = prompt index**（`--seed 0` + `start_index`），不是固定 3000。
3. **negative prompt = "animation"**，三个模型统一。
4. **ours 跳过 50 步全精确**（`--no-full`）：fidelity 的参考视频就是 RA 的 Full，
   自己再跑一遍既慢又无意义。
5. **Cog 不在 H20 跑**：49 帧 29.9 s/步，V100 已有结果。
6. **mpc 臂的负结果保留**，不删除、不隐藏，作为 planner 与 flow matching 不匹配的证据。
7. **排期用「每阶段独占 4 卡、串行」而不是「按臂拆卡并行」**。同样 18.1 GPU·h，
   拆卡会让每个臂只有 1 卡，竞品臂在 1 卡上串行：
   - 4 卡串行关键路径 = 2.2(Full) + 0.6(ours) + 0.7(CFG) + 0.7(Tea) + 1.1(Fewer) ≈ **5.3 h**
   - 拆卡（2 卡 Full + 1 卡 ours + 1 卡三竞品）= max(4.4, 2.3, 9.0) ≈ **9.0 h**
   卡时相同，但关键路径更长，所以不拆。
8. **竞品只跑单点，不跑 N=10/13/17**。竞品是阈值触发、NFE 自适应，本身没有 N 档；
   N 档只对 ours 有意义（用于质量-NFE 曲线）。三档预算若也要竞品，成本 ×3，不做。


## 7. 机器被回收后如何续跑

```bash
# 1. 环境（若新机器）
bash h20/RUNBOOK.md 第 2 节，切记补 cuBLAS 12.4（PITFALLS P1）

# 2. 代码 + prompt
git clone -b h20-run https://github.com/Yizhou-Yang/MPCache.git /root/MPCache
mkdir -p /root/viewdit/{experiments/video,results/video,logs,repro,resources/prompts}
cp /root/MPCache/h20/*.py /root/viewdit/experiments/video/
cp /root/MPCache/resources/prompts/prompt_video.txt /root/viewdit/resources/prompts/

# 3. 权重（HF 直下，约 80 GB）
export HF_HUB_ENABLE_HF_TRANSFER=1
hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers --local-dir /root/viewdit/weights/Wan2.1-T2V-1.3B
hf download hunyuanvideo-community/HunyuanVideo --local-dir /root/viewdit/weights/HunyuanVideo
hf download zai-org/CogVideoX-2b --local-dir /root/viewdit/weights/CogVideoX-2b

# 4. 从仓库恢复已完成的数据（避免重跑）
cp -r /root/MPCache/repro/results/video/* /root/viewdit/results/video/

# 5. 续跑（closed.jsonl 去重，自动跳过已完成）
bash h20/RUNBOOK.md 第 9 节的循环
```

## 8. 结果目录约定

```
/root/viewdit/results/
  video/<run>/closed.jsonl          # rel 口径的行
  video/<run>/zfull_{pi}_{seed}.pt  # 全精确基准缓存
  ours_wan/tea_gap_n13/videos/      # ours 的 mp4（video_%06d.mp4）
  ra_wan/wan_<mode>/videos/         # RA 各方法的 mp4
/root/MPCache/repro/results/video/<run>/{closed.jsonl,manifest.json}   # 回传
```

命名：`<model><frames>_<config>_<date>a`，如 `wan81_n13_20261022a`（已删）、`hy49_20261022a`。
