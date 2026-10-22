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

## 2.1 方法定义（2026-10-22 起，默认组件）

```
ours = tea_tok --token-gate 0.5
```

- 用最近 3 个精确锚点估逐 token 速度/加速度：`v1=(x1-x2)/dt1`、`v2=(x2-x3)/dt2`、
  `a=(v1-v2)/dt1`，曲率 `κ = ‖a‖ / ‖v1‖²`（per token，64 维通道空间）；
- `κ` 高于分位数 τ（默认 0.5）的 token 用 state-model forecast，其余直接复用 `s['v']`；
- `tea_gap`（对所有 token 统一修正）降级为**消融臂**，用于证明 per-token 分类的必要性。

依据：V100 `vbA`（VBench 100 prompts）上 `tea_gap` 增益 +7.0 / +3.0 / +3.8（N=10/13/17，
胜率 1.00，全部 >2SEM）；去掉 gate 后 **−4.7 / −3.8 / −8.4**（胜率 0.25–0.31）。
门控是必要组件，而"逐 token 分类"是它最自然的下一步。

待办：τ 的敏感性（0.3/0.5/0.7），当前 0.5 是按 WorldCache 异质分类的中位数直觉取的，未扫。

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

## 5. TODO（按优先级，剩余 ~8.5 h）

| # | 任务 | 命令要点 | 耗时 | 状态 |
|---|---|---|---|---|
| 1 | RA Full (T=50) × 100 | `demo_wan.sh --mode original` | 2.3 h | 跑中 |
| 2 | **ours** (tea_gap N=13) × 100 | `mpcache_wan.py --no-full --save-video` | 0.6 h | 自动接续 |
| 3 | RA CFGCache-Slow | `--mode CFGCache --rel_l1_thresh 0.6` | 0.9 h | 排队 |
| 4 | RA TeaCache | `--mode TeaCache --rel_l1_thresh 0.15` | 0.9 h | 排队 |
| 5 | RA Fewer (T=25) | `--num_steps 25 --mode original` | 1.1 h | 排队 |
| 6 | Hunyuan Full + ours | 10 prompt × 2 臂 | 0.8 h | 排队 |
| 7 | 指标 + 合并 + 推送 | `evaluation/eval.sh` | 0.7 h | 排队 |
| 8 | Euler ablation（可选） | `--solver euler --sched uniform` | 0.6 h | 有余额才跑 |
| 9 | **Cog 49f 100 prompt 主表** | `mpcache_controller.py --frames 49 --seed-from-prompt` | **~8 h** | 下一台机器 |

放弃（本机 9.5 h 内做不完）：RA-CFGCache-Fast、TaylorSeer、Hunyuan 的 4 个竞品行。

### 5.1 Cog 100-prompt 主表（已排进 TODO，不在本机跑）

实测 49 帧 480×720 是 **29.9 s/步**（70k tokens × 30 层，GPU 无降频），所以：

按 29.9 s/步、100 prompt、4 卡：

| 配置 | GPU·h | 4 卡墙钟 |
|---|---|---|
| Full（参考，只跑一次） | 41.5 | **10.4 h** |
| ours（13 步） | 10.8 | 2.7 h |
| Full + ours | 52.3 | **13.1 h** |
| 5 臂全量 | 84.7 | **21.2 h** |

对比 Wan 81 帧（6.4 s/步）：Full 8.9 GPU·h / 2.2 h，5 臂 18.1 GPU·h / 4.5 h。
Cog 49 帧是 Wan 的 4.7 倍单步成本，**单 Full 就要 10.4 h**，这是它排不进本机的根本原因。

注意不要用 17 帧的 0.75 s/步去估 Cog 49 帧：那会得到"25 分钟跑完"的错误结论，
两者差 40 倍（17f 5 个 latent vs 49f 13 个 latent，attention 二次增长）。

本机剩余时间 < 8 h，且要留给 Wan 竞品与 Hunyuan，因此不在本机排。
下一台 H20 的**第一个**任务就跑它，命令：

```bash
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i setsid nohup /root/miniconda3/envs/racfg/bin/python \
    /root/viewdit/experiments/video/mpcache_controller.py \
    --frames 49 --nprompt 25 --prompt-offset $((i*25)) \
    --seeds 0 --seed-from-prompt --negative "animation" \
    --Ns 13 --planners tea_gap \
    --out /root/viewdit/results/video/cog49_p$i > /root/viewdit/logs/cog49_p$i.log 2>&1 &
done
```

注意 `mpcache_controller.py` 目前还没有 `--save-video`，要补 VAE 解码才能算 LPIPS/SSIM/PSNR；
`mpcache_wan.py` 里的 `save_video()` + AutoencoderKL 用法可以直接照搬（Cog 用 `AutoencoderKLCogVideoX`，
8×8 空间压缩、4× 时间压缩，解码 49 帧 720×480 实测约 8 s / 3 GiB）。

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
