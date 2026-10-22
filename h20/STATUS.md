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

## 3. 已有数据（rel 口径，只能作消融）

| run | 内容 | 行数 | 口径 |
|---|---|---|---|
| `wan33_20261022a` | Wan 33f，60 prompt × Ns 10/13/17 × {reuse,ex1,tea_gap,mpc} | 699 | latent rel L2，seed 3000，非 VBench |
| `hy49_pilot_20261022a` | Hunyuan 49f，4 prompt × 2 Ns × 3 臂 | 18 | 同上 |
| `wan81_n13_20261022a` | Wan 81f N=13，4 臂 | 4 | 已放弃（口径不符，停在第一个 prompt） |

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

放弃（9.5 h 内做不完）：RA-CFGCache-Fast、TaylorSeer、Hunyuan 的 4 个竞品行、Cog 在 H20 的重跑。

## 6. 关键决策记录

1. **主表用 UniPC**（与 RA 基线同 solver），Euler 只作消融——否则二阶/一阶混比不可比。
2. **seed = prompt index**（`--seed 0` + `start_index`），不是固定 3000。
3. **negative prompt = "animation"**，三个模型统一。
4. **ours 跳过 50 步全精确**（`--no-full`）：fidelity 的参考视频就是 RA 的 Full，
   自己再跑一遍既慢又无意义。
5. **Cog 不在 H20 跑**：49 帧 29.9 s/步，V100 已有结果。
6. **mpc 臂的负结果保留**，不删除、不隐藏，作为 planner 与 flow matching 不匹配的证据。

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

命名：`<model><frames>_<config>_<date>a`，如 `wan81_n13_20261022a`、`hy49_20261022a`。
