# H20 执行清单（2026-10-27）

## 0. 方法状态（已确认，可直接开跑）

| 项 | 值 |
|---|---|
| 控制器 | `experiments/video/mpcache_controller.py`，sha256 `baeb0e61…`（本地 = H20 部署版） |
| Cog 上的方法 | 对角在线模型 + 门控 `age ≥ 2`；`--auto-base 1` 在 Cog 上是**空操作**（48/48 逐位相同，已验证） |
| Wan 上的方法 | **均匀调度 → 纯一阶外推（ex1）**，不做模型修正。所以 Wan 的 ours 行 = `--save-op ex1` |
| Hunyuan | 端口当前产出 **NaN**，未修复前不能进主表 |

一句话：**Cog 跑 `tea_gap`，Wan 跑 `ex1`**。这两条分别对应方法里 “Which forecaster to trust” 的两支。

---

## 1. 主表状态

| Backbone | Full | TeaCache | TaylorSeer | RA-Slow | RA-Fast | ours |
|---|---|---|---|---|---|---|
| CogVideoX-2B | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Wan2.1-T2V-1.3B | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ tea_gap 有，需换 ex1 |
| HunyuanVideo-13B | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

（基线 LPIPS/SSIM/PSNR 已在 `paper/sections/tab_main.tex`）

---

## 2. 第一件事：口径校验（10 分钟，必须最先做）

ours 是**对我们的 full50** 评测的，基线是**对 RA Full** 评测的。先确认两者等价：

```bash
python /root/RA-CFGCache/scripts/eval.py \
  --ref /root/viewdit/results/ra_wan/wan_original_full/videos \
  --cmp /root/viewdit/results/ours_wan/full50/tea_gap/videos \
  --out /tmp/full_vs_full.csv
python - <<'PY'
import csv, statistics
rows=list(csv.DictReader(open('/tmp/full_vs_full.csv')))
lp=[float(r['lpips']) for r in rows]; ps=[float(r['psnr']) for r in rows]
print("n=%d mean LPIPS=%.5f mean PSNR=%.2f" % (len(rows), statistics.mean(lp), statistics.mean(ps)))
PY
```

- **LPIPS < 0.01 且 PSNR > 35** → 两个 Full 等价，现有 ours 数字可直接用；
- 否则 → ours 必须对 `ra_wan/wan_original_full` 重新评测，否则主表两列不同源。

---

## 3. 跑什么（按优先级）

### 3.1 Wan：ex1 臂（≈0.6 h，4 卡）

```bash
P=/root/miniconda3/envs/racfg/bin/python
cd /root/viewdit
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i setsid nohup $P mpcache_wan.py \
    --seed-from-prompt --negative "animation" --solver unipc --sched sigma \
    --no-full --save-video --save-op ex1 --save-n 13 \
    --nprompt 25 --prompt-offset $((i*25)) \
    --out results/wan_arms/ex1 > logs/wan_ex1_p$i.log 2>&1 < /dev/null &
done
```

跑完对 **RA Full** 评测：

```bash
python /root/RA-CFGCache/scripts/eval.py \
  --ref results/ra_wan/wan_original_full/videos \
  --cmp results/wan_arms/ex1/videos \
  --out results/metrics/wan_ex1.csv
```

判定：LPIPS ≈ 0.11 → 赢 RA-Fast（0.135），仍输 RA-Slow（0.094）。

### 3.2 Cog 49 帧（≈0.9 h，4 卡）

```bash
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i setsid nohup $P experiments/video/mpcache_controller.py \
    --frames 49 --nprompt 25 --prompt-offset $((i*25)) --seed-from-prompt \
    --Ns 13 --planners reuse,tea_gap --auto-base 1 \
    --out results/video/cog49_p$i > logs/cog49_p$i.log 2>&1 < /dev/null &
done
```

控制器输出潜变量 rel，主表要 LPIPS/SSIM/PSNR → 需保存视频，再用 RA eval 对 Cog 的 50 步 Full 评测。

### 3.3 Hunyuan：先修 NaN

现状（`hy49_s480.log`）：`rel=nan`，β 掉到 0，步速 17.27 s/step、显存 41 GiB。

1. `--nprompt 1 --sanity` 定位 NaN 出现在文本编码 / VAE / transformer；
2. 若 fp16 溢出 → bf16 + fp32 累加 + latent clamp；
3. 修好后 Full ≈ 6 h/4 卡，13 步臂 ≈ 1.6 h。

---

## 4. 排期（4 卡）

| 时段 | GPU 0-1 | GPU 2-3 |
|---|---|---|
| 0-0.2 h | 口径校验 | 同左 |
| 0.2-0.8 h | Wan ex1（50 prompt） | Wan ex1（50 prompt） |
| 0.8-1.7 h | Cog 49f Full | Cog 49f ours N=13 |
| 1.7-3 h | Cog + Wan 评测 | Hunyuan NaN 定位 |

约 **3 小时**补齐 Cog/Wan 主表行；Hunyuan 另算。

---

## 5. 验收标准

| 检查 | 期望 |
|---|---|
| Wan ex1 LPIPS | 优于 tea_gap 的 0.1375，目标 ≤ 0.12 |
| Cog ours LPIPS | 与 RA-Slow（0.0990）同量级或更好 |
| 主表每格 | 有数或 `--`，不估算 |
| 加速比 | 实测 wall-clock，不用 NFE 比 |
