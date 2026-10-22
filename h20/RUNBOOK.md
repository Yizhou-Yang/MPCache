# H20 标准初始化手册（v3，2026-10-22）

> **每次拿到新的 H20 机器，从第 1 步照抄到最后一步即可。**
> 本机实测条件：4×H20-97GB、驱动 535.247.01（CUDA 12.2）、网络经公司代理全通（GitHub/PyPI/HF）、系统盘 7.8T 可用。

---

## 0. 本机事实（已实测，不要再改）

| 项 | 值 | 影响 |
|---|---|---|
| GPU | 4 × H20 97 GB | 可 4 卡分片，49 帧任务约 2 小时 |
| 驱动 | 535.247.01 / **CUDA 12.2** | torch 最高 **2.5.1+cu121**（cu124 需驱动 ≥550） |
| 系统 python | 3.9.16，torch 2.1.2+cu121，diffusers **0.26.0** | diffusers 太老，**不能用**；必须建新环境 |
| `/root/miniconda3` | **不存在** | 第 2 步先装 |
| 网络 | 全通（含 HF），5.4 MB/s | 权重直接下载，无需中转 |
| Wan 仓库 | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | 本机主跑 |

---

## 1. 探测（30 秒，确认机器状态没变）

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
for u in https://github.com https://pypi.org https://huggingface.co; do
  printf "%-26s " "$u"; curl -s -m 8 -o /dev/null -w "%{http_code}\n" "$u"
done
df -h /root | tail -1
```

**期望**：4 行 H20、三个 200、≥100 GB 可用。若 HF 不通，改用备用：
`export HF_ENDPOINT=https://hf-mirror.com`。

---

## 2. 建环境（一次，约 10–15 分钟）

```bash
# 2.1 Miniconda
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
bash /tmp/mc.sh -b -p /root/miniconda3

# 2.2 环境（版本已按驱动 12.2 钉死，不要升级）
/root/miniconda3/bin/conda create -y -n racfg python=3.10
PIP=/root/miniconda3/envs/racfg/bin/pip
$PIP install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
$PIP install diffusers==0.37.0 transformers==4.57.1 accelerate safetensors numpy sentencepiece protobuf \
  imageio imageio-ffmpeg opencv-python-headless einops ftfy regex "huggingface_hub[cli]" hf_transfer
```

**检查点**

```bash
/root/miniconda3/envs/racfg/bin/python - <<'EOF'
import torch, diffusers, transformers
print("torch", torch.__version__, "cuda avail", torch.cuda.is_available(), "devs", torch.cuda.device_count())
print("diffusers", diffusers.__version__, "transformers", transformers.__version__)
from diffusers import WanPipeline, WanTransformer3DModel
from transformers import AutoTokenizer
print("imports ok")
EOF
```

必须打印 `cuda avail True`、`devs 4`、`imports ok`。
若 `diffusers 0.37` 与 `transformers 4.57` 冲突，降级：`$PIP install diffusers==0.35.1`（对 Wan 无影响）。

---

## 3. 代码（GitHub 分支 `h20-run`）

```bash
cd /root && rm -rf MPCache && git clone -b h20-run https://github.com/Yizhou-Yang/MPCache.git
mkdir -p /root/viewdit/{experiments/video,results/video,logs,repro}
cp /root/MPCache/h20/*.py /root/viewdit/experiments/video/
cp /root/MPCache/h20/*.sh /root/viewdit/repro/ 2>/dev/null || true
cp /root/MPCache/h20/RUNBOOK.md /root/viewdit/
ls /root/viewdit/experiments/video/    # 应有 mpcache_controller.py、step_oracle.py

# prompt 列表（论文协议）：必须与 RA-CFGCache 的 resources/prompts/prompt_video.txt
# 逐行一致，否则 ours 与 baseline 的视频索引对不上，LPIPS/SSIM/PSNR 不可比。
mkdir -p /root/viewdit/resources/prompts
cp /root/MPCache/resources/prompts/prompt_video.txt /root/viewdit/resources/prompts/
python3 - <<'EOF'
lines = [l.strip() for l in open('/root/viewdit/resources/prompts/prompt_video.txt') if l.strip()]
print('prompts:', len(lines))
print('p0:', lines[0][:60])
print('p99:', lines[-1][:60])
EOF
# 期望：prompts: 100；p0 以 "a dilapidated phone booth" 开头；
# p99 = "an orange on the bottom of a carrot, front view"。
# 三个 runner 的查找顺序：$MPCACHE_PROMPTS -> /root/viewdit/resources/prompts/prompt_video.txt
# -> 脚本同目录 prompts_100.txt -> 内置 8 条（仅兼容旧 run）。启动日志会打印
# "[prompts] 100 from ..."，没有这一行说明路径没对上。
```

---

## 4. 权重（Wan2.1-T2V-1.3B，约 27 GB / 15 分钟）

```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
/root/miniconda3/envs/racfg/bin/hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --local-dir /root/viewdit/weights/Wan2.1-T2V-1.3B-Diffusers
ls /root/viewdit/weights/Wan2.1-T2V-1.3B-Diffusers/model_index.json
```

> CogVideoX-2B **不需要**（Cog 全部在 V100 上跑）。
> HunyuanVideo 在阶段 D 再下（约 60 GB）。

## 5. Wan 冒烟（2 分钟，必须先过）

```bash
cd /root/viewdit
CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/racfg/bin/python experiments/video/wan_smoke.py \
  --frames 33 --height 256 --width 256 --steps 20
```

**必须看到**：`loaded in ...s`、然后一行 JSON：`seconds / per_step / shape / finite / peak_gib`。
`finite` 为 true、shape 5 维、peak_gib < 90 才算过。失败就把最后 20 行贴我。

## 6. H20 的任务编排（Wan + Hunyuan，不是 Cog）

**H20 只跑 Wan2.1-T2V-1.3B 与 HunyuanVideo；CogVideoX 全部留在 V100。**

### 6.0 阶段顺序（不要跳步）

| 阶段 | 内容 | 依赖 |
|---|---|---|
| A | 环境 + **Wan stock 冒烟**（`wan_smoke.py`） | 无 |
| B | Wan 缓存 runner（`wan_cache.py`，我按 A 的反馈写） | A 通过 |
| C | Wan 全量：N=10/13/17 × 4 卡分片 | B 通过 |
| D | HunyuanVideo 冒烟 + runner | C 稳定 |
| E | HunyuanVideo 全量 | D 通过 |

### 6.1 阶段 A：Wan 权重 + 冒烟

```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
/root/miniconda3/envs/racfg/bin/hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --local-dir /root/viewdit/weights/Wan2.1-T2V-1.3B-Diffusers

cd /root/viewdit
CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/racfg/bin/python experiments/video/wan_smoke.py \
  --frames 33 --height 256 --width 256 --steps 20
```

把输出贴给我（`seconds / per_step / shape / finite / peak_gib`）。
**通过标准**：finite=true、shape 为 5 维、peak_gib<90、无异常。

### 6.2 阶段 C：Wan 全量（4 卡）

```bash
cd /root/viewdit
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i setsid nohup /root/miniconda3/envs/racfg/bin/python \
    experiments/video/wan_cache.py --frames 81 --height 480 --width 832 --steps 50 \
    --nprompt 25 --prompt-offset $((i*25)) --seeds 3000 --Ns 10,13,17 \
    --planners reuse,ex1,tea_gap \
    --out /root/viewdit/results/video/wan_p$i > logs/wan_p$i.log 2>&1 < /dev/null &
done
```

每片 225 行（25 prompt × 3 预算 × 3 臂），四片合计 **900 行**。

### 6.3 阶段 D：HunyuanVideo

```bash
/root/miniconda3/envs/racfg/bin/hf download hunyuanvideo-community/HunyuanVideo \
  --local-dir /root/viewdit/weights/HunyuanVideo
```

冒烟脚本我会照 Wan 的模式写；正式配置 49 帧 / 480×832 / 50 步。

### 6.4 4 卡编排原则

1. 一个进程一张卡（`CUDA_VISIBLE_DEVICES` 显式绑定）；
2. **按 prompt 分片**（`--prompt-offset`），不要按 seed 分片；
3. 每片独立 out 目录；
4. 跑前 `nvidia-smi` 确认卡是空的。

## 7. 结果回传

```bash
cd /root/viewdit
tar czf /root/results_cog49.tgz results/video/cog49_p* results/video/smoke_h20 logs/cog49_p*.log
ls -lh /root/results_cog49.tgz

# 本机执行（把 <H20> 换成登录串）
scp <H20>:/root/results_cog49.tgz ~/Desktop/
```

放到桌面我直接出统计。

---

## 8. 排错表

| 现象 | 原因 / 处理 |
|---|---|
| `CUDA driver version is insufficient` | torch 装成了 cu124/cu128；重装 `torch==2.5.1 --index-url .../cu121` |
| `ModuleNotFoundError: numpy` | 用了系统 python；一律 `/root/miniconda3/envs/racfg/bin/python` |
| `ImportError: WanPipeline / WanTransformer3DModel` | diffusers 太老（系统自带 0.26）；确认用的是新环境（0.37） |
| `ImportError: T5Tokenizer` | transformers ≥5 移除了 slow tokenizer；钉 4.57.1 |
| HF 下载卡住 | 重跑同一条 `hf download`（自动续传）；或 `export HF_ENDPOINT=https://hf-mirror.com` |
| 任务是否活着 | `ps -eo pid,etimes,cmd \| grep cog49`（**不要用 `pgrep -f`**，会匹配 ssh 自身） |
| 中途停 | `pkill -f "cog49_p"`；JSONL 追加写，已完成不丢 |
| 分片重叠/漏 prompt | 检查 `--prompt-offset`：0/25/50/75，各 `--nprompt 25` |
| 显存不足 | Wan-1.3B 约 25 GB/卡；先跑 `wan_smoke.py --frames 33 --height 256` 冒烟 |

---

## 9. 每次开机的极简版（环境与权重都在时）

```bash
cd /root/viewdit
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i setsid nohup /root/miniconda3/envs/racfg/bin/python \
    experiments/video/wan_cache.py --frames 81 --height 480 --width 832 --steps 50 \
    --nprompt 25 --prompt-offset $((i*25)) --seeds 3000 --Ns 10,13,17 \
    --planners reuse,ex1,tea_gap \
    --out /root/viewdit/results/video/wan_p$i > logs/wan_p$i.log 2>&1 < /dev/null &
done
```

## 10. 我这边（V100）同时在跑的

- 组件优化：TRUST>1、偏置项等消融，结果会同步进 `repro/ABLATIONS.md`；
- 论文：`对比/MPCache_main.pdf` 为最新快照，8 页、0 overfull。

H20 有任何输出/报错，直接贴给我。

---

## 11. Wan2.1 / HunyuanVideo（新增 runner，2026-10-22 实测）

`mpcache_wan.py`、`mpcache_hunyuan.py` 复用同一套 `StateModel` / planner，只替换模型、
VAE 与 latent 布局；两者的 patch 都是 `16x1x2x2 = 64` 维，和 CogVideoX 的通道空间一致，
planner 逻辑零改动。

```bash
# Wan2.1-T2V-1.3B，33 帧 480p，1.66 s/步（fp16，12.8 GiB）
python experiments/video/mpcache_wan.py --frames 33 --nprompt 34 --prompt-offset 0 \
  --seeds 3000 --Ns 10,13,17 --planners reuse,ex1,tea_gap,mpc \
  --out /root/viewdit/results/video/wan33_p0

# HunyuanVideo，49 帧 544x960，27 s/步（batch=2，42.5 GiB）
python experiments/video/mpcache_hunyuan.py --frames 49 --nprompt 4 \
  --seeds 3000 --Ns 10,13 --planners reuse,tea_gap,mpc \
  --out /root/viewdit/results/video/hy49
```

实测（H20 单卡）：

| 模型 | 配置 | 步时 | 显存 |
|---|---|---|---|
| CogVideoX-2b | 17 帧 480x720 | 0.75 s | 10.8 GiB |
| CogVideoX-2b | 49 帧 480x720 | 29.9 s | 9.6 GiB |
| Wan2.1-1.3B | 33 帧 832x480 | 1.66 s | 12.8 GiB |
| HunyuanVideo | 49 帧 544x960 | 27.1 s | 42.5 GiB |
| HunyuanVideo | 49 帧 832x480 | 17.3 s | 41.2 GiB |

### 三个必踩的坑

1. **cuBLAS 小 M GEMM 触发 SIGFPE**（driver 535 + cu121 + sm90）。报错为
   `Fatal Python error: Floating point exception`，栈顶在 `linear.py:125`，与数值无关。
   Wan 用 `torch_dtype=torch.float16` 即绕开（bf16 必崩）；Hunyuan fp16 仍崩在 adaLN（M=1），
   用 batch=2 复制输入绕开、输出取 `[:1]`。
2. **Wan 默认 UniPC 是二阶**，跳步时二阶历史被伪造，N=13/50 的 reuse 误差 0.65；换一阶
   `FlowMatchEuler` + `--sched uniform` 后降到 0.30。Cog / Hunyuan 本来就是一阶，三者可比。
3. **内置 prompt 只有 8 条**，`--nprompt 100` 会在 `embs[8]` 崩。控制器现在优先加载同目录
   `prompts_100.txt`，并按 `[offset, offset+nprompt)` 只编码本片需要的条目。

---

## 12. 论文评估协议（H20 侧，必读）

主表 `tab_main` 的指标**不是** latent rel L2，而是对 50 步 Full 视频算的：

```
LPIPS(AlexNet) / SSIM / PSNR   +   Speedup
```

协议（与 `paper/sections/experiments.tex` 逐项对应）：

| 项 | 值 |
|---|---|
| prompts | `resources/prompts/prompt_video.txt`（RA-CFGCache 的 100 条） |
| seed | **= prompt index**（RA 的 `--seed 0` 语义就是 `0 + global_video_idx`，等价于 `--seed-from-prompt`） |
| negative prompt | `"animation"` |
| Wan | 81 帧 832×480，16 fps，50 步 **UniPC**，flow shift 3，CFG 5 |
| Cog | 49 帧 720×480，8 fps，50 步 DDIM，CFG 6 |

> **决策（2026-10-22，已修正）**：所有臂用 **seed = prompt index**。RA 的实现是 `manual_seed(opts.seed + global_video_idx)`，所以 `--seed 0` 等价于“第 k 个视频用 seed k”，与 `--seed-from-prompt` 逐位相同。证据：`video_000025.json` 里 `"seed": 25`。曾一度误改为“固定 seed 0”，会破坏与已跑 baseline 的配对，已回滚。
| Hunyuan | 49 帧 480×832，50 步，embedded guidance 6 |

### 12.1 接入 RA-CFGCache 基线（首次）

```bash
cd /root && git clone --depth 1 https://github.com/yiming-l21/RA-CFGCache.git
/root/miniconda3/envs/racfg/bin/pip install nvidia-cublas-cu12==12.4.5.8   # 关键，见 PITFALLS P1
/root/miniconda3/envs/racfg/bin/pip install lpips scikit-image open-clip-torch clean-fid

cd /root/RA-CFGCache
CUDA_VISIBLE_DEVICES=0,1,2,3 bash RUN/demo_wan.sh   --mode original --limit 100 --gpus 0,1,2,3   --python /root/miniconda3/envs/racfg/bin/python   --model_path /root/viewdit/weights/Wan2.1-T2V-1.3B   --output_dir /root/viewdit/results/ra_wan   --negative_prompt "animation" --run-name full
```

`--mode` 可选：`original`（Full）、`TeaCache`（配 `--rel_l1_thresh 0.15`）、
`CFGCache`（配 `--rel_l1_thresh 0.6/1.4` 得 Slow/Fast）。
`--num_steps 25 --mode original` 即 Fewer steps 那一行。

输出先落 `<out>/.multi_gpu_tmp/wan_<mode>_<ts>/gpu*/`，四片都结束后才 merge 到
`<out>/wan_<mode>/`。**推送脚本要同时扫临时目录**（PITFALLS P9）。

### 12.2 ours（tea_gap，N=13）

```bash
for i in 0 1 2 3; do
CUDA_VISIBLE_DEVICES=$i setsid nohup /root/miniconda3/envs/racfg/bin/python   /root/viewdit/experiments/video/mpcache_wan.py   --frames 81 --height 480 --width 832   --nprompt 25 --prompt-offset $((i*25)) --seeds 0 --seed-from-prompt   --negative "animation" --solver unipc --sched sigma   --Ns 13 --planners tea_gap --no-full   --save-video --save-op tea_gap --save-n 13   --video-dir /root/viewdit/results/ours_wan/tea_gap_n13/videos   --out /root/viewdit/results/ours_wan/n13_p$i > /root/viewdit/logs/ours_n13_p$i.log 2>&1 &
done
```

`--no-full` 很重要：参考视频直接用 RA 的 Full，否则每 prompt 多付 50 步。

### 12.3 指标

```bash
cd /root/RA-CFGCache
REF_DIR=/root/viewdit/results/ra_wan/wan_original CMP_DIR=/root/viewdit/results/ours_wan/tea_gap_n13 PROMPT_FILE=/root/viewdit/resources/prompts/prompt_video.txt DEVICE=cuda:0 bash evaluation/eval.sh
```

按 `video_%06d.mp4` 的文件名索引配对，所以两边命名必须一致（ours 已对齐）。

### 12.4 已有参考值（论文 `tab_main`，Wan2.1-T2V-1.3B）

| Method | LPIPS↓ | SSIM↑ | PSNR↑ | Speedup↑ |
|---|---|---|---|---|
| Full (T=50) | 0.0000 | 1.0000 | ∞ | 1.00× |
| Fewer (T=25) | 0.4204 | 0.5700 | 15.85 | 1.96× |
| TeaCache | 0.1681 | 0.7887 | 22.84 | 2.53× |
| TaylorSeer (N=3,O=1) | 0.3496 | 0.6283 | 17.10 | 2.47× |
| RA-CFGCache-Slow | 0.0941 | 0.8820 | 28.11 | 2.65× |
| RA-CFGCache-Fast | 0.1352 | 0.8420 | 26.14 | 3.14× |
| ours | — | — | — | — |

HunyuanVideo 整行在论文里还是 `--`，H20 上要补。

---

## 13. 踩坑与进度

- **踩坑清单（13 条，含症状/根因/修复）**：`h20/PITFALLS.md`
- **进度快照与 TODO**：`h20/STATUS.md`
- **自动推送**：`repro/push_results.sh`（每 10 min，合并分片 → commit → push）
- **无人值守编排**：`repro/after_ra_full.sh`、`repro/after_wan.sh`

新机器从零开始的完整顺序：第 0 → 1 → 2 → **2.5 补 cuBLAS 12.4** → 3 → 4 → 5 → 12.1 → 12.2 → 12.3。
