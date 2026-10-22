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
| CogVideoX 仓库 | `THUDM/CogVideoX-2b` → **`zai-org/CogVideoX-2b`** | 用新名字 |

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
from diffusers import CogVideoXTransformer3DModel, CogVideoXDDIMScheduler
from transformers import T5Tokenizer
print("imports ok")
EOF
```

必须打印 `cuda avail True`、`devs 4`、`imports ok`。
若 `diffusers 0.37` 与 `transformers 4.57` 冲突，降级：`$PIP install diffusers==0.35.1`（对 CogVideoX 无影响）。

---

## 3. 代码（GitHub 分支 `h20-run`）

```bash
cd /root && rm -rf MPCache && git clone -b h20-run https://github.com/Yizhou-Yang/MPCache.git
mkdir -p /root/viewdit/{experiments/video,results/video,logs,repro}
cp /root/MPCache/h20/*.py /root/viewdit/experiments/video/
cp /root/MPCache/h20/*.sh /root/viewdit/repro/ 2>/dev/null || true
cp /root/MPCache/h20/RUNBOOK.md /root/viewdit/
ls /root/viewdit/experiments/video/    # 应有 mpcache_controller.py、step_oracle.py
```

---

## 4. 权重（CogVideoX-2B，约 13 GB / 40 分钟）

```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p /root/viewdit/weights
/root/miniconda3/envs/racfg/bin/hf download zai-org/CogVideoX-2b \
  --local-dir /root/viewdit/weights/CogVideoX-2b
```

**检查点**

```bash
ls /root/viewdit/weights/CogVideoX-2b/model_index.json
du -sh /root/viewdit/weights/CogVideoX-2b          # ≈13G
ls /root/viewdit/weights/CogVideoX-2b | tr '\n' ' '  # transformer vae text_encoder tokenizer scheduler
```

> Wan 1.3B / HunyuanVideo 暂不下载：控制器目前只支持 CogVideoX，等 runner 写好再加。

---

## 5. 冒烟（2 分钟，必须先过）

```bash
cd /root/viewdit
CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/racfg/bin/python experiments/video/mpcache_controller.py \
  --frames 17 --nprompt 1 --seeds 3000 --Ns 13 \
  --planners reuse,tea_gap --out /root/viewdit/results/video/smoke_h20
```

**必须看到**：`sanity roundtrip 0.0` → `one CFG step ...` → 两行 `reuse` / `tea_gap`（数值非 NaN）。
失败就把最后 20 行发我。

---

## 6. 主任务：49 帧 × 100 prompt（4 卡分片，约 2 小时）

4 张卡各跑 25 个 prompt，输出目录互不干扰：

```bash
cd /root/viewdit
for i in 0 1 2 3; do
  off=$((i*25))
  CUDA_VISIBLE_DEVICES=$i setsid nohup /root/miniconda3/envs/racfg/bin/python \
    experiments/video/mpcache_controller.py \
    --frames 49 --nprompt 25 --prompt-offset $off --seeds 3000 \
    --Ns 10,13,17 --planners reuse,tea_gap \
    --out /root/viewdit/results/video/cog49_p$i \
    > logs/cog49_p$i.log 2>&1 < /dev/null &
done
```

**进度**（每 15 分钟）：

```bash
for i in 0 1 2 3; do echo "p$i $(wc -l < /root/viewdit/results/video/cog49_p$i/closed.jsonl 2>/dev/null)/150"; done
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
```

每个分片 150 行（25 prompt × 3 预算 × 2 方法），**四片合计 600 行**即完成。

> 单卡跑法（不推荐，6–8 小时）：`--nprompt 100 --prompt-offset 0 --out .../cog49_prod`，去掉 `CUDA_VISIBLE_DEVICES` 循环。

---

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
| `ImportError: CogVideoXTransformer3DModel` | diffusers 太老（系统自带 0.26）；确认用的是新环境 |
| `ImportError: T5Tokenizer` | transformers ≥5 移除了 slow tokenizer；钉 4.57.1 |
| HF 下载卡住 | 重跑同一条 `hf download`（自动续传）；或 `export HF_ENDPOINT=https://hf-mirror.com` |
| 任务是否活着 | `ps -eo pid,etimes,cmd \| grep cog49`（**不要用 `pgrep -f`**，会匹配 ssh 自身） |
| 中途停 | `pkill -f "cog49_p"`；JSONL 追加写，已完成不丢 |
| 分片重叠/漏 prompt | 检查 `--prompt-offset`：0/25/50/75，各 `--nprompt 25` |
| 显存不足 | 49 帧 Cog-2B 峰值约 12 GB；先跑 `--nprompt 1` 冒烟 |

---

## 9. 每次开机的极简版（环境与权重都在时）

```bash
cd /root/viewdit
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i setsid nohup /root/miniconda3/envs/racfg/bin/python \
    experiments/video/mpcache_controller.py --frames 49 --nprompt 25 --prompt-offset $((i*25)) \
    --seeds 3000 --Ns 10,13,17 --planners reuse,tea_gap \
    --out /root/viewdit/results/video/cog49_p$i > logs/cog49_p$i.log 2>&1 < /dev/null &
done
```

就这三行。

---

## 10. 我这边（V100）同时在跑的

- 组件优化：TRUST>1、偏置项等消融，结果会同步进 `repro/ABLATIONS.md`；
- 论文：`对比/MPCache_main.pdf` 为最新快照，8 页、0 overfull。

H20 有任何输出/报错，直接贴给我。
