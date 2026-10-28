# 部署与测试手册（hy4 / 4×H20，2026-10-28 实战版）

> 目标：一台全新的 H20 机器，从零到能跑出与基线**同源可比**的视频指标，约 1.5 小时。
> 每一步都写了“检查点”，不通过不要往下走。

---

## 0. 机器事实

| 项 | 值 |
|---|---|
| GPU | 4× H20 97GB |
| 系统 | TLinux，root 直接可用，无需 sudo |
| 网络 | 全通（GitHub / PyPI / HF），走公司代理 |
| 磁盘 | /root 约 8T 可用 |
| 关键坑 | **RA 脚本内部调用裸 `python`**，必须把 conda bin 放进 PATH |

---

## 1. 环境（约 15 分钟）

```bash
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
bash /tmp/mc.sh -b -p /root/miniconda3
/root/miniconda3/bin/conda create -y -n racfg python=3.10
PIP=/root/miniconda3/envs/racfg/bin/pip

$PIP install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
$PIP install diffusers==0.37.0 transformers==4.57.1 accelerate safetensors numpy sentencepiece \
    protobuf imageio imageio-ffmpeg opencv-python-headless einops ftfy regex lpips \
    "huggingface_hub[cli]" hf_transfer

# 关键：cublas 必须最后装，且版本 12.4.5.8；后续任何 pip 都可能把它降级 → SIGFPE
$PIP install nvidia-cublas-cu12==12.4.5.8 --force-reinstall --no-deps
```

**检查点**

```bash
/root/miniconda3/envs/racfg/bin/python - <<'PY'
import torch, diffusers, transformers
print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())
print(diffusers.__version__, transformers.__version__)
from diffusers import CogVideoXTransformer3DModel, AutoencoderKLCogVideoX
from transformers import T5EncoderModel, T5Tokenizer
print("imports ok")
PY
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
```

---

## 2. 权重（约 40 分钟，HF 全通）

```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
W=/root/viewdit/weights; mkdir -p $W
/root/miniconda3/envs/racfg/bin/hf download zai-org/CogVideoX-2b          --local-dir $W/CogVideoX-2b
/root/miniconda3/envs/racfg/bin/hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers --local-dir $W/Wan2.1-T2V-1.3B
# 需要 Hunyuan 时（约 60G）
/root/miniconda3/envs/racfg/bin/hf download hunyuanvideo-community/HunyuanVideo --local-dir $W/HunyuanVideo
```

**检查点**：`du -sh $W/*`（Cog≈13G、Wan≈27G）；`ls $W/CogVideoX-2b/model_index.json`。

> 下载中断不要删目录，重跑同一条命令即可续传。

---

## 3. 代码

```bash
cd /root && git clone https://github.com/Yizhou-Yang/MPCache.git
cd /root/viewdit 2>/dev/null || mkdir -p /root/viewdit
# 我们的代码按约定放在 /root/viewdit/{experiments,results,logs,resources}
cp -r /root/MPCache/h20/. /root/viewdit/
mkdir -p /root/viewdit/{experiments/video,results,logs}
```

**检查点**：`sha256sum /root/viewdit/experiments/video/mpcache_controller.py` 应与 `h20/TODO_MAIN_TABLE.md` 里记录的一致。

---

## 4. 基线（RA-CFGCache）

```bash
cd /root && git clone https://github.com/yiming-l21/RA-CFGCache.git
ls /root/RA-CFGCache/calibration/      # 应有 wan_rho_cfg5.npz, cogvideox_rho_cfg6.npz
```

**关键**：调用前必须

```bash
export PATH=/root/miniconda3/envs/racfg/bin:$PATH
```

否则 `demo_*.sh` 报 `python: command not found`（P26）。

---

## 5. 测试（每次部署必做）

### 5.1 一步耗时与显存

```bash
cd /root/viewdit
CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/racfg/bin/python experiments/video/mpcache_wan.py \
  --sanity --frames 33 --nprompt 1 --out /tmp/sanity
```

期望：`[cfg] warmup=3 damp=1.0 ...`、`sanity roundtrip 0.0`、`one CFG step x.xxs`。
参考值（H20 单进程空载）：Wan-1.3B 81f ≈ 6.8s/step；Cog-2B 17f ≈ 1.5s/step；**Cog-2B 49f ≈ 46–56s/step**（异常慢，见 P27）。

### 5.2 解码验证（**不要相信 rel**）

```bash
# 保存视频后，检查首帧 std：<40 基本是纯色/坏视频
python -c "import imageio.v2 as iio,numpy as np;a=iio.get_data('<某视频>',0);print(a.std())"
```

### 5.3 同源性验证（最重要）

我们的全量 50 步 vs 基线的全量：**Wan 应 LPIPS≈0.035**；Cog 目前 **0.674（未对齐，见 P22）**。

```bash
python /root/MPCache/repro/metrics.py --ref <RA Full 目录> --cmp <我们的 Full 目录> --out-json /tmp/x.json
```

参考视频**不随代码发布**，换机器必须重跑（Wan 约 2.2h/100 prompt）。

---

## 6. 跑实验的模板

```bash
P=/root/miniconda3/envs/racfg/bin/python
cd /root/viewdit
# Wan ours（当前方法：tea_gap）
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i setsid nohup $P experiments/video/mpcache_wan.py \
    --seed-from-prompt --negative animation --solver unipc --sched uniform --frames 81 \
    --no-full --save-video --save-op tea_gap --save-n 19 --warmup 3 --Ns 19 \
    --planners tea_gap --nprompt 25 --prompt-offset $((i*25)) \
    --out results/wan_tg19 > logs/wan_tg19_p$i.log 2>&1 < /dev/null &
done
```

**每次启动后 30 秒必须二次确认进程存活**（P25）。

---

## 7. 排错表

| 现象 | 原因 / 处理 |
|---|---|
| `python: command not found` | PATH 没导出（P26） |
| 四卡同时 SIGFPE (-8) | cublas 被降级 → 重装 12.4.5.8（P18） |
| rel 全 NaN | 用了 `--no-full`，正常；真指标是 LPIPS |
| 视频纯色/棋盘 | 跳过了高噪声段（warmup）或调度间隔过大（P24） |
| 同 seed 内容不同 | 初始化布局或文本 mask 不一致（P22） |
| 进程莫名消失 | 后台任务被回收，改短任务 + 二次确认（P25） |
| Cog 49f 极慢 | 该组合固有（46–56s/step），非争抢（P27） |
