# H20 执行手册 v2（2026-10-22）— 只能连 GitHub 的空白机器

前提：机器只能访问 github.com，没有 HF / PyPI / 内网共享；你是通过 ssh 登上去的。
因此：**代码从 GitHub 拉，环境与权重从 V100 中转**。

---

## 第 1 步：连通性探测（30 秒，先做这个）

```bash
for u in https://github.com https://pypi.org https://huggingface.co https://hf-mirror.com; do
  printf "%-28s " "$u"; curl -s -m 8 -o /dev/null -w "%{http_code}\n" "$u"
done
python3 -c "import torch, diffusers; print('torch', torch.__version__, 'diffusers', diffusers.__version__)" 2>&1 | tail -1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h /root | tail -1
```

**把这段输出发给我**。三种情况对应三种后续：
- `pypi.org` 200 且 torch 已装 → 直接跳到第 3 步；
- `pypi.org` 200 但没装 → 第 2 步 A；
- `pypi.org` 不通 → 第 2 步 B（从 V100 搬 wheel）。

---

## 第 2 步：环境

### A. 能连 PyPI

```bash
/root/miniconda3/bin/conda create -y -n racfg python=3.10 || true
PIP=/root/miniconda3/envs/racfg/bin/pip
$PIP install -q torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
$PIP install -q diffusers==0.37.0 transformers==5.2.0 accelerate safetensors numpy sentencepiece protobuf imageio imageio-ffmpeg opencv-python-headless einops ftfy regex
```

（如果 `download.pytorch.org` 也不通，把 torch 那行去掉，改用第 2 步 B 的方式搬 wheel。）

### B. 不能连 PyPI —— 从 V100 搬环境

在 **V100** 上打包整个 conda 环境（约 3–5 GB）：

```bash
cd /root/miniconda3/envs
tar czf /root/racfg_env.tgz racfg
ls -lh /root/racfg_env.tgz
```

把它拉到你的本机，再推到 H20：

```bash
# 本机执行
scp <V100>:/root/racfg_env.tgz ~/Desktop/
scp ~/Desktop/racfg_env.tgz <H20>:/root/
```

H20 上解包：

```bash
mkdir -p /root/miniconda3/envs && cd /root/miniconda3/envs && tar xzf /root/racfg_env.tgz
# 修正绝对路径（tar 里带的是 /root/miniconda3/envs/racfg，一般可直接用）
/root/miniconda3/envs/racfg/bin/python -c "import torch, diffusers; print(torch.__version__, diffusers.__version__)"
```

> 环境里若出现 `libcudart` 之类缺失，先确认 H20 的驱动版本 ≥ CUDA 12.8（`nvidia-smi` 右上角）。

---

## 第 3 步：代码（GitHub，唯一的网络来源）

```bash
cd /root && git clone -b h20-run https://github.com/Yizhou-Yang/MPCache.git
mv /root/MPCache /root/mpcache_repo
ls /root/mpcache_repo/h20/
```

**检查点**：`h20/` 下有 `mpcache_controller.py`、`step_oracle.py`、`bootstrap_h20.sh`、`H20_RUNBOOK_20261022.md`。

控制器里把 `ROOT` 写死为 `/root/viewdit`，所以：

```bash
mkdir -p /root/viewdit/experiments/video /root/viewdit/results/video /root/viewdit/logs
cp /root/mpcache_repo/h20/*.py /root/viewdit/experiments/video/
cp /root/mpcache_repo/h20/*.sh /root/viewdit/repro/ 2>/dev/null || (mkdir -p /root/viewdit/repro && cp /root/mpcache_repo/h20/*.sh /root/viewdit/repro/)
```

---

## 第 4 步：权重（H20 连不上 HF，必须从 V100 搬）

我们只需要 **CogVideoX-2b（13 GB）** 就能跑今晚的任务。Wan / HunyuanVideo 等以后再搬。

### 4.1 如果 H20 能直连 V100（同内网）

```bash
# 在 H20 上
mkdir -p /root/viewdit/weights
scp -r <V100>:/root/viewdit/weights/CogVideoX-2b /root/viewdit/weights/
```

### 4.2 如果不能直连 —— 经你本机中转

```bash
# V100 上打包（13 GB，压缩后约 12 GB）
cd /root/viewdit/weights && tar czf /root/cog2b.tgz CogVideoX-2b

# 本机执行
scp <V100>:/root/cog2b.tgz ~/Desktop/
scp ~/Desktop/cog2b.tgz <H20>:/root/

# H20 上
mkdir -p /root/viewdit/weights && cd /root/viewdit/weights && tar xzf /root/cog2b.tgz
```

**检查点**

```bash
ls /root/viewdit/weights/CogVideoX-2b/model_index.json
du -sh /root/viewdit/weights/CogVideoX-2b     # 应约 13G
ls /root/viewdit/weights/CogVideoX-2b/{transformer,vae,text_encoder,tokenizer,scheduler} | head
```

---

## 第 5 步：冒烟（2 分钟）

```bash
cd /root/viewdit
/root/miniconda3/envs/racfg/bin/python experiments/video/mpcache_controller.py \
  --frames 17 --nprompt 1 --seeds 3000 --Ns 13 \
  --planners reuse,tea_gap --out /root/viewdit/results/video/smoke_h20
```

**必须看到**：`sanity roundtrip 0.0`、`one CFG step ...s`、然后两行 `reuse` / `tea_gap` 且数值不为 NaN。
若报错，把最后 20 行发我。

---

## 第 6 步：主任务（49 帧 × 100 prompt，约 6–8 小时）

```bash
cd /root/viewdit
setsid nohup /root/miniconda3/envs/racfg/bin/python experiments/video/mpcache_controller.py \
  --frames 49 --nprompt 100 --seeds 3000 --Ns 10,13,17 \
  --planners reuse,tea_gap \
  --out /root/viewdit/results/video/cog49_prod \
  > logs/cog49_prod.log 2>&1 < /dev/null &
```

**进度**（每 10 分钟）：

```bash
wc -l /root/viewdit/results/video/cog49_prod/closed.jsonl    # 目标 600 行
tail -3 /root/viewdit/logs/cog49_prod.log
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
```

600 = 100 prompt × 3 预算 × 2 方法。满了就完成。

---

## 第 7 步：结果回来

```bash
cd /root/viewdit
tar czf /root/results_cog49.tgz results/video/cog49_prod/closed.jsonl results/video/smoke_h20/closed.jsonl logs/cog49_prod.log

# 本机
scp <H20>:/root/results_cog49.tgz ~/Desktop/
```

放到桌面我直接算统计。

---

## 排错

| 现象 | 处理 |
|---|---|
| `ModuleNotFoundError: numpy` | 用了系统 python；改用 `/root/miniconda3/envs/racfg/bin/python` |
| `No module named diffusers` | 环境没装好，回到第 2 步 |
| `CUDA out of memory` | 49 帧峰值约 12 GB；先 `--nprompt 1 --frames 49` 冒烟确认 |
| 任务是否活着 | `ps -eo pid,etimes,cmd \| grep mpcache`（**不要用 `pgrep -f`**，会匹配到 ssh 自身） |
| 中途停 | `pkill -f "cog49_prod"`；JSONL 追加写，已完成的不丢 |
| 权重路径错 | 控制器里 `ROOT = Path('/root/viewdit')`，权重须在 `/root/viewdit/weights/CogVideoX-2b` |
