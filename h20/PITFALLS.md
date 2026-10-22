# H20 踩坑清单（2026-10-22 实测，每条都有症状 / 根因 / 修复）

机器：4×H20-97GB，驱动 535.247.01（CUDA 12.2），TLinux 5.4，网络经 `star-proxy.oa.com:3128` 全通。

---

## P1. cuBLAS 小 M GEMM 触发 SIGFPE（最坑，会吃掉半天）

**症状**

```
Fatal Python error: Floating point exception
Current thread ... File "torch/nn/modules/linear.py", line 125 in forward
```

没有任何 Python traceback，进程直接消失（shell 报 `Floating point exception`，退出码 -8）。
`dmesg` 无 OOM，GPU 显存正常，`nvidia-smi` 一切正常。

**根因**

`torch 2.5.1+cu121` 自带的 `nvidia-cublas-cu12==12.1.3.1` 在 sm90 上对**小 M、大 N** 的 bf16 GEMM
会选到错误 kernel。实测必崩的形状：

| 形状 | 出现位置 |
|---|---|
| `M=2, K=1536, N=9216` | Wan `time_proj` |
| `M=1, K=768, N=3072` | Hunyuan `text_embedder` |
| `M=1, K=3072, N=18432` | Hunyuan adaLN |

大 M（如 32760）同形状完全正常，所以只在小 batch / 单样本时暴露。

**修复（推荐，一次根治）**

```bash
/root/miniconda3/envs/racfg/bin/pip install nvidia-cublas-cu12==12.4.5.8
```

pip 会报 `torch 2.5.1+cu121 requires nvidia-cublas-cu12==12.1.3.1` 的冲突警告，**忽略**。
装完后上面三个形状直接通过，Wan 可以用 bf16（不必再 fp16），Hunyuan 也不必 batch=2 复制。

**备选（不升级 cublas 时）**

- Wan：`torch_dtype=torch.float16` 即绕开（bf16 必崩）；
- Hunyuan：fp16 仍崩在 M=1 的 adaLN，只能把输入复制成 batch=2、输出取 `[:1]`；
  代价是显存 27 GiB → 42.5 GiB、算力翻倍。

---

## P2. 控制器内置 8 条手写 prompt，`--nprompt 100` 必崩

**症状**：`KeyError: 8`，位置 `pe = embs[pi]`。

**根因**：`PROMPTS` 硬编码 8 条（天鹅/红巴士/棕熊…），与"100 VBench prompts"的论文协议无关。

**修复**：三个 runner 统一按下面顺序解析，并打印来源：

```
$MPCACHE_PROMPTS
→ /root/viewdit/resources/prompts/prompt_video.txt   # RA-CFGCache 的 100 条
→ <脚本目录>/prompts_100.txt
→ 内置 8 条（仅兼容旧 run）
```

启动日志必须有 `[prompts] 100 from ...`，没有就是路径没对上。

**计数注意**：`prompt_video.txt` 是 99 个换行 + 无尾换行的末行，
`wc -l` 显示 99，`awk 'NF' | wc -l` 才是 **100**。以 100 为准，否则索引 99 会缺。

---

## P3. `--prompt-offset` 下 sanity 预热读 `embs[0]`

**症状**：`--prompt-offset 99` 时 `KeyError: 0`。

**根因**：预热用 `model(lat, embs[0], ...)`，而分片只编码了自己的区间。

**修复**：改成 `embs[need[0]]`（本片的第一条）。

---

## P4. Wan 的 UniPC 是二阶多步，跳步误差被放大

**症状**：N=13/50 的 `reuse` 相对误差 0.65，且 schedule 集中在尾段
`[0,19,28,34,...,49]`，前 19 步一次性跳过。

**根因**：两层叠加——(a) UniPC 用前两步的 model_output 做二阶修正，跳步时历史是伪造的；
(b) `dist` 用 sigma 相对变化，flow matching 的 sigma 尾段最陡，导致 schedule 前疏后密。

**修复（仅用于消融，主表必须跟论文）**

```bash
--solver euler --sched uniform     # N=13 从 0.65 降到 0.30
```

主表按论文用 `UniPC + flow_shift 3`，Euler 只作为"solver 阶数"消融单独报。

---

## P5. HunyuanVideo 用 embedded guidance，不是 CFG 双分支

**症状**：按 `cat([lat]*2)` + `chunk(2)` 写会得到无意义输出。

**根因**：`guidance_embeds=True` 的模型把 guidance 作为条件嵌入，
`guidance = guidance_scale * 1000`，batch=1，**不做** negative prompt 的 CFG。

**修复**：

```python
o = d(hidden_states=lat, timestep=t.expand(1),
      encoder_hidden_states=pe['e'], encoder_attention_mask=pe['m'],
      pooled_projections=pe['p'], guidance=torch.tensor([6.0*1000], dtype=dt, device='cuda'))
```

---

## P6. Hunyuan 的条件嵌入属性叫 `time_text_embed`

不是 `condition_embedder`（那是 Wan 的命名）。`d.time_text_embed` 是
`HunyuanVideoConditionEmbedding`，内含 `time_proj / timestep_embedder / text_embedder / guidance_embedder`。

---

## P7. Cog 49 帧的 attention 是真慢，不是配置问题

`one CFG step 29.9 s`（49 帧 480×720，70k tokens × 30 层）。
实测 GPU 141.7 TFLOPS、1980 MHz、P0、无降频、功耗 330 W，所以不是限频。
17 帧只要 0.75 s，长序列 attention 是主导项。

结论：Cog 49 帧的 100-prompt 全量在 H20 上是 4 卡约 4–5 h（单卡 16 h+），
不要以为是环境没配好。

---

## P8. RA-CFGCache 的依赖是 torch 2.10，但在 2.5.1 上能跑

仓库 `requirements.txt` 写 `torch==2.10.0`，H20 驱动只支持到 cu121（torch ≤2.5.1）。
实测除 P1 的 cuBLAS 崩溃外没有 API 不兼容：`pip install nvidia-cublas-cu12==12.4.5.8` 后
`RUN/demo_wan.sh` 直接跑通，`models.wan.sample` 正常加载 UniPC + flow_shift 3。

---

## P9. RA 的输出先落在 `.multi_gpu_tmp`，合并前回收机器会丢

`RUN/demo_wan.sh` 先写
`<out>/.multi_gpu_tmp/wan_<mode>_<ts>/gpu{0..3}/{videos,metadata,eval_frames}`，
四个分片都结束后才 `[MERGE]` 到 `<out>/wan_<mode>/`。

如果机器在 merge 前被回收，完成的 mp4 还在隐藏的临时目录里。
推送脚本必须同时扫 `.multi_gpu_tmp`，不能只扫最终目录。

---

## P10. GitHub 只能走 HTTPS + token，SSH 不可用

`~/.ssh/config` 把 github.com 指向 36000 端口，实测 `Network is unreachable`。
`git push` 必须用一次性 URL：

```bash
git push "https://x-access-token:<PAT>@github.com/Yizhou-Yang/MPCache.git" h20-run
```

不要把 PAT 写进 `git remote set-url`（会落到 `.git/config`），用一次性 URL 或
`push_results.sh` 里的变量。

---

## P11. `zfull` 没缓存时，补一条臂要重付 50 步

每个 `(prompt, seed)` 的 all-exact run 是 50 步。原来每次进程重启都重算，
补 `tea_mp` 一条臂（本来只要 13 步）实际要 63 步。

已改为落盘 `OUT/zfull_{pi}_{seed}.pt`（fp32, ~21 MB/样本，Wan 81 帧）。
补臂、补预算、换臂都只付实际步数。

---

## P12. 分片续跑靠 `closed.jsonl` 去重，不要删文件

每行 `(prompt, seed, N, op)` 唯一，启动时读入 `done` 集合并跳过。
中途 `pkill` 不会丢已完成的数据；**但删掉 `closed.jsonl` 就等于从头再来**。

---

## P13. 驱动 535 的 torch 上限

| wheel | 需要的驱动 | 本机 535.247.01 能否用 |
|---|---|---|
| torch 2.5.1+cu121 | ≥525 | ✅ |
| torch 2.5.1+cu124 | ≥550 | ❌ |
| torch 2.10+cu128 | ≥570 | ❌ |

`CUDA driver version is insufficient` 就是这个原因，不要试图升 torch。

---

## P14. 两篇同名 "WorldCache"，引用前先核对

| 论文 | 机制 | 关键术语 |
|---|---|---|
| `arXiv:2603.22286` *Content-Aware Caching for Accelerated Video World Models* | L1 两步速度 + 通道方差显著性 + ATS 阈值调度 + OSI 投影 + LK warp | motion-adaptive threshold、saliency-weighted drift、phase-aware |
| *Accelerating World Models for Free via Heterogeneous Token Caching* | per-token 曲率分类（stable/linear/chaotic）+ 分类型预测器 | κ=‖a‖/‖v‖²、damped Hermite、heterogeneous token |

2603.22286 全文**没有** entropy，也**没有** κ 或 stable/linear/chaotic 分类（逐字核对过 HTML 全文含附录）。
把两篇的机制混着写，审稿人对照原文会直接指出来。
