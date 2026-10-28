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

---

## P15. ours 与 RA 复现的视频不可比（未定位，已改用自洽基准）

**症状**：同 prompt、同 seed（=0）、同 bf16、同为 50 步全精确，ours 的视频与 RA 的
`wan_original_full/video_000000.mp4` 相比 **LPIPS 0.98 / SSIM 0.08 / PSNR 6.4**，
即内容完全无关（不是色彩偏移）。

**已排除**：

| 假设 | 验证 |
|---|---|
| 初始噪声生成方式（bf16 直采 vs fp32 转 bf16） | 同 seed 下 `corr=1.0`，无差异 |
| dtype（我们 fp16 / RA bf16） | 切到 bf16 后仍 0.98 |
| VAE 反归一化 | 已补 `z*latents_std+latents_mean`，仍 0.98 |
| solver / flow shift | 同为 UniPC + shift 3，timesteps 一致 |
| prompt 列表错位 | metadata 里 prompt 文本一致，索引一致 |
| planner 误差 | **N=50 全精确也 0.98**，与 planner 无关 |

**未定位**：RA 的 `sample.py` 加载后会 patch diffusers 的 transformer
（日志 `Expected types for transformer: models.wan.transformer_wan.WanTransformer3DModel`、
`[patch-check] transformer has cache_context: True`）。该 patch 可能改变了数值路径
（例如 probe 前向、attention 后端），使同权重同输入的输出与纯 diffusers 路径分叉。

**后果**：H20 上不能用 RA 的视频当 fidelity 基准，否则 ours 的 LPIPS 恒为 ~1.0。
**临时方案**：用我们自己的 50 步 Full（`--Ns 50`，等价全精确）作为基准，ours 与它比，
口径自洽；竞品行只能引用 RA 论文发表值，不能与本机复现并表。

**根因（2026-10-27 已定位，见 P16/P17）**：不是 RA 的 patch，是 **我们自己的两处 bug**——
文本嵌入未零填充（P16）+ 潜变量用 bf16 累计（P17）。修好后 ours N=50 vs RA Full 降到
LPIPS 0.030 / PSNR 30.7，属于「同内容、数值路径不同」的正常下限。

---

## P16. 文本嵌入没按真实长度截断并零填充 → Wan 出纯色视频（已修，最阴）

**症状**：ours 的 Wan 视频是**一整块深蓝色渐变**（frame mean 41 / std 34），与 RA 同 prompt
的 LPIPS 0.89–0.98、PSNR 9.4。但 `rel`（自比自）一切正常（N=13 vs N=50 约 0.11–0.16），
所以这个 bug 活了很久，直到把帧导出来看了一眼才发现。

**根因**：runner 直接把 `UMT5EncoderModel(...)[0]` 整段（512 token）喂给 transformer，
其中 483 个是 padding 位的激活——不是零。transformer 的 cross-attention 把这几百个
"死 token" 一起 attend，条件信号被稀释，输出退化成常量。
diffusers 的 `WanPipeline._get_t5_prompt_embeds` 做的是「**截到 `seq_len` 个真实 token，
其余补 0**」。

**验证**（同一进程、同权重、同 prompt）：

| 版本 | pos rel | neg rel | std (pos/neg) |
|---|---:|---:|---|
| 未零填充 | 4.2003 | 28.6229 | 0.074 / 0.082 |
| 零填充后 | **0.00000** | **0.00000** | 0.017 / 0.003 |

修好后首帧 mean/std 从 40.8/34.2 变成 114.8/79.7（官方管线 113.1/78.8），画面就是 prompt 的
那个电话亭。

**修复**：`mpcache_wan.py` 的 `_embed()`：

```python
ti = tok(text, padding='max_length', max_length=max_len, truncation=True,
         add_special_tokens=True, return_attention_mask=True, return_tensors='pt')
ids, msk = ti.input_ids.cuda(), ti.attention_mask.cuda()
n = int(msk.gt(0).sum())
h = enc(ids, msk)[0][0, :n].to(torch.bfloat16)
emb = torch.cat([h, h.new_zeros(max_len - n, h.size(1))], dim=0)[None]
```

**注意**：改完必须删掉旧的 `embeddings_*.pt` 缓存，否则继续读坏嵌入、表现和没修一样。
CogVideoX 的官方 `encode_prompt` 不零填充（保留 226 个位置的激活），controller 里那段
**不要**照抄这个改法。

---

## P17. 潜变量用 bf16 累计 → 与官方 50 步全精确差 LPIPS 0.09

**症状**：修完 P16 后，ours `--Ns 50`（全精确）对 RA Full 仍只有 LPIPS 0.089 / PSNR 25.5。
**根因**：runner 的 latents 一直用 bf16 保存并过 scheduler；`WanPipeline` 用 **float32** 保存
latents，只在喂 transformer 前 `.to(bf16)`。50 步累积下来是肉眼可见的漂移。
**修复**：latents 全程 float32，模型输入 cast bf16：

```python
lat = init(seed)                       # float32 randn * init_noise_sigma
o = d(hidden_states=torch.cat([lat.to(torch.bfloat16)] * 2), ...)
lat = sch.step(tok2lat(v).float(), t, lat.float(), return_dict=False)[0]
```

**效果**：ours N=50 vs RA Full：LPIPS 0.089 → **0.030**，PSNR 25.5 → **30.7**
（对纯 diffusers 管线是 0.021 / 31.8）。剩下的 0.03 是跨实现的数值下限，
方法效应 0.09–0.14 仍能分辨，但**报数时要写清楚这一层底噪**。

---

## P18. cuBLAS 12.4.5.8 会被后面任意一条 pip 静默降级 → RA 四卡全 SIGFPE

**症状**：`bash RUN/demo_wan.sh --gpus 0,1,2,3` 启动 90 s 后四个卡一起
`[ERROR] Process on GPU 0 exited with code -8`，无 traceback、无 OOM。
**根因**：P1 的修复装的是 `nvidia-cublas-cu12==12.4.5.8`，但之后 `pip install lpips`
会重算 torch 的依赖树，把 cublas 拉回 `12.1.3.1`。`pip list` 一查就露馅。
**修复**：**cublas 必须是最后一条 pip**；装完立刻两条自检：

```bash
/root/miniconda3/envs/racfg/bin/pip list | grep -i cublas         # 必须是 12.4.5.8
# 三个必崩形状：M=2,K=1536,N=9216 / M=1,K=768,N=3072 / M=1,K=3072,N=18432
```

`h20/bootstrap_h20.sh` 已按这个顺序固定下来（step 4 最后一步），不要再往它后面追加 pip。

---

## P19. Wan 81 帧必须显式 `--frames 81`

`mpcache_wan.py` 的 `--frames` 默认 **33**。命令里漏掉它不会报错，只会安静地生成 33 帧视频，
LPIPS 照样能算出来，但和 RA 的 81 帧参考不可比（帧对齐后仍差一截）。
`TODO_MAIN_TABLE.md` 里最初的 ex1 命令就漏了这个参数。凡是 81 帧的臂，命令检查表里必须有
`--frames 81`。

---

## P20. `scripts/eval.py` 不存在（旧机临时脚本，随机器回收）

RA 公开仓库里只有 `evaluation/eval_video.py` 与 `evaluation/evaluate.py`。主表口径直接用
`repro/metrics.py`：lpips(alex) + skimage SSIM/PSNR，按 `video_%06d.mp4` 配对，
`--max-frames 0` 表示全帧（与 RA 的 eval 一致）。不要再去 `/root/RA-CFGCache/scripts/` 找。

---

## P21. 机器回收后参考视频不会自己回来（4.2 GB，不在 git 里）

`h20-run` 分支只有 metadata（1.2 MB），没有 mp4。每次换机要算 LPIPS 就必须重跑
RA Full（4 卡 100 prompt × 50 步 ≈ 2.2 h），否则只能拿 ours 自比自。
**下次的开机顺序**：`bootstrap_h20.sh` → `queue_wan.sh`（ex1 → tea_gap → full → eval）→
再决定是否补 RA Full。想省这 2.2 h 就把 `results/ra_wan/*/videos` 传对象存储。

## P22 · 2026-10-28 · 我们自己的三处 Cog/Wan 管线 bug

1. **Cog 缺视频导出** → 只能拿到潜变量 rel，无法算 LPIPS。已在 `mpcache_controller.py` 补 `--save-video`（VAE 解码 + 8fps）。
2. **初始化噪声张量布局错**：官方是 `[B,C,T,H,W]`，我们是 `[B,T,C,H,W]`。同 seed 得到完全不同的噪声 → 内容不同的视频。已改为按官方布局采样再 permute。
3. **文本嵌入未按 attention mask 清零 padding**（与 Wan 的 P16 同源）。已改为 `embeds * mask`。

**仍然未解决**：修完这三处后，我们的 Cog 30 步全量 vs 官方 `CogVideoXPipeline` 30 步仍是 **LPIPS 0.674**。
说明还有更深层不一致（模型调用约定 / 调度器 / VAE 缩放）。**在解决前，Cog 的 LPIPS 不能与 RA 的数字同表比较**。

Wan 管线是对的：我们 full vs RA full = **0.035**，可同表比较。

## P23 · 2026-10-28 · RA 的重建机制与我们差在哪（读源码得到）

Wan 上 RA 跳过一步时：

```python
cfg_base_hidden_states = hidden_states.detach()          # 当前 latent 的嵌入（便宜，不跑 blocks）
hidden_states = cfg_base_hidden_states + previous_residual  # 加回上一步全量的残差
```

- 基座**跟着当前输入走**（每次重新嵌入当前 latent），残差是**逐 token 的高维张量**；
- 我们：基座是**冻结的输出外推**，修正是 **64 参数的全局对角模型**。

这解释了我们落后约 0.07 LPIPS。**注意：直接照搬“嵌入+残差”等于采用竞品机制**，需要独立的重建原则。

## P24 · 2026-10-28 · 调度构造的两个坑

- warmup 后**从中间删步**保持预算 → 造出 8 步大间隔，LPIPS 从 0.12 掉到 0.74；
- 用 sigma 相对变化率驱动调度 → `|S|=15`（预算没凑够）且间隔 15 步。

结论：高噪声段必须密（1,1,1,2,3…），不能均匀。目前实测最好的构造是“保留 dense head + 中间删步”，prompt 0 上 0.120。

## P25 · 2026-10-28 · 后台任务存活

`setsid nohup` 不足以保证长任务存活：昨晚 14h 流水线启动即死、评测脚本只完成第一个臂。
**对策**：每步一个短任务 + 30 秒后二次确认 `pgrep`；不要把多阶段串在一个 bash 里。

## P28 · 2026-10-28 · 隐藏层残差（feat-res）实现尝试

**目标**：复现“嵌入(当前输入) + 上一步隐藏层残差”的重建方式（RA 的核心，见 P23）。

**已实现**：
- `d.blocks[0]` 前置 hook 捕获 blocks 的输入（`base`），`d.blocks[-1]` 后置 hook 捕获输出，残差 = out − base；
- `--feat-res 1` + op `fr` 尝试“blocks 替换为 `x + residual` 后只跑 embed/norm/proj”。

**结果**：hook 正常捕获（打印确认 `keys=['base','res']`），但跳过时的重建调用与我们的 latent/token 布局不匹配：
`RuntimeError: shape '[1,16,21,30,2,52,2]' is invalid for input of size 4193280`。

**结论**：feat-res 需要重写整个 forward（patch_embed → rope → blocks → norm_out → proj_out），不能只靠“替换 blocks 的 forward”绕过——因为模型内部的 rope/condition 嵌入与我们的布局转换是耦合的。

**当前状态**：`fr` op 已回退为 `tea_gap` 路径（安全），`--feat-res` 仅保留脚手架。下一次分配若要做，应按“复制 forward 并替换 blocks 循环”的方式重写，预计 3–4 小时。

**教训（再次）**：任何新 op 如果与对照**逐位相同**，先假定没接上；本次两次踩到（hook 未注册、引号被 shell 吞掉导致语法错误）。

## P29 · 2026-10-28 晚 · 隐藏层残差跑通 + 三个结论

**实现**：`_fr_forward()` 完整复制 WanTransformer3DModel.forward（patch_embed → 条件嵌入 → **blocks 替换为 `x + residual`** → norm_out → proj_out → unpatchify）。hooks 在 `d.blocks[0]` 前、`d.blocks[-1]` 后捕获 base 与残差。

**结果（N=19，prompts 0-5，对 RA Full）**：

| 臂 | LPIPS |
|---|---:|
| `fr`（残差重建） | 0.1458 |
| `frg`（残差 + 我们的对角修正） | 0.1594 |
| `tg`（我们的 tea_gap） | **0.1244** |

**三个结论**：
1. 残差重建**跑通了**（不再崩），但在这 6 个 prompt 上**不如我们的 tea_gap**；
2. 把我们的修正**叠加**到残差重建上反而更差（0.159）——两种重建的误差结构不同，修正方向不兼容；
3. 单 prompt 上 `tea_gap` 最好（p0: 0.079 vs fr 0.106），但 100-prompt 均值我们更差（0.169 vs RA 0.094）→ **差距在难 prompt 的尾部**，不在典型样本。

**下一步**：要让残差重建在难 prompt 上体现优势，并找出与 tea_gap 的**分工条件**（例如按预测误差选择用哪种重建），这本身就是一个可发表的“控制器选择重建器”的贡献。

## P30 · 2026-10-28 21:10 · 选择器（残差 vs 我们的修正）在 32 个 prompt 上失败

**设计**：`sel` op —— 锚点年龄 ≥ `--sel-age` 时用残差重建（`_fr_forward`），否则用我们的输出修正。

**结果（N=19，32 个 prompt，配对）**：

| 臂 | mean LPIPS | 备注 |
|---|---:|---|
| `tea_gap`（现有） | **0.1790** | 中位 0.1737 |
| `sel`（age≥2 切残差） | 0.2178 | 中位 0.2217，win 0.09 |

**−21.7%，即选择器更差**。5 个 prompt 上的 −17% 是**易样本上的巧合**，扩到 32 个就反转了。

**结论**：残差重建（`fr`）与我们的修正（`tg`）**不能简单按年龄拼接**。残差的优势没有在难样本上体现出来——相反，它在大多数 prompt 上更差。
`tea_gap` 仍是唯一在 32/100 prompt 上稳定有效的方法。选择器需要**按预测误差**判断（而不是按年龄），但这需要更细的误差信号，留待下次。
