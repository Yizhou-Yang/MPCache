# Data state, 2026-11-03

## Frozen and usable

| setting | n | arms | LPIPS(VGG) | SSIM | PSNR |
|---|---:|---|---:|---:|---:|
| Cog 49f, 50 steps | 48 | RA-CFGCache | 0.1549 | 0.9040 | 29.10 |
|  |  | **ours** | **0.1436** | **0.9138** | **29.89** |
| Cog 17f, 50 steps | 100 | TaylorSeer | 0.4917 | 0.5259 | 13.91 |
|  |  | TeaCache | 0.2786 | 0.7522 | 21.05 |
|  |  | RA-CFGCache | 0.1587 | 0.8948 | 27.72 |
|  |  | **ours** | **0.1454** | **0.9059** | **28.58** |
| Wan 81f, 50 steps | 16 | TaylorSeer | 0.3794 | 0.6196 | 17.61 |
|  |  | TeaCache | 0.1991 | 0.8268 | 24.04 |
|  |  | RA-CFGCache | 0.1544 | 0.8703 | 27.79 |
|  |  | **ours** | **0.1343** | **0.8884** | **28.85** |

Also frozen: Taylor interval sweep (32 prompts, i=2/3/4 -> 0.402/0.531/0.615), budget curves
for both backbones, the single-skip sensitivity profile (2 prompts, r=0.94, head 71-75%).

## Not yet run, required for the final freeze

The competitor spec is 100 prompts for **both** backbones at 49/81 frames. We have 48 (Cog 49f)
and 16 (Wan 81f). Both need a **dedicated** machine: on a shared 4-GPU box the 49f step went from
2.7 s to 5.8 s and the 81f case from 6.5 s to much worse.

| # | run | why | est. on 4 dedicated GPUs |
|---|---|---|---|
| 1 | Cog 49f @100: orig, cfg, ours | main table at the competitors' spec | 3 x 56 min |
| 2 | Wan 81f @100: orig, cfg, ours | second backbone at the same spec | 3 x 135 min |
| 3 | Wan @100: tea, taylor | fills the four-row table | 2 x 135 min |
| 4 | head ablation H in {6,10,12,16} | shows H is read, not tuned | 4 x 12 min |
| 5 | shape ablation (uniform vs r=1.4/2.0/3.0) | the only evidence for the tail shape, currently one point (-34%) | 4 x 12 min |
| 6 | wall clock per arm | the table reports calls only | 10 min |

`scripts/run_freeze.sh` runs items 1 and 2 in one command and prints the evaluation.

## Traps found while running these (do not repeat)

1. **Killing a launcher leaves orphan workers** holding GPU memory; they silently starve the next
   job. After any `pkill`, check `nvidia-smi --query-compute-apps=pid,used_memory`.
2. A shared GPU box doubles the 49f step time; never estimate a deadline from a shared machine.
3. The harness merges shard outputs at the end of an arm, so an empty output directory does **not**
   mean the arm failed; check the log and the temp shard directory.
4. Report the prompt count with every number: the 16-prompt subset is easier than the 100-prompt
   set (49f: 0.121 vs 0.144 for the same method), so a subset number flatters the method.

## 2026-11-03 17:39 —— 丢失的一批实验（必须重跑）

机器在 **17:39** 前后被回收（早于原定 20:00）。当时正在跑、**未来得及评测**的三组实验随机器丢失：

| 实验 | 当时状态 | 结论 |
|---|---|---|
| 形状消融（u / r=1.4 / r=2.0 / r=3.0） | 第一对跑到 30/64 片 | **无数据**，需重跑 |
| 头长消融（H=6 / H=16） | 排队中，未启动 | **无数据**，需重跑 |
| 墙钟实测（49f full vs ours） | 排队中，未启动 | **无数据**，需重跑 |

**教训（升级 I23）**：不只是"评估完立刻写 JSON"，而是**每跑完一个臂就立刻评测并写回本地**。
昨晚的流程是"全部跑完再统一评测"，机器一回收就全部作废。正确顺序是：

```
for arm in ...; do
  run $arm
  python eval3.py ref $arm >> results.jsonl     # 立刻落盘
done
```

形状主张目前仍只有 `I16` 里那一个数据点（均匀尾 0.114 vs 几何尾 0.075，−34%，12 prompt、50 步、AlexNet 口径；
比值与主干无关，但它只是**一个点**，不是梯度证据）。

## 为什么 19:55 的推送没有发生

计划里写的是"19:55 推送"，但执行方式是 `sleep 到点` 的后台任务——**会话在等待期间不会自己醒来**，
后台任务也不产生一次新的回合。机器 20:00 回收之后才被唤醒，推送自然就晚了。

**正确做法**：用 `schedule_create` 设一个 19:50 的提醒。到点会**唤醒会话并投递提示**，
推送动作真正执行。下次任何"到点必须做某事"都应该用提醒，而不是 sleep。

## 为什么"设个提醒"也救不了

提醒跑在**本地 Mac** 上。下班后 Mac 睡眠，提醒同样不会触发——和 `sleep` 一样依赖同一台机器醒着。

唯一不依赖本地状态的方案：**让产出数据的机器自己推送**。

`scripts/sync_results.sh` 就是干这个的：在 GPU 机器上每 10 分钟把
`metrics.jsonl` / `*.json`（只推小文件，不推视频）提交并推到 `results` 分支。

```bash
GH_TOK=<token> RESULTS_DIR=/root/viewdit/results REPO_DIR=/root/MPCache \
  nohup bash sync_results.sh > /root/sync.log 2>&1 &
```

token 只在启动时的环境变量里，通过临时 askpass 使用，不写进仓库、不改 remote URL。
这样即使本地关机、机器半夜被回收，已经评测的数字也已经在 GitHub 上。

## 2026-11-04 —— 新机器上跑通，两个设定补齐

### 环境教训（会浪费一天）

RA-CFGCache 的 `requirements.txt` 要 **torch==2.10.0 / torchvision 0.25.0**。我先装了 2.5.1，
表现是**第一条 transformer 调用就 SIGFPE / core dump**（`--mode original` 也一样崩），
看起来像 CUDA 或模型问题，实际是版本不匹配。换回 2.10.0 后一次跑通。

判据：**在 RA 的 harness 里，任何原生崩溃先核对 torch 版本**，再怀疑驱动、显存和模型。

### 今天完成

| 设定 | n | 方法 | LPIPS | SSIM | PSNR |
|---|---:|---|---:|---:|---:|
| Cog 49f | 100 | RA-CFGCache | 0.1554 | 0.8973 | 29.17 |
|  |  | **ours** | **0.1424** | **0.9093** | **30.21** |
| 形状消融（17f, 32 prompt, 同头12/同预算20） | | uniform | 0.1656 | 0.8958 | 27.44 |
|  |  | r=1.4 | 0.1560 | 0.9101 | 28.61 |
|  |  | r=2.0 | 0.1566 | 0.9066 | 28.32 |
|  |  | r=3.0 | 0.1541 | 0.9118 | 29.08 |

形状结论：**几何尾部比均匀好 5–7%，比例本身不敏感**（论文已按此改写，旧 34% 已删）。

### 仍需补（n 不一致会暴露）

| 设定 | 缺 | 原因 |
|---|---|---|
| Cog 49f @100 | TaylorSeer、TeaCache | 主表该 block 只有 CFG 与 ours 是 100 |
| Wan 81f @100 | TaylorSeer、TeaCache | 它们还是 n=16，而 CFG/ours 将变 100 |
