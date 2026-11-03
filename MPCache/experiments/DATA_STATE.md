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
