# Resume point — 2026-11-04 19:05

Everything in the paper is reproducible from this file plus `experiments_results/`.
The GPU box was being reclaimed while the last two experiments were running; they are
listed as **not done** and the commands are given so the next session starts from here.

## Done and frozen

| setting | n | arms | LPIPS(VGG) | SSIM | PSNR |
|---|---:|---|---:|---:|---:|
| **Cog 49f** (main table) | **100** | RA-CFGCache | 0.1554 | 0.8973 | 29.17 |
|  |  | **ours** | **0.1424** | **0.9093** | **30.21** |
| Cog 17f (appendix) | 100 | TaylorSeer | 0.4917 | 0.5259 | 13.91 |
|  |  | TeaCache | 0.2786 | 0.7522 | 21.05 |
|  |  | RA-CFGCache | 0.1587 | 0.8948 | 27.72 |
|  |  | **ours** | **0.1454** | **0.9059** | **28.58** |
| Wan 81f (main table) | 16 | TaylorSeer | 0.3794 | 0.6196 | 17.61 |
|  |  | TeaCache | 0.1991 | 0.8268 | 24.04 |
|  |  | RA-CFGCache | 0.1544 | 0.8703 | 27.79 |
|  |  | **ours** | **0.1343** | **0.8884** | **28.85** |

Shape ablation (Cog 17f, 32 prompts, head 12, budget 20):

| tail | LPIPS | SSIM | PSNR |
|---|---:|---:|---:|
| uniform | 0.1656 | 0.8958 | 27.44 |
| r=1.4 | 0.1560 | 0.9101 | 28.61 |
| r=2.0 | 0.1566 | 0.9066 | 28.32 |
| r=3.0 | 0.1541 | 0.9118 | 29.08 |

Taylor interval sweep (Cog 17f, 32 prompts): i=2/26 calls 0.402; i=3/19 calls 0.531; i=4/15 calls 0.615.

## Not done — start here next time

| # | run | why it matters | cost on 4 dedicated GPUs |
|---|---|---|---|
| 1 | **Wan 81f @100**: orig, ours, cfg | the Wan block of the main table is n=16 while Cog is n=100 | 3 x ~150 min |
| 2 | **Wan @100** + **Cog 49f @100**: TaylorSeer, TeaCache | those two rows are still at n=16 / absent, so the block has mixed n | 4 x ~60-150 min |
| 3 | Head ablation H in {6,10,12,16} | shows H is read from the profile, not tuned | 4 x 12 min |
| 4 | Wall clock, one prompt full vs ours at 49f | the paper currently derives it from step times | 2 x 3 min |
| 5 | HunyuanVideo 17f | deferred to the final version by decision | hours |

### Exact commands

```bash
# 1 + 2, per-arm evaluation built in:
RA_REPO=/path/RA-CFGCache COG_MODEL=... WAN_MODEL=... \
  bash sourcecode/scripts/latest/run_freeze.sh both

# 3, head ablation: same head/budget, vary H
PYTHONPATH=sourcecode python -m mpcache.runner --backend cogvideox --model $COG \
  --prompts prompts.txt --frames 17 --steps 50 --budget 20 --out out/h6 --ledger metrics.jsonl
```

### Watch out for

* **torch 2.10.0 / torchvision 0.25.0** are pinned by the RA release. With 2.5.1 every run
  dies with SIGFPE at the first transformer call, which looks like a CUDA fault and is not.
* The Wan run at 100 prompts takes ~2.5 h for the reference alone. Run it first and evaluate
  every arm as it lands; the box may be reclaimed between arms.
* A killed launcher leaves orphan workers holding GPU memory; check
  `nvidia-smi --query-compute-apps=pid,used_memory` after any `pkill`.
