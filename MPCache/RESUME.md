# Resume point — 2026-11-04 20:35

The paper's numbers are all in `experiments_results/`. The GPU box is reclaimed at 23:00
on 2026-11-04; an autopilot on that box finishes the two remaining runs, evaluates them and
pushes the results to `experiments_results/raw/` every few minutes (`sourcecode/scripts/latest/…`,
see "Autopilot" below). Anything not there did not finish.

## Frozen

| setting | n | arms | LPIPS | SSIM | PSNR |
|---|---:|---|---:|---:|---:|
| **Cog 49f** | **100** | RA-CFGCache | 0.1554 | 0.8973 | 29.17 |
|  |  | **ours** | **0.1424** | **0.9093** | **30.21** |
| Cog 17f | 100 | TaylorSeer | 0.4917 | 0.5259 | 13.91 |
|  |  | TeaCache | 0.2786 | 0.7522 | 21.05 |
|  |  | RA-CFGCache | 0.1587 | 0.8948 | 27.72 |
|  |  | **ours** | **0.1454** | **0.9059** | **28.58** |
| Wan 81f | 80 (65 evaluated) | orig + **ours** 0.1469 / 0.8899 / 28.88 | | | |

Shape ablation (17f, 32 prompts, head 12, budget 20): uniform 0.1656; r=1.4 0.1560; r=2.0 0.1566;
r=3.0 0.1541. Taylor interval sweep: 0.402 / 0.531 / 0.615 at 26 / 19 / 15 calls.

## Autopilot on the box

`/root/autopilot.sh` (launched with the repo token in the environment):
waits for the Wan cfg arm, evaluates ours and cfg against the Wan reference, runs the head-length
ablation and the wall-clock pair, then pushes `results/*.json` to `MPCache/experiments_results/raw/`
on branch `h20-run` every two to five minutes. Nothing needs to be done on the laptop.

## Still to run next time

| # | run | why |
|---|---|---|
| 1 | Wan 81f @100: orig, ours, cfg | the block is at 80 with 65 evaluated; n=100 is the competitor spec |
| 2 | Wan @100: TaylorSeer, TeaCache | those rows are still n=16 |
| 3 | Cog 49f @100: TaylorSeer, TeaCache | the block currently has only CFG and ours at 100 |
| 4 | HunyuanVideo 17f | deferred to the final version |

## Traps that cost time (all hit on 2026-11-04)

1. **torch 2.10.0 / torchvision 0.25.0** are pinned by the RA release. With 2.5.1 every run dies
   with SIGFPE at the first transformer call; it looks like a CUDA fault and is not.
2. **The evaluator must walk hidden directories.** A killed run leaves its videos under
   `.multi_gpu_tmp/`, and `glob("**/*.mp4")` skips dot-directories, which silently reported n=0.
3. **A killed launcher leaves orphan workers** holding GPU memory; check
   `nvidia-smi --query-compute-apps=pid,used_memory` after any `pkill`.
4. **Evaluate every arm as it lands.** Everything lost so far was lost by batching evaluation.
