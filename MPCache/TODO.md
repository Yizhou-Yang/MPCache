# Todo — state at 2026-11-12 17:45

## Done (all pushed)

| # | item | result |
|---|---|---|
| 1 | Cog 49f @100, CFGCache vs ours | 0.1554 / 0.8973 / 29.17 vs **0.1300 / 0.9247 / 31.63** |
| 2 | Wan 81f @100, CFGCache vs ours | 0.1685 / 0.8686 / 27.91 vs **0.1527 / 0.8853 / 29.27** |
| 3 | Cog 17f @100, four baselines | ours **0.1454**, CFG 0.1587, Tea 0.2786, Taylor 0.4917 |
| 4 | Head-length law | H=6 0.288, H=12 0.157, H=16 0.133 |
| 5 | Tail-shape ablation | uniform 0.1656 vs growing 0.1541; ratio 1.4-3.0 within noise |
| 6 | Feedback control | unidentifiable with a scalar error scale (three implementations) |
| 7 | Additivity of single-skip gains | fails: R^2 = 0.59, 29% above prediction |
| 8 | Damping law (17f, 32 prompts) | alpha 0 / 0.25 / **0.5** / 0.75 / 1.0 = 0.1556 / 0.1239 / **0.1084** / 0.1131 / 0.1250; gap-scaled 0.4419 |
| 9 | Reuse at the main setting | plain reuse 0.1549 vs one-step extrapolation **0.1174** at 49f/100 |
| 10 | Novelty audit | SenCache, DPCache, IdealCache, SoftCap, ReCache, CEM-MPC all occupy the control framing |
| 11 | Paper | 10 pages, 0 error, discussion carries the two axes and the damping law |
| 12 | Repository | CuratorMem layout, SDK, experiment scripts, results ledger, all pushed |

## Next session, in order

1. **Finish alpha = 0.5 at 49f / 100 prompts** (28 of 100 were generated; about 25 min on four GPUs,
   shard with `reuse_cmp.py --offset`). This is the only number missing for the main claim.
2. **Run the damping sweep at 49 frames** (alpha 0.25 / 0.5 / 0.75), since the law is currently
   measured at 17 frames only.
3. **Rewrite the paper around the three axes**: where to place the exact calls, what a skip
   returns, and how far the correction may reach. The title should follow the finding, not the
   old framing.
4. HunyuanVideo 17 frames: still deferred by decision.

## Commands

```bash
# one arm of the reuse comparison, sharded
CUDA_VISIBLE_DEVICES=$sid python reuse_cmp.py --model /weights/CogVideoX-2b \
  --prompts prompts_video.txt --frames 49 --mode extrap --alpha 0.5 \
  --offset $((sid*25)) --limit 25 --out out/half/s$sid

# evaluation (matches by basename across shards)
python - <<'PY'
# see EMPIRICAL_FINDINGS.md for the exact block
PY
```
