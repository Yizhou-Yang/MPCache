# Reproduction status — 2026-10-16

## Verified progress

- CogVideoX-2B: Vanilla50, Vanilla25, TeaCache and TaylorSeer each have 16 unique generated prompt IDs. Quality evaluation has run; audited aggregation is being regenerated.
- Wan2.1-1.3B: Vanilla50/Vanilla25 each have 16 unique generated prompt IDs; TeaCache is generating [0,16). Remaining competitor rows are queued.
- All future generation is GPU-only. CPU offload fallback is prohibited, including on OOM.
- G0a has completed 192 perturbation observations over 16 runs. Its existing online-J pricing probe failed the stated gates; this does not stop competitor reproduction and is not a software exception.

## Critical RA-CFGCache correction

The pinned upstream sample.py forwarded `rel_l1_thresh`, while `cache_init.py` reads `cfgcache_thresh`. Consequently both nominal Slow/Fast runs used the default effective tau=0.4. Their byte-identical Cog outputs confirmed the problem. Previously reported RA quality/speedup rows are invalid as Slow/Fast comparisons.

`patches/apply_v100_patch.py` now explicitly forwards `cfgcache_thresh=float(opts.rel_l1_thresh)` and asserts/logs the effective runtime threshold. Corrected RA outputs go under `results/repro/<model>/<method>/cfg_tau_v2/<chunk>`; old videos are retained but excluded by aggregation. The queue reruns these two rows before enlarging the Cog subset.

## Evaluation and supervisor fixes

- Use the racfg interpreter containing cv2/torch/lpips, not the base interpreter.
- Count unique global IDs; a smoke run does not turn 16 prompts into 17.
- Isolate merged datasets by n; evaluate only matched reference/method IDs.
- Cache metrics only when source/pair/evaluator fingerprints match; publish missing/failed rows as pending, not zero.
- Completion markers require successful generation of all expected IDs; failed jobs remain failures.
- A single supervisor lock prevents duplicate queued launches. A currently running Wan TeaCache process is allowed to finish, then the fixed queue takes over.

## Protocol limitations

1. V100 uses fp16 instead of the paper's H200 bf16. This is protocol-aligned reproduction on different hardware, not a claim to exactly reproduce paper numbers.
2. Wan uses GPU VAE tiling/slicing to prevent decode OOM; equivalence to untiled decoding has not been established.
3. Shared-GPU chunk-average latency/speedup is provisional and must be re-timed without contention.
4. The server prompt resource was re-counted with splitlines and contains **100 nonempty prompts**. The earlier 99-line observation was not a reliable prompt count (a missing trailing newline can affect wc -l). The runner validates the actual prompt list and preserves seed=global ID.
5. Cog TaylorSeer is an additional row: RA-CFGCache Table 1 has no Cog TaylorSeer reference number.
6. No MPCache SOTA or statistical-gate claim is justified by these baseline runs.

## Checkpoints

GitHub overnight checkpoints are published by the existing server workflow after each chunk. Corrected tables contain exact pair counts and source manifests. Full two-model reproduction remains in progress; no full-completion claim is made.
