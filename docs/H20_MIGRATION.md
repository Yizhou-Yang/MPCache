# H20 migration plan (96 GiB Hopper, sm_90, bf16)

Written 2026-10-16 16:10, before the H20 exists. Everything below is prepared and tested
locally; the only unknowns are the machine's access details and its measured speed.

## Why this matters for the paper

| | V100 31.7 GiB (today) | H20 96 GiB (tomorrow) |
|---|---|---|
| dtype | fp16 (sm_70 has no bf16 tensor cores) | **bf16 == upstream default == the paper's precision** |
| TaylorSeer | OOMs; needs a text-encoder-release adapter | fits easily, **no adapter at all** |
| Wan VAE decode 81 frames | needs `enable_tiling()+enable_slicing()` | untiled, identical to upstream |
| GPU contention | shared with `g2_wmtrain` | dedicated card |
| latency / speedup | provisional (contended, chunk averages) | **can be re-timed on an idle GPU** |
| deviating patches | dtype + tiling + encoder release + threshold fix | **only the CFGCache threshold fix** |

So the H20 run removes three of the four documented protocol deviations. The remaining
honest caveat is silicon: the paper used H200 (~6.7x the bf16 tensor throughput of H20 at
comparable bandwidth), so **quality metrics are directly comparable, latency/speedup are
not**.

## What to run (in order)

```bash
# 0) one-time environment. Prints a preflight block first; fails fast if the driver,
#    conda, sm_90 or the mirrors are wrong.
bash repro/setup_h20.sh                      # ends with "[setup] DONE"

# 1) smoke: 1 prompt, 6 methods, Cog then Wan. This is the go/no-go and the speed calibration.
RACFG_PROFILE=h20 bash repro/run_repro.sh cogvideox 0 1
RACFG_PROFILE=h20 bash repro/run_repro.sh wan 0 1
#    check: no OOM, no NaN/black frames, per-video seconds, dtype shows bfloat16 in the log

# 2) six-row n=16 on both models (matches the statistical gate's >=16 pairs)
CHUNKS="cogvideox:0:16 wan:0:16" RACFG_PROFILE=h20 PUSH_BRANCH=h20 \
  bash repro/queue_overnight.sh

# 3) full protocol: 100 prompts x 6 methods x 2 backbones, chunked and resumable
RACFG_PROFILE=h20 PUSH_BRANCH=h20 bash repro/queue_overnight.sh
```

`queue_overnight.sh` resumes automatically (it skips only when every expected global video
ID already exists), aggregates with exact common IDs, and pushes a checkpoint after each
chunk. Outputs land in `results/repro_h20/`, so nothing can be mixed with the V100 runs.

## Prepared and verified tonight

- `patches/apply_repro_patch.py` - profile aware. `h20` applies **only** the dtype
  indirection (default still `bfloat16`, i.e. a no-op) and the CFGCache threshold fix;
  it refuses to leave VAE tiling or the encoder release in an h20 tree.
- `patches/apply_v100_patch.py` - now a shim to the above, so every older reference works.
- `run_repro.sh` - `RACFG_PROFILE=v100|h20` selects dtype, adapters and output root.
- `push_overnight.sh` - profile aware; pushes to `$PUSH_BRANCH` (use `h20`).
- `setup_h20.sh` - preflight, pinned clone, env, patch, bf16/sm_90 checks, weights, and a
  hard check that the prompt file really has 100 nonempty prompts.
- Local regression suite for the patch (profiles, idempotence, upgrade path, runtime
  behaviour of the adapter, inert-when-off): all pass, no GPU needed.

## Expected wall clock (single H20, 6 rows)

One prompt costs about 3.1 vanilla-T=50-clip equivalents across the six methods
(1.00 + 0.50 + 0.42 + 0.40 + 0.45 + 0.36). Planning numbers: the H20 budget in
`experiments/tiers.md` is ~4 min/clip for Wan 1.3B 480P; the shared V100 took 14.6 min.
So on one H20: Wan 100 prompts ~= 21-36 h, CogVideoX 100 prompts ~= 16-26 h, plus ~2 h of
LPIPS evaluation. **Measure it in step 1 and extrapolate; do not trust these numbers.**

## Still true after migrating

- TaylorSeer on CogVideoX is an extra row (Table 1 does not report it).
- The upstream CFGCache threshold bug is real and our fix stays; it must be stated in the
  paper's reproduction section.
- No MPCache SOTA or statistical-gate claim follows from reproducing baselines.

## Needed from you before I can start

1. SSH access to the H20 (host/port/user, and whether the same `any4` jump host applies).
2. Whether it has miniconda at `/root/miniconda3` (else set `CONDA_SH`).
3. Outbound access: the Tencent PyPI mirror and `hf-mirror.com`. If it is offline, I will
   copy `weights/` (~40 GB) and the conda env from the V100 over SSH, which is slower but
   avoids the flaky mirror path that already cost us a day.
4. Confirmation of the GPU count (the estimate above assumes one card; 4 cards divide it).
