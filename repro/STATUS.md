# Reproduction status — 2026-10-16

## Verified progress

- As of 15:50: CogVideoX-2B Vanilla50/Vanilla25/TeaCache and corrected RA-Slow/RA-Fast each have 16 unique prompt IDs and evaluated quality results. Corrected RA-Slow LPIPS/SSIM/PSNR = 0.081907/0.882860/26.5716; RA-Fast = 0.188414/0.750668/20.8581.
- Cog TaylorSeer legacy n=16 used CPU offload, confirmed by the supervisor log. These results are NOT compliant with the user's all-GPU requirement and are now excluded; gpu_only_v2 reruns are required.
- Wan2.1-1.3B Vanilla50/Vanilla25/TeaCache/corrected RA-Slow each have 16 unique prompt IDs (878/462/367/347 s per video). Corrected RA-Fast was at 14/16 at 15:49. Wan TaylorSeer's GPU-only attempt failed at the first step with CUDA OOM (560 MiB allocation, ~541.5 MiB free; own process ~31.2 GiB), not a network failure.
- All future generation is GPU-only. CPU offload of the denoising model is prohibited, including on OOM. The single documented exception is the opt-in TaylorSeer text-encoder release below.
- G0a has completed 192 perturbation observations over 16 runs. Its existing online-J pricing probe failed the stated gates; this does not stop competitor reproduction and is not a software exception.

## TaylorSeer memory adapter (V100 only, opt-in)

Weight measurement: the Wan text encoder is 21.16 GiB on disk (fp32 umt5-xxl) and 10.58 GiB in fp16, CogVideoX's is 8.87 GiB; both stay resident and are completely idle during denoising. TaylorSeer additionally keeps a per-layer derivative cache, so the two together exceed 31.73 GiB.

`patches/apply_v100_patch.py:patch_release_text_encoder` wraps the pipeline's own `encode_prompt` so that, when `RACFG_RELEASE_TEXT_ENCODER=1`, the encoder is parked on CPU immediately after each prompt is encoded and restored just before the next one. `run_repro.sh` sets that variable for the TaylorSeer row only, so every other row keeps upstream residency and the already-completed rows stay comparable.

This is not CPU offload of the model:
- prompt encoding is byte-identical (same call, same dtype, same device);
- the entire denoising loop — everything that is timed, plus the whole cache implementation — still runs on the GPU;
- the adapter is inert unless the env var is set, and its off-state is regression-tested;
- it slightly penalises TaylorSeer's latency (one device transfer per prompt), never benefits it.

The two `taylorseer`-only probe chunks (`cogvideox:0:16:taylorseer`, `wan:0:16:taylorseer`) run first, so a failure costs minutes rather than delaying the schedule. If they still OOM, the row is reported as "not reproducible on a 31.7 GiB V100" and the paper's published TaylorSeer number is cited with that footnote instead of being silently substituted.

## Critical RA-CFGCache correction

The pinned upstream sample.py forwarded `rel_l1_thresh`, while `cache_init.py` reads `cfgcache_thresh`. Consequently both nominal Slow/Fast runs used the default effective tau=0.4. Their byte-identical Cog outputs confirmed the problem. Previously reported RA quality/speedup rows are invalid as Slow/Fast comparisons.

`patches/apply_v100_patch.py` now explicitly forwards `cfgcache_thresh=float(opts.rel_l1_thresh)` and asserts/logs the effective runtime threshold. Corrected RA outputs go under `results/repro/<model>/<method>/cfg_tau_v2/<chunk>`; old videos are retained but excluded by aggregation. The queue reruns these two rows before enlarging the Cog subset.

## Evaluation and supervisor fixes

- Use the racfg interpreter containing cv2/torch/lpips, not the base interpreter.
- Count unique global IDs; a smoke run does not turn 16 prompts into 17.
- Isolate merged datasets by n; evaluate only matched reference/method IDs.
- Cache metrics only when source/pair/evaluator fingerprints match; publish missing/failed rows as pending, not zero.
- Completion markers require successful generation of all expected IDs; failed jobs remain failures.
- A single supervisor lock prevents duplicate queued launches. The fixed queue took over after Wan TeaCache completed and is running corrected Wan RA-Slow.
- TaylorSeer generation and aggregation now use gpu_only_v2 only; legacy CPU-offload artifacts remain on disk for audit but are inadmissible.
- Deploy shell updates atomically via a new file and rename. Never overwrite an active Bash script inode: the earlier in-place update caused a trailing syntax error in an old wrapper.
- Children of the supervisor are launched with `9>&-`. Otherwise they inherit the flock fd and a killed supervisor leaves the lock held by its orphaned children, blocking the next supervisor.
- An unresolved Taylor GPU memory issue would prevent claiming full six-row completion, even when other methods finish. The opt-in adapter above is the current attempt to resolve it; it is not yet proven on the GPU.

## Protocol limitations

1. V100 uses fp16 instead of the paper's H200 bf16. This is protocol-aligned reproduction on different hardware, not a claim to exactly reproduce paper numbers.
2. Wan uses GPU VAE tiling/slicing to prevent decode OOM; equivalence to untiled decoding has not been established.
3. Shared-GPU chunk-average latency/speedup is provisional and must be re-timed without contention.
4. The server prompt resource was re-counted with splitlines and contains **100 nonempty prompts**. The earlier 99-line observation was not a reliable prompt count (a missing trailing newline can affect wc -l). The runner validates the actual prompt list and preserves seed=global ID.
5. Cog TaylorSeer is an additional row: RA-CFGCache Table 1 has no Cog TaylorSeer reference number.
6. The TaylorSeer row alone runs with the text encoder parked between encoding and denoising (see above). Every other row is upstream-resident. Any row whose generator emits `release_text_encoder=1` in its log is that row.
7. No MPCache SOTA or statistical-gate claim is justified by these baseline runs.

## Checkpoints

GitHub overnight checkpoints are published by the existing server workflow after each chunk. Corrected tables contain exact pair counts and source manifests. Full two-model reproduction remains in progress; no full-completion claim is made.
