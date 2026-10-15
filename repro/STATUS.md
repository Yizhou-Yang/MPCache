# Reproduction status

| date | step | state |
|---|---|---|
| 2026-10-15 | harness, V100 dtype patch, run/aggregate scripts, pinned submodules | done |
| 2026-10-15 | server env `racfg`: torch 2.10.0+cu128 (has sm_70), diffusers 0.37.0, transformers 5.2.0, lpips | done |
| 2026-10-15 | CogVideoX-2B smoke test, prompt 0 (vanilla50 / TeaCache / TaylorSeer / RA-Slow) | done, all run |
| 2026-10-15 | Wan2.1-T2V-1.3B-Diffusers weights (hf-mirror, resumable `download_wan.sh`) | downloading |
| – | Wan smoke (prompt 0: vanilla50 / TeaCache) | queued after G0a + download |
| – | CogVideoX-2B [0,32) → Wan [0,32) → CogVideoX [32,100) → Wan [32,100), all 6 rows | queued (`queue_full.sh`) |

## Smoke test, CogVideoX-2B, n = 1 (prompt 0, seed 0) — function check only, not a result

| Method | LPIPS↓ ours / paper(n=100) | SSIM↑ | PSNR↑ | Speedup | Latency (s) V100 / H200 |
|---|---|---|---|---|---|
| Vanilla (T=50) | 0 / 0 | 1 / 1 | ∞ / ∞ | 1.00× / 1.00× | 690.0 / 40.48 |
| TeaCache τ=0.2 | 0.2921 / 0.2040 | 0.4712 / 0.7538 | 16.99 / 21.48 | 2.73× / 2.42× | 252.4 / 16.72 |
| TaylorSeer N=3,O=1 | 0.5934 / – | 0.2304 / – | 11.82 / – | 2.23× / – | 309.3* / – |
| RA-CFGCache-Slow τ=0.6 | 0.0427 / 0.0968 | 0.9220 / 0.8746 | 28.26 / 26.93 | 2.46× / 2.39× | 280.0 / 16.96 |

\* TaylorSeer smoke run used CPU offload (others did not). One prompt says nothing about means;
speedups are already in the paper's range, ranking RA-Slow > TeaCache matches.

## Deviations from the paper setup (to report in our paper)

1. **dtype**: fp16 on V100 instead of bf16 on H200 (`RACFG_DTYPE=float16`; CogVideoX-2B is an fp16 checkpoint).
2. **CPU offload for all rows**: TaylorSeer's per-layer derivative cache OOMs on 32 GB with the text
   encoder resident (H200 has 141 GB). `enable_model_cpu_offload` only moves the text encoder / VAE
   between stages, the denoising loop is unchanged; applied to every method so speedups stay comparable.
3. **Latency**: ~17× slower than H200 in absolute terms (13.3 s / step for CogVideoX-2B, two sequential
   CFG branches). Only speedup ratios are comparable to the paper.
4. **Shared GPU**: another process (`g2_wmtrain.py`, ~6 GB) runs on the same V100; absolute latencies
   are noisy, ratios less so. Final latency numbers should be re-timed on an idle GPU.

## Cost on V100 (measured)

CogVideoX-2B: ≈35 GPU-min per prompt for all 6 rows → 32 prompts ≈ 19 h, 100 prompts ≈ 58 h.
Wan2.1-1.3B: to be measured in the smoke test (expected ≈2–2.5× CogVideoX).
