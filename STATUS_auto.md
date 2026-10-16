# MPCache reproduction checkpoint

Generated: 2026-10-16T11:19:11.453356

GPU-only generation; V100 fp16 differs from paper H200 bf16.
Legacy RA Slow/Fast outputs used tau=0.4 and are excluded; cfg_tau_v2 reruns required.
Video counts are unique global IDs, not overlapping chunk totals.
Timing uses chunk averages on a shared GPU and remains provisional.

## cogvideox

- vanilla50: 16 valid unique videos
- vanilla25: 16 valid unique videos
- teacache: 16 valid unique videos
- taylorseer: 16 valid unique videos
- racfg_slow: 0 valid unique videos
- racfg_fast: 0 valid unique videos
## wan

- vanilla50: 16 valid unique videos
- vanilla25: 16 valid unique videos
- teacache: 8 valid unique videos
- taylorseer: 0 valid unique videos
- racfg_slow: 0 valid unique videos
- racfg_fast: 0 valid unique videos
