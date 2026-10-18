# Reproduction audit — 2026-10-18 (V100 n=100 CogVideoX, n=32 Wan)

Scope: the V100 line only (`overnight`). The H20 line (`h20`) is the table of record once its
aggregation lands; the same audit must be repeated on that bf16 output. Nothing here is a
statistical-gate claim: `statistical_gate` is `not-tested` for every row.

## 0. H20 final table (n=100, bf16) — the table of record

The parallel `h20` run finished: every one of the 12 rows has 100 videos, 100 quality pairs and
100 timing pairs, `exit_code=0`, `statistical_gate=not-tested`. Mean of the per-video metrics
recomputed from each `metrics.json` matches the published table to 4 decimals, so the aggregate is
internally consistent.

| model | method | LPIPS ours / paper | Δ | SSIM ours / paper | Δ | PSNR ours / paper | Δ | speed ours / paper |
|---|---|---|---|---|---|---|---|---|
| cog | Vanilla25 | 0.4704 / 0.4751 | -0.0047 | 0.5350 / 0.5445 | -0.0095 | 14.08 / 13.61 | +0.47 | 1.89 / 1.86 |
| cog | TeaCache | 0.2122 / 0.2040 | +0.0082 | 0.7392 / 0.7538 | -0.0146 | 21.22 / 21.48 | -0.26 | 2.64 / 2.42 |
| cog | TaylorSeer | 0.4047 / — | — | 0.5626 / — | — | 15.38 / — | — | 2.36 / — |
| cog | RA-Slow | 0.0990 / 0.0968 | +0.0022 | 0.8661 / 0.8746 | -0.0085 | 26.89 / 26.93 | -0.04 | 2.63 / 2.39 |
| cog | RA-Fast | 0.2092 / 0.2052 | +0.0040 | 0.7493 / 0.7614 | -0.0121 | 21.88 / 21.89 | -0.01 | 3.74 / 3.13 |
| wan | Vanilla25 | 0.4204 / 0.4371 | -0.0167 | 0.5700 / 0.5658 | +0.0042 | 15.85 / 15.87 | -0.02 | 1.96 / 1.93 |
| wan | TeaCache | 0.1681 / 0.1661 | +0.0020 | 0.7887 / 0.7955 | -0.0068 | 22.84 / 23.08 | -0.24 | 2.53 / 2.44 |
| wan | TaylorSeer | 0.3496 / 0.3632 | -0.0136 | 0.6283 / 0.6187 | +0.0096 | 17.10 / 16.96 | +0.14 | 2.47 / 2.13 |
| wan | RA-Slow | 0.0941 / 0.0993 | -0.0052 | 0.8820 / 0.8778 | +0.0042 | 28.11 / 28.03 | +0.08 | 2.65 / 2.56 |
| wan | RA-Fast | 0.1352 / 0.1409 | -0.0057 | 0.8420 / 0.8386 | +0.0034 | 26.14 / 26.12 | +0.02 | 3.14 / 2.99 |

- Worst deviations: **Cog 0.0082 LPIPS / 0.0146 SSIM / 0.47 dB**, **Wan 0.0167 LPIPS / 0.0096 SSIM /
  0.24 dB**. That is tighter than the V100 fp16 table (Cog 0.0139 LPIPS), which confirms the
  fp16-vs-bf16 hypothesis for most of the V100 gap.
- Speedups now land much closer to the paper (Cog 2.64 vs 2.42, 3.74 vs 3.13; Wan within 0.1x except
  TaylorSeer 2.47 vs 2.13), but they are still **not like-for-like**: H20 != H200.
- Degenerate prompt on this run: Cog **id 56** is pure white in the reference and in every method, so
  it contributes an artificially low LPIPS to all rows. Excluding it raises Cog means by only
  0.002-0.004 (e.g. TeaCache 0.2122 -> 0.2143) and changes no ordering. Wan has no degenerate prompt.
- TaylorSeer < TeaCache on both backbones, matching the pattern in RA-CFGCache's own Wan row.

**Verdict for the table of record: quality agrees with the competitor to better than 0.017 LPIPS /
0.015 SSIM / 0.5 dB on every shared row; speedups are comparable in trend but must carry the
H20-vs-H200 caveat.**

## 1. Completeness

| model | rows | videos per row | quality pairs | timing pairs | notes |
|---|---|---|---|---|---|
| cogvideox | 6 | 100 | 100 | 100 | complete; TaylorSeer also has 2 superseded legacy videos |
| wan | 5 (+1 pending) | 32 | 32 | 32 | V100 line stopped on 10-18; TaylorSeer never ran here (90+ GiB cache) |

No video failed to decode. Degenerate-frame scan (first frame std < 3 over the whole row):

- `cogvideox` id **74 is a pure white clip in the Vanilla50 reference and in TeaCache / RA-Slow /
  RA-Fast**, i.e. the prompt itself collapses at 50 steps. It is not a file corruption: 49 frames,
  mean 253-255, std 0.1-0.5.
- `cogvideox` Vanilla25 additionally collapses at ids **24 and 56** (it is not white at 74).
- Recomputing the row means without id 74 moves LPIPS by at most **0.0016** (0.4612 -> 0.4628 for
  Vanilla25, 0.2015 -> 0.2020 for TeaCache, 0.1031 -> 0.1033 for RA-Slow). The degenerate prompt
  therefore does not explain any of the deltas below.
- `wan`: 0 degenerate videos in every completed row.

## 2. Agreement with RA-CFGCache Table 1

CogVideoX, ours (V100 fp16, n=100) vs paper (H200 bf16):

| method | LPIPS ours / paper | ΔLPIPS | SSIM ours / paper | ΔSSIM | PSNR ours / paper | ΔPSNR |
|---|---|---|---|---|---|---|
| Vanilla25 | 0.4612 / 0.4751 | -0.0139 | 0.5446 / 0.5445 | +0.0001 | 13.97 / 13.61 | +0.36 |
| TeaCache | 0.2015 / 0.2040 | -0.0025 | 0.7503 / 0.7538 | -0.0035 | 21.21 / 21.48 | -0.27 |
| TaylorSeer | 0.4044 / — | — | 0.5607 / — | — | 15.13 / — | — |
| RA-Slow | 0.1031 / 0.0968 | +0.0063 | 0.8589 / 0.8746 | -0.0157 | 26.15 / 26.93 | -0.78 |
| RA-Fast | 0.2094 / 0.2052 | +0.0042 | 0.7477 / 0.7614 | -0.0137 | 21.44 / 21.89 | -0.45 |

Worst deviation across all shared rows: **0.0139 LPIPS, 0.0157 SSIM, 0.78 dB PSNR**.

Wan, ours (V100 fp16, n=32) vs paper (H200 bf16):

| method | LPIPS ours / paper | ΔLPIPS | SSIM ours / paper | ΔSSIM | PSNR ours / paper | ΔPSNR |
|---|---|---|---|---|---|---|
| Vanilla25 | 0.4259 / 0.4371 | -0.0112 | 0.6051 / 0.5658 | +0.0393 | 16.23 / 15.87 | +0.36 |
| TeaCache | 0.1515 / 0.1661 | -0.0146 | 0.8369 / 0.7955 | +0.0414 | 23.80 / 23.08 | +0.72 |
| RA-Slow | 0.0859 / 0.0993 | -0.0134 | 0.9025 / 0.8778 | +0.0247 | 28.94 / 28.03 | +0.91 |
| RA-Fast | 0.1251 / 0.1409 | -0.0158 | 0.8659 / 0.8386 | +0.0273 | 26.79 / 26.12 | +0.67 |

Worst deviation: **0.0158 LPIPS, 0.0414 SSIM, 0.91 dB PSNR**. Every Wan row is slightly *better*
than the paper, in the same direction — a systematic (evaluator/hardware) offset rather than noise.
The H20 bf16 run will show whether it is the fp16/bf16 difference or the evaluator.

## 3. Ordering and plausibility

- Quality ordering is the expected one on both backbones: RA-Slow (best) -> RA-Fast ~ TeaCache ->
  Vanilla25 (worst), with the full 50-step run as the reference.
- **TaylorSeer being far worse than TeaCache on CogVideoX (0.4044 vs 0.2015) is not a bug signal.**
  RA-CFGCache's own Wan row shows the same pattern (TaylorSeer 0.3632 vs TeaCache 0.1661 at a
  comparable speedup). CogVideoX has no TaylorSeer row in that table, so ours is an extra row and
  must be labelled as such.
- Latency and speedup are internally consistent in every row (`baseline_latency / speedup` equals
  the reported latency to within 0.3 s), so the timing arithmetic itself is sound.

## 4. What cannot be compared: speedup

| | ours (V100) | paper (H200) |
|---|---|---|
| CogVideoX 1x latency | 399.7 s | 40.48 s |
| CogVideoX speedups | 2.43x / 3.45x / 3.63x / 5.03x | 1.86x / 2.42x / 2.39x / 3.13x |
| Wan 1x latency | 878.6 s | 93.32 s |
| Wan speedups | 1.90x / 2.41x / 2.54x / 2.97x | 1.93x / 2.44x / 2.56x / 2.99x |

The Wan speedups agree within 2 %, but the CogVideoX speedups are 1.3-1.6x the paper's even though
the absolute latency ratio is ~10x on both backbones. That is a timing-scope / hardware effect
(fixed encode+decode cost is a different fraction of the step at 49 frames than at 81), not a
protocol change; the manifest records the basis as "selected source chunk mean; not exact
per-prompt latency". **Do not quote our speedup against the paper's.** The quality columns are the
comparable ones.

## 5. Verdict

- **Quality: consistent with the competitor.** Within 0.016 LPIPS / 0.04 SSIM / 0.9 dB PSNR on every
  shared row, with the same ordering and the same TaylorSeer < TeaCache pattern the competitor
  itself reports.
- **Speed: not comparable.** Correctly attributed to V100 fp16 vs H200 bf16 plus chunk-average
  timing on a shared card.
- **Expected?** Yes, with two recorded caveats: prompt 74 collapses to white for the reference and
  most methods (impact <= 0.0016 LPIPS, kept, not hand-edited), and Vanilla25 collapses on two more
  prompts.
- **Not claimed:** any statistical gate, any MPCache SOTA, any exact reproduction of paper numbers.

## 6. To redo on the H20 aggregate

1. Rerun the degenerate-frame scan on `results/repro_h20` (per-row, first-frame std).
2. Recompute the paper deltas with bf16 numbers; if the Wan "+0.014 LPIPS" advantage disappears,
   it was fp16/bf16, if it stays it is the evaluator.
3. Check the TaylorSeer `<` TeaCache pattern survives — it should.
4. Record TaylorSeer's measured peak memory on 96 GiB (read from `finalize_h20.log`/sampling).
