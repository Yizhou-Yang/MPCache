# What the experiments actually established — 2026-11-10

Everything here is measured, at a matched number of transformer calls, inside one harness
(the RA-CFGCache release for the main tables; plain diffusers for the control probes).
No claim in this file depends on a calibration set beyond one single-skip ablation.

## 1. At the competitor spec, the head-dense schedule wins on both backbones

| setting | n | baseline | ours | ratio |
|---|---:|---|---:|---:|
| CogVideoX-2B, 49f, 50 steps | 100 | RA-CFGCache 0.1554 / 0.8973 / 29.17 | **0.1300 / 0.9247 / 31.63** | 1.19x |
| Wan2.1-1.3B, 81f, 50 steps | 100 | RA-CFGCache 0.1685 / 0.8686 / 27.91 | **0.1527 / 0.8853 / 29.27** | 1.10x |

Three metrics, same calls, one pipeline. This is the part of the paper that survives any
priority dispute, because it is a measurement, not a claim of firstness.

## 2. The head length is the parameter that matters, and it is not small

H=6 → 0.288, H=12 → 0.157, H=16 → 0.133 at a 20-call budget (17f, 32 prompts). At 49f/100,
H=12 → 0.142 and H=16 → 0.130. A head of six is catastrophic; a uniform grid is the H=0
limit of the same family.

## 3. Feedback does not pay, and the reason is structural

In a fixed-budget allocation problem a *global* error scale multiplies every cost, so the
optimal allocation is invariant to it: re-planning on a better estimate of the scale cannot
change the schedule. Three implementations confirmed this (budgeted DP, per-bucket rate,
absolute-threshold rule), the first two by construction and the third by degenerating.

**Consequence.** The adaptive line (SenCache, ReCache) can only win by carrying a
*step-dependent* model — which is exactly what it carries. There is no free lunch to be had
from "more feedback" alone.

## 4. Single-skip gains do not add up

Predicted terminal loss `sum_{i not in S} g_i` versus realised LPIPS, three schedules,
16 prompts each:

| schedule | calls | predicted | realised mean | realised max |
|---|---:|---:|---:|---:|
| b16 | 16 | 1.588 | 0.253 | 0.414 |
| b22 | 22 | 1.199 | 0.158 | 0.259 |
| b28 | 28 | 0.933 | 0.093 | 0.149 |

One free scale, fitted: alpha = 0.140, **R^2 = 0.59**; ratio realised/predicted has median
0.91, p90 1.28, max 1.86, and **29% of samples exceed the prediction**. The additive model
therefore cannot serve as an error certificate, and a path-dependent cost (DPCache) is
required to fix it.

## 5. The replacement is NOT inert — an earlier conclusion of ours was an artefact

Holding the schedule and the calls fixed (head 16 + six tail steps, 22 calls) and changing only
what a skipped step returns, on CogVideoX-2B at 17 frames, 16 prompts:

| what a skip returns | LPIPS | SSIM | PSNR |
|---|---:|---:|---:|
| the last exact output (plain reuse) | 0.1471 | 0.9093 | 27.47 |
| one-step velocity extrapolation `2 v_s - v_{s-1}` | **0.1161** | **0.9359** | **30.42** |

A fifth of the error, at identical cost, reproduced deterministically. This **contradicts the
earlier claim that the replacement does not matter**; that claim came from a predictor applied
over the whole gap instead of one step, and that version does hurt.

The open question it raises is the one worth a paper: *why* does output-space extrapolation
work here when the feature-space Taylor predictor of the same family fails badly on this
backbone (0.49 LPIPS at the same budget)? "What to extrapolate" looks like a real axis, and it
is not the one the caching literature has been tuning.

## What this adds up to

A paper can be built on 1 + 2 with 3 + 4 + 5 as the analysis: **the schedule is a head-length
decision; the tail is nearly flat; feedback and forecasting do not buy anything at a matched
budget, and we show why the additive picture that motivates them is wrong.** That is an
empirical paper, not a method paper, and it must be titled and framed as such.

A paper cannot be built on "we are the first to model caching as control": SenCache,
DPCache, IdealCache, SoftCap, ReCache and CEM-MPC all occupy parts of that framing
(see NOVELTY_AUDIT.md).

## Machine loss, 2026-11-10 21:00

The box was reclaimed with the 49-frame confirmation of section 5 unfinished (it had produced
one of eight reference runs). Everything above was pushed before the loss. The next session
starts from here:

1. **Confirm section 5 at the main setting**: 49 frames, 100 prompts, reference + both reuse
   modes at the shipped schedule (~40 min on four GPUs). Until then the 21% is a 17-frame,
   16-prompt, deterministic result.
2. **The ablation that follows from it**: what to extrapolate (output vs feature space) and how
   far (one step vs scaled by the gap). TaylorSeer's feature-space predictor reaches 0.49 at a
   comparable budget while output-space one-step extrapolation reaches 0.116, so the axis is
   real and unexplored.
3. Only then decide the framing: schedule + replacement as two axes is a measurement paper;
   "MPC as the core" is not supportable (NOVELTY_AUDIT.md).
