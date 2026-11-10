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

## 5. Forecasting the skipped output hurts

Recorded earlier and unchanged: replacing the cached value with a Taylor-style extrapolation
of the last two exact outputs is worse than plain reuse at every budget. K=2 anchor selection
is a variant of the same idea and is measured in `k2.log` (16 prompts) for completeness.

## What this adds up to

A paper can be built on 1 + 2 with 3 + 4 + 5 as the analysis: **the schedule is a head-length
decision; the tail is nearly flat; feedback and forecasting do not buy anything at a matched
budget, and we show why the additive picture that motivates them is wrong.** That is an
empirical paper, not a method paper, and it must be titled and framed as such.

A paper cannot be built on "we are the first to model caching as control": SenCache,
DPCache, IdealCache, SoftCap, ReCache and CEM-MPC all occupy parts of that framing
(see NOVELTY_AUDIT.md).
