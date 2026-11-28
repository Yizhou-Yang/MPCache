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

## 6. How far to extrapolate — the damping law (2026-11-12)

Same schedule (head 16 + six tail steps, 22 calls), same prompts, 17 frames, 32 prompts.
`alpha` is how much of the one-step output difference is added at every skip:
`v_j = v_s + alpha * (v_s - v_{s-1})`, with `s` the anchor. It does **not** depend on the gap.

| alpha | LPIPS | SSIM | PSNR |
|---|---:|---:|---:|
| 0.0 (plain reuse) | 0.1556 | 0.8985 | 27.23 |
| 0.25 | 0.1239 | 0.9252 | 28.97 |
| **0.5** | **0.1084** | 0.9394 | 30.41 |
| 0.75 | 0.1131 | **0.9399** | 30.80 |
| 1.0 | 0.1250 | 0.9303 | 29.98 |
| scaled by the gap | 0.4419 | 0.6304 | 15.63 |

Three findings, in order of how surprising they are:

1. **Scaling the extrapolation by the gap destroys the result** (0.442). That is the textbook
   first-order choice — and the one feature-space Taylor predictors make. It also matches
   TaylorSeer's 0.49 on this backbone at a comparable budget, so the mechanism is not
   "Taylor is worse", it is "the step is not locally linear enough to scale".
2. **The optimum is damped, at about half a step** (0.108, a 30% improvement over reuse), and it
   is flat between 0.5 and 0.75. The best alpha is a property of the sampler, not of the prompt.
3. **Confirmed at the main setting**: at 49 frames and 100 prompts, plain reuse gives 0.1549 and
   a one-step extrapolation 0.1174 (24% better, +0.05 SSIM, +4 dB PSNR) at identical cost.

The paper's story therefore becomes: *placement and prediction are two axes, and the prediction
must be damped rather than gap-scaled.* The second half of that sentence is a measured law that
explains a published failure mode.

## State at hand-off, 2026-11-12 17:40

Finished and pushed: everything above, plus the main tables at the competitor spec, the head
law, the three negative results (feedback, additivity, forecasting-as-practised), and the
novelty audit.

Running when the box was released: `alpha = 0.5` at 49 frames / 100 prompts (24 of 100 done).
Next session starts by finishing it (about 25 minutes on four GPUs) and then running the same
sweep at 49 frames, which is the only number still missing for the paper's main claim.

## 2026-11-20 — paper rewritten around the two decisions

The abstract, the introduction and the method now lead with the two decisions a cache makes
(where to place the exact calls, what a skipped step returns) and with the damping law, instead
of leading with the placement-only story. The new method section states the reuse rule and why
scaling the correction by the gap is the choice that fails.

The 49-frame damping sweep was interrupted by the machine being reclaimed with the reference at
96 of 100 prompts and no arm finished, so the law is still measured at 17 frames only. The
49-frame evidence for the reuse axis remains the alpha = 1 comparison (0.1549 to 0.1174 at 100
prompts), which is already in `experiments_results/main_table.json`.

## 2026-11-26 — the damping law is a CogVideoX result, and the paper must say so

Attempting the same α sweep on Wan2.1 at 81 frames, in a plain diffusers harness, produced a
result that is worth recording even though it is not the one we wanted:

| policy (Wan, 21 calls of 50, same harness) | LPIPS | SSIM | PSNR |
|---|---:|---:|---:|
| plain reuse | 0.4607 | 0.4795 | 16.14 |
| damped 0.5 | 0.4081 | 0.5570 | 17.46 |
| one-step, no damping | 0.3982 | 0.5690 | 17.49 |

The ordering is **monotone in α**, not the inverted U measured on CogVideoX-2B, and the absolute
error is three times worse than what the RA harness reaches on the same model and budget (0.153
with our schedule injected into its cache). Two facts explain the gap and both matter for the
paper:

1. **Reuse granularity.** The RA harness caches inside the transformer (features), while our
   correction replaces the *output* of the whole network. On a flow-matching multistep solver,
   holding the output for several steps is far more damaging than holding features.
2. **Solver coupling.** Wan uses UniPC over a flow field; a constant output correction is not the
   right object there.

**Consequence for the claims.** The damping law is a CogVideoX result. The Wan number in the main
table is the *schedule* injected into the RA pipeline, not the damped correction, and the text must
not describe it as "our combination". This is now the paper's main scope limitation, and it is
stated in the appendix.

## 2026-11-26 — first real qualitative grid (asset only, not yet in the paper)

Generated inside the RA pipeline at 49 frames on the first three VBench prompts, one frame from the
middle of each clip: Original, TeaCache, RA-CFGCache, and our schedule injected into that pipeline.
Stored as `paper_figures/qual_A_draft.png`.

**Assessment: not decisive at this budget.** The four columns are visually close; nothing can be
circled honestly. That is consistent with the metrics (0.130 against 0.155 is a metric gap, not a
visible failure), and it is the reason a qualitative figure is the wrong *first* figure for this
paper. If one is wanted, it needs either a much tighter budget (ten to twelve calls), where the
baselines are known to break, or prompts with motion and faces, where errors localise.

## 2026-11-28 — the qualitative figure, second attempt, and why there will not be one

Two attempts, both inside the RA pipeline, both on real outputs:

| attempt | budget | result |
|---|---|---|
| three static prompts, one frame each | ours 22 calls, baselines at their defaults | four columns visually indistinguishable |
| six motion prompts (bungee, horse, dog, giraffe, sheep, polar bear), one frame each | **ours 12 calls**, baselines at their defaults | ours is visibly the worst: the jumper smears, the horse blurs, the dog blobs |

Both are honest outcomes, and together they settle the question. At equal-ish cost the four
methods look the same, so there is nothing to circle. When we push our budget down to twelve
calls, the schedule itself is over-extended (the head-length law needs about twenty-two calls at
fifty steps) and ours becomes the worst column. A qualitative figure can only be made favourable
by comparing at unequal cost, which is not a figure we are willing to publish.

**Decision: no qualitative figure.** The paper argues from measurements, and the appendix says so
in one sentence. The asset from the first attempt is kept at `paper_figures/qual_A_draft.png` for
the record; the twelve-call grid is at `paper_figures/qual_12calls_motion.png`.

**Side effect worth recording**: the fresh RA checkout does not read `MPC_SCHED` by itself. The
paper's runs used a three-line injection into `models/cogvideox/cache_functions/cal_type.py`,
recorded in `ra_injection.md`, and it has to be re-applied after every fresh clone. Its absence is
detectable: an injected twelve-call run of six prompts takes five minutes, the unpatched one took
thirteen.
