# From scheduling to certified control — 2026-11-10

The shallow use of control is "choose which steps to evaluate". Every recent paper does
that (SenCache, DPCache, IdealCache, ReCache). The deeper use is to make the controller
produce a **guarantee**, which none of them do: a per-sample bound on the terminal error
of the schedule it chooses, and a policy that spends the fewest calls that meet a target.

## Formulation

Let `S` be the set of exact steps and `g_i` the terminal loss caused by skipping step `i`
alone, measured once by single-skip ablation. Take the **additive model**

    L(S) ≈ sum_{i not in S} g_i ,                                            (1)

as a *predicted* terminal loss. The control problem becomes a chance-constrained one:

    minimise  |S|
    subject to P( L(S) <= eps ) >= 1 - delta ,   T-1 in S ,   |S| <= budget.  (2)

Two things follow that the schedule papers cannot state:

1. **A certificate.** For a chosen `S`, report `sum_{i not in S} g_i` as the predicted
   error with a measured validity ratio `realised / predicted`; if that ratio stays below
   one across prompts and schedules, (1) is an upper bound and the controller is certified.
2. **A stop rule.** With (1), the controller can stop spending calls as soon as the
   predicted bound falls under `eps`; which is a rule, not a tuned schedule.

## The test that decides it

We already generate the ingredients: per-prompt realised LPIPS for several schedules
(b16 / b22 / b28 at 17 frames) and the full reference. Then:

* compute `sum_{i not in S} g_i` for each schedule and each prompt (profile measured at the
  same step count);
* report the **ratio** `realised / predicted` per prompt and schedule: max, median, and the
  fraction above one (a valid bound needs max <= 1);
* report the **rank correlation** between predicted and realised across schedules and
  prompts (needed for control: the bound must also order schedules correctly).

**Decision rule.** Bound valid (max ratio <= 1) and correlation >= 0.7 → certified control
is a real contribution and the paper can be built on (2). Bound invalid → report it as a
negative result about the additive model, which is itself useful, since SenCache and
DPCache both rely on additive-ish costs without testing additivity.

## The open risk, stated up front

Single-skip gains almost certainly **do not add**: our own ablations show that skipping
twenty consecutive steps is far worse than twenty independent single skips would predict
(the compounding we hit on 2026-11-02). If additivity fails, the fix is a **path-dependent**
bound, e.g. `sum_i g_i * (1 + lambda * run_length_i)` with `lambda` fitted from the same
data, and then the question becomes whether a *simple* interaction term is enough to
certify. That is a well-posed, cheap experiment.

## Why this is not SenCache / DPCache / IdealCache / ReCache

| they | this |
|---|---|
| a point estimate of the error, tuned to minimise the mean | a bound, validated per sample |
| a schedule fixed at calibration | a policy over `eps` and `delta`, i.e. over the risk the user accepts |
| reports LPIPS vs speedup | reports calls needed to certify a target, and the fraction of samples that meet it |
| no failure statement | a per-sample statement of what can go wrong and how often |

## What to run next, in order

1. Measure `g` at the *same* step count as the schedules (50 steps, 17 frames, 16 prompts,
   ~50 single-skip runs, one GPU, about 40 min).
2. Reuse the per-prompt data already generated (b16 / b22 / b28) and compute the ratios.
3. If the bound holds, fit the interaction term and re-check on b28 (unseen schedule).
