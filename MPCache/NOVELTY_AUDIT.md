# Novelty audit — 2026-11-10

Read this before writing another sentence of the paper. Three recent papers cover most of
what the first draft claims.

## What is already published

| prior work | what it already does | what that kills |
|---|---|---|
| **SenCache** (CVPR 2026, [arXiv 2602.24208](https://ar5iv.labs.arxiv.org/html/2602.24208)) | formalises caching error through the denoiser's **local sensitivity** (Jacobian wrt latent and timestep), **calibrates it once on 8 videos**, and uses the calibrated profile to choose cache steps; has a `max cache length` (a max gap); states that caching quality is largely prompt-independent | "measure a profile offline and allocate calls by it"; "the profile does not depend on the prompt"; the max-gap constraint |
| **DPCache** (CVPR 2026, [arXiv 2602.22654](https://ar5iv.labs.arxiv.org/html/2602.22654)) | formulates acceleration as **global path planning**, builds a path-aware cost tensor on a calibration set, solves it **exactly by dynamic programming**, and **fixes the first M timesteps as mandatory** | "the schedule is the open-loop optimum of an allocation program"; "the head must be exact"; "trajectory shape is content-independent" |
| **LeMiCa** (cited in SenCache) | formulates cache scheduling as a **global path optimisation** (lexicographic minimax) | the "global allocation" framing again |

## What is left that is ours

1. **The measured object is the terminal effect, not a local one.** We measure
   `terminal loss of skipping one step alone`; SenCache measures a local Jacobian;
   DPCache measures a path cost. If the terminal ordering differs from the local one,
   then a policy built on local sensitivity misplaces calls — that is a falsifiable claim
   against SenCache, and our earlier experiments hint at it (input change fires hardest in
   the head and then abandons it).
2. **The head-length law.** H=6 → 0.288, H=12 → 0.157, H=16 → 0.133 at 17f; at 49f/100,
   H=12 → 0.142 and H=16 → 0.130. A short head is catastrophic, and the optimum at a
   twenty-call budget is at or above sixteen. DPCache fixes M=3 without such an analysis.
3. **The control framing, if it survives.** Caching can be written as a finite-horizon
   program with a terminal cost; the paper's claim would be that the framework, not the
   closed loop, is the contribution: it says *what to measure* (the value of a skip),
   *what to solve* (the horizon program) and *what the prior heuristics approximate*.

## The decisive experiment (not yet run)

Implementation is in `sourcecode/mpcache/control.py`, runner in
`sourcecode/scripts/latest/mpc_vs_dp.py`:

- **closed loop**: receding-horizon solve, drift rate updated from every exact evaluation;
- **open loop**: the same program solved once with the calibrated rate (the same policy
  family as DPCache), i.e. our shipped schedule.

Both consume exactly the same number of calls. If the closed loop is not better, then
"MPC" can only be sold as a *framework and analysis*, not as a controller; if it is better,
the controller is the paper.

Run it with

```bash
python sourcecode/scripts/latest/mpc_vs_dp.py --model /weights/CogVideoX-2b \
  --prompts prompts.txt --frames 17 --steps 50 --budget 22 --limit 32 --out out/mpcx
```

## Honest position

As framed in the first draft ("placement matters; the profile is prompt-independent; the
schedule is the open-loop optimum") the paper is **anticipated** by SenCache and DPCache.
It becomes publishable only if one of the three items above is demonstrated against those
two baselines at a matched budget, in one harness.

## Addendum, same day: the "control" framing is also taken

Searching the phrase rather than the mechanism finds four more, all recent:

| prior work | framing | what it takes from us |
|---|---|---|
| **IdealCache** (NeurIPS 2026, [arXiv 2611.10340](https://arxiv.org/abs/2611.10340), submitted 7 Nov 2026) | "abandons step-wise imitation of the reference trajectory… what truly matters is the ideal result"; schedules found by a **constrained shortest path** on a graph of ideal trajectories | both the *optimal non-uniform schedule* and the *terminal-outcome rather than local-error* argument — item ① of the earlier list |
| **SoftCap** ([link](https://www.catalyzex.com/paper/softcap-soft-budget-control-for-diffusion)) | "**Soft-Budget Control** for DiT acceleration" | the word *control* over a *budget* |
| **ReCache** ([link](https://www.semanticscholar.org/paper/f5da4f234363201fbd8c4760e0c5f042bd242e4d)) | "Learning **Budget-Aware** Caching Schedules … via **REINFORCE**" | caching as a sequential decision process (MDP) with a budget |
| **CEM-MPC** ([arXiv 2610.18677](https://export.arxiv.org/pdf/2610.18677)) | cross-entropy-method **MPC** | the receding-horizon solver itself |

**Therefore "we are the first system to model caching as a control problem" is not defensible**, and neither is
"first to optimise the schedule globally" (DPCache, IdealCache), "first to use a calibrated profile"
(SenCache), or "first to target the terminal outcome instead of the trajectory" (IdealCache).

## What this leaves

The only statements that survive a priority check are narrow and empirical:

1. **A head-length law**, measured: H=6 → 0.288, H=12 → 0.157, H=16 → 0.133 at a twenty-call
   budget (17f); H=12 → 0.142 vs H=16 → 0.130 at 49f/100. Nobody publishes this curve.
2. **The value of feedback at a matched budget**: closed loop vs the open-loop optimum, same
   calls. If feedback is worth little, that is a useful negative result against the adaptive
   line (SenCache, ReCache); if it is worth a lot, it is a method — but then the novelty is the
   controller, and CEM-MPC/ReCache are the baselines it must beat.
3. **A cheap measurement**: one single-skip ablation per (model, step count) replaces the
   calibration sets that SenCache (8 videos), DPCache (~10 samples) and IdealCache (an ensemble
   of ideal trajectories) need. "Same schedule quality, one pass of measurement" is a modest but
   real claim — and it is only real if we run those methods and match them.
