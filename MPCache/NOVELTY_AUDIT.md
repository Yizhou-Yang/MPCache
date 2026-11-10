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
