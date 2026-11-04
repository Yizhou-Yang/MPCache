# experiments/ — everything that produces a number in the paper

The method and the benchmark adapters live in `../sdk/`. This directory holds only what
is specific to producing and recording the paper's numbers.

```
experiments/
  configs/    one JSON per table or figure: the schedule, the settings, the seed rule
  scripts/    drivers
  results/    the evaluated numbers (small, versioned); raw videos are not kept
  DATA_STATE.md   what is frozen, what is missing, and the traps that cost time
```

## Two drivers

| script | when to use |
|---|---|
| `scripts/run_arm.sh` | any benchmark, through the SDK runner (`mpcache.runner`) — the general path |
| `scripts/run_ra_harness.sh` | the released RA-CFGCache pipeline, which produced the paper's numbers; the injection is documented in `scripts/ra_injection.md` |

Both write to a `metrics.jsonl` ledger through `sdk/mpcache/ledger.py`, one row per arm,
at the moment the arm finishes. `scripts/sync_results.sh` pushes that ledger from the GPU
box every ten minutes, so a sleeping laptop or a reclaimed machine cannot lose the numbers.

## Rules every experiment here follows

1. **One harness, one reference.** All arms of a setting share a pipeline; the reference
   is a full run of that same pipeline.
2. **Budget, not wall clock.** An arm is described by its number of transformer calls; two
   arms are compared only when the counts match to within one.
3. **VGG LPIPS.** On the same pair of videos AlexNet reports 0.041 where VGG reports 0.118.
4. **Paired prompts.** Seed equals prompt index, so a per-prompt comparison is paired; we
   report n, mean, median and win rate.
5. **Evaluate per arm.** Never batch the evaluation to the end of a run.
