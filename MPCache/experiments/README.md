# experiments/ — everything that produces a number in the paper

The method itself lives in `../project/`. This directory holds what is needed to
regenerate the tables and figures, and the small JSON that records every number
we report.

```
experiments/
  configs/    one JSON per table or figure: the schedule, the settings, the seed rule
  scripts/    the shell drivers and the evaluator
  results/    the evaluated numbers (small, versioned); raw videos are not kept
```

## Rules that every experiment here follows

1. **One harness, one reference.** All arms of a setting run in the released
   RA-CFGCache pipeline; the reference is a full run of that same pipeline.
2. **Budget, not wall clock.** An arm is described by the number of transformer
   calls; two arms are compared only when the counts match to within one.
3. **VGG LPIPS.** On the same pair of videos AlexNet reports 0.041 where VGG
   reports 0.118; mixing backbones is the easiest way to be off by 3x.
4. **Paired prompts.** Seed equals prompt index, so a per-prompt comparison is
   paired; we report n, mean, median and win rate.

## Reproduce the main table

```bash
bash scripts/run_main_table.sh   # needs the weights and the RA release
python scripts/eval_lpips_vgg.py <ref_dir> <arm_dir>
```

The numbers land in `results/main_table.json`; the schedules come from
`configs/main_table.json` and are identical to `project/mpcache`'s output for the
measured profile.
