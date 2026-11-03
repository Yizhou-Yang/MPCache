# MPCache — where to spend the exact steps

Training-free caching accelerates a video diffusion model by evaluating a subset
of the denoising steps and reusing the last exact output elsewhere. This
repository separates the *method* from the *experiments* that support it.

```
project/       the method: the allocation solver, the schedule we ship, the injection hooks, tests
experiments/   configs, drivers, evaluator and the evaluated numbers for every table and figure
```

## The idea in one paragraph

A skipped step costs what the trajectory inherits from it. We measure that cost
once per sampler by removing one step at a time, and find it concentrated at the
head of the trajectory and independent of the prompt. Placing `B` exact calls
then becomes an allocation problem,

```
min over S, |S| = B, T-1 in S   of   sum_{i not in S} w_i ,
```

whose exact solution is greedy on the measured profile: keep the last step, then
the `B-1` most expensive ones. A model-predictive controller would re-solve this
at every step; because `w` is a property of the sampler, re-solving returns the
same set, so the loop collapses to an open-loop schedule with an exact head and
gaps that grow along the tail.

## Quickstart

```python
from mpcache import solve, geometric
solve(profile, budget=20)                 # exact allocation from a measured profile
geometric(budget=20, T=50, head=12)       # the parametric schedule used in the paper
```

```bash
bash experiments/scripts/run_main_table.sh TeaCache 17 100 out/ours "<schedule>"
python experiments/scripts/eval_lpips_vgg.py out/orig out/ours
```

## Status

`experiments/results/` holds every number reported in the paper, including the
baseline reproduction check. Raw videos are not versioned; the scripts regenerate
them from the released weights.

## Layout rationale

`project/` has no dependency on the harness or on the paper's paths, so the
scheduler can be imported by another pipeline. `experiments/` depends on both the
harness and the weights, and holds only what a reader needs to reproduce a row of
a table: the schedule, the settings, the seed rule, and the numbers.
