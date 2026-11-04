# MPCache — where to spend the exact steps

Training-free caching accelerates a video diffusion model by evaluating a subset of the
denoising steps and reusing the last exact output elsewhere. This repository separates
the *method* from the *benchmarks* from the *experiments*.

```
sdk/           the benchmark-agnostic layer: solver, controller, instrumentation, metrics, runner
               (a new benchmark is one file of ~20 lines under sdk/mpcache/backends/)
experiments/   configs, drivers, results; one JSON per table or figure
```

## The idea

A skipped step costs what the trajectory inherits from it. We measure that cost once per
sampler by removing one step at a time, and find it concentrated at the head of the
trajectory and independent of the prompt. Placing `B` exact calls is then an allocation
program,

```
min over S, |S| = B, T-1 in S   of   sum_{i not in S} w_i ,
```

whose exact solution is greedy on the measured profile. A model-predictive controller
would re-solve it at every step; because `w` is a property of the sampler rather than of
the trajectory, re-solving returns the same set, and the loop collapses to an open-loop
schedule with an exact head and gaps that grow along the tail.

## Quickstart

```python
from mpcache import solve, geometric
solve(profile, budget=20)             # exact allocation from a measured profile
geometric(budget=20, T=50, head=12)   # the parametric schedule used in the paper
```

```bash
python -m mpcache.runner --backend cogvideox --model /weights/CogVideoX-2b \
  --prompts prompts.txt --frames 49 --steps 50 --budget 20 --out out/ours --ledger metrics.jsonl
python -m mpcache.metrics out/orig out/ours
```

## Backends

| name | model | status |
|---|---|---|
| `cogvideox` | CogVideoX-2B (49 and 17 frames) | used for the paper's main table |
| `wan` | Wan2.1-T2V-1.3B (81 frames) | used for the cross-backbone check |
| `hunyuan` | HunyuanVideo (13B) | adapter written, not yet exercised end to end |

## Status

`experiments/results/` holds every number reported in the paper. Raw videos are not
versioned; the scripts regenerate them from the released weights, and `sdk/mpcache/ledger.py`
records every arm to JSONL as it finishes, because machines get reclaimed between arms.
