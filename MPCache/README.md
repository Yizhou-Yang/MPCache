# MPCache — where to spend the exact steps

Training-free caching accelerates a video diffusion model by evaluating a subset of the
denoising steps and reusing the last exact output elsewhere. This repository separates the
method from the benchmarks from the traces, in the layout of CuratorMem.

```
experiments_results/     every trace behind every number in the paper
sourcecode/              the method, the benchmark adapters and the harness
```

## Layout

### `sourcecode/`

```
mpcache/                 the method, installable on its own
  schedule.py              the allocation solver: greedy on a profile, plus the geometric family
  controller.py            the step state machine: counts calls, decides evaluate or reuse
  instrument.py            wraps any forward method; no hooks, no state_dict changes
  profile.py               the single-skip measurement protocol, parameterised by a `run` callable
  metrics.py               LPIPS(VGG) / SSIM / PSNR over matched videos
  ledger.py                append results to JSONL the moment they exist
  runner.py                the one runner every benchmark uses
src/latest/              per-benchmark adapters, one file each, ~20 lines
  cogvideox.py  wan.py  hunyuan.py
benchmarks/loader.py     unified loader: a name maps to an adapter; the method package stays clean
scripts/latest/          runners, the freeze driver, the ledger sync, the RA recipe
tests/                   invariants: budget exact, last step exact, cache reuse, restore clean
```

### `experiments_results/`

```
configs/                 one JSON per table or figure: schedule, settings, seed rule
main_table.json          the paper's main table, evaluated
results_final_*.json     the full inventory of evaluated numbers
DATA_STATE.md            what is frozen, what is missing, and the traps that cost time
```

## The idea

A skipped step costs what the trajectory inherits from it. We measure that cost once per
sampler by removing one step at a time, and find it concentrated at the head of the
trajectory and independent of the prompt. Placing `B` exact calls is then an allocation
program,

```
min over S, |S| = B, T-1 in S   of   sum_{i not in S} w_i ,
```

whose exact solution is greedy on the measured profile. A model-predictive controller would
re-solve it at every step; because `w` is a property of the sampler rather than of the
trajectory, re-solving returns the same set, and the loop collapses to an open-loop schedule
with an exact head and gaps that grow along the tail.

## Quickstart

```python
from mpcache import solve, geometric
solve(profile, budget=20)             # exact allocation from a measured profile
geometric(budget=20, T=50, head=12)   # the parametric schedule used in the paper
```

```bash
PYTHONPATH=sourcecode python -m mpcache.runner --backend cogvideox \
  --model /weights/CogVideoX-2b --prompts prompts.txt --frames 49 --steps 50 \
  --budget 20 --out out/ours --ledger out/metrics.jsonl
PYTHONPATH=sourcecode python -m mpcache.metrics out/orig out/ours
```

## Benchmarks

| name | model | status |
|---|---|---|
| `cogvideox` | CogVideoX-2B (49 and 17 frames) | the paper's main table |
| `wan` | Wan2.1-T2V-1.3B (81 frames) | cross-backbone check |
| `hunyuan` | HunyuanVideo (13B) | adapter written, not yet exercised end to end |

## Adding a benchmark

One file in `src/latest/` exporting `backend`, one line in `benchmarks/loader.py`. The
schedule, the controller, the instrumentation, the ledger and the metrics are already wired.

## Reproducibility rule

Every arm is evaluated the moment it finishes and appended to `metrics.jsonl`
(`mpcache/ledger.py`); `scripts/latest/sync_results.sh` pushes that ledger from the GPU box
every ten minutes. Machines get reclaimed between arms, and a number that is not on disk
when the box disappears is gone.
