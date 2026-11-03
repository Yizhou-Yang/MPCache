# project/ — the method

A training-free cache decides, once, which denoising steps to evaluate. This
package solves that allocation given a measured sensitivity profile.

| file | what it is |
|---|---|
| `mpcache/schedule.py` | the open-loop solution: greedy allocation on a profile, and the geometric schedule we ship |
| `mpcache/hooks.py` | the two edits that inject a schedule into the RA-CFGCache release |
| `tests/test_schedule.py` | invariants: budget is exact, the last step is always evaluated, the greedy rule keeps the head |

## The problem

With horizon `T`, budget `B` of transformer calls and a per-step sensitivity
`w_i` (the terminal loss caused by skipping step `i` alone),

```
min over S, |S| = B, T-1 in S   of   sum_{i not in S} w_i
```

whose exact solution is greedy: keep the last step, then the `B-1` steps with
the largest `w`. A model-predictive controller would re-solve this at every
step. Because `w` is a property of the sampler rather than of the trajectory,
re-solving returns the same set, so the loop collapses to this open-loop
schedule.

## Quickstart

```python
from mpcache import solve, geometric, head_length, relative_gap

w = [0.0, 0.569, 0.452, 0.313, 0.192, 0.118, 0.094, 0.063]  # measured, normalised
solve(w, budget=5)                 # exact allocation
geometric(budget=20, T=50, head=12)  # what we ship, no profile needed beyond its head
```

## Invariants

`check(profile, budget, schedule)` asserts that the number of calls equals the
budget, that the last step is exact, and that no step appears twice. Every
schedule in the paper satisfies them.
