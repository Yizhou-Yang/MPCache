# MPCache — Risk-Certified Horizon Synthesis & Phase-Gated Midpoint Shrinkage

Training-free caching accelerates video Diffusion Transformers (DiTs) by evaluating $B < T$
anchor steps and reconstructing skipped guided velocities from cached evaluations. `MPCache`
formulates caching as a coupled bilevel operator-design problem combining **boundary-pinned
diffusive Bellman DP**, **conformal minimum-budget certification**, and **2-slot phase-gated
midpoint chord-shrinkage**.

```
experiments_results/     every trace behind every number and theorem in the paper
sourcecode/              the method SDK, benchmark adapters, verification scripts, and tests
```

## Layout

### `sourcecode/`

```
mpcache/                 installable zero-training framework
  schedule.py              Bellman DP (`solve_dp`, `solve_dp_table`), KKT density (`kkt_schedule`),
                           bifurcation detector (`detect_bifurcation_step`), stiff-sloppy analysis
  certify.py               conformal risk-certified minimum-budget synthesis (`certify_budget`)
  shrinkage.py             2-slot phase-gated midpoint chord-shrinkage (`ChordShrinkagePredictor`)
  controller.py            step state machine coupling outer schedule S* and inner shrinkage R_alpha
  instrument.py            wraps any forward method; no weight or state_dict changes
  profile.py               O(T) closed-loop terminal impulse measurement protocol
  metrics.py               VGG-LPIPS / SSIM / PSNR over paired videos
  ledger.py                append results to JSONL immediately upon completion
  runner.py                unified CLI (`--policy dp|kkt|geometric`, `--predictor midpoint|phase_gated`)
src/latest/              per-benchmark adapters (`cogvideox.py`, `wan.py`, `hunyuan.py`)
scripts/latest/          runners and `verify_framework_traces.py` (>2,600 GPU trace verification)
tests/                   18 unit tests covering Bellman optimality, KKT laws, MPC invariance,
                         conformal budget certification, and phase-gated Hilbert shrinkage
```

### `experiments_results/`

```
configs/                      JSON configs per table/figure
main_table.json               100-prompt VBench end-to-end tables (CogVideoX-2B & Wan2.1-1.3B)
framework_verification.json   trace-level verification of all 4 structural laws & theorems
results_final_*.json          complete inventory of evaluated runs
```

## Core Operators & Quickstart

```python
from mpcache import (
    solve_dp,
    solve_dp_table,
    detect_bifurcation_step,
    calibrate_certificate,
    certify_budget,
    ChordShrinkagePredictor,
)

# 1. Outer Stage: exact boundary-pinned diffusive Bellman DP (Theorem 1, <1.5 ms CPU)
sched = solve_dp(profile, budget=22, beta=0.5)
h_crit = detect_bifurcation_step(profile)

# 2. Inverse Risk-Certified Budget Synthesis (Theorem 3): save 4-6 DiT calls at target LPIPS
cert = calibrate_certificate(profile, calib_schedules, mean_lpips, worst_lpips, shrinkage_multiplier=0.672)
alloc = certify_budget(profile, cert, target_lpips=0.15, risk_mode="mean", use_shrinkage=True)
# -> alloc.budget == 21 (vs. 26 without midpoint shrinkage), alloc.schedule == S*_{21}

# 3. Inner Stage: 2-slot phase-gated midpoint chord-shrinkage (Theorem 4, 0 extra FLOPs)
pred = ChordShrinkagePredictor(policy="phase_gated", alpha=0.5, regime_head=h_crit)
```

```bash
PYTHONPATH=sourcecode python -m mpcache.runner --backend cogvideox \
  --model /weights/CogVideoX-2b --prompts prompts.txt --frames 49 --steps 50 \
  --budget 22 --policy dp --predictor phase_gated --alpha 0.5 \
  --out out/ours --ledger out/metrics.jsonl
```

## Benchmarks

| name | model | status |
|---|---|---|
| `cogvideox` | CogVideoX-2B (49 and 17 frames, DDIM-50) | 100-prompt main table & predictor sweeps |
| `wan` | Wan2.1-T2V-1.3B (81 frames, UniPC-50) | 100-prompt zero-shot cross-backbone transfer |
| `hunyuan` | HunyuanVideo (13B) | adapter wired |
