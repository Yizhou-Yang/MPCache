# Reproducing the MPCache results

This directory holds everything needed to regenerate every number in the paper from
raw traces: the code, the per-run JSONL traces, and the analysis scripts.

```
repro/
  code/        controller + probes (the exact scripts that produced the traces)
  results/     one directory per run; every row is one (prompt, seed, N, op)
  analyze/     paired statistics, gap diagnostics, ablations archive
  ABLATIONS.md generated catalogue of all runs, including negative ones
```

## 0. Environment

The traces were produced on **V100-32GB** with

| package | version |
|---|---|
| python | 3.10 |
| torch | 2.10.0+cu128 |
| diffusers | 0.37.0 |
| transformers | 5.2.0 |

On H20 (driver CUDA 12.2) the ceiling is `torch==2.5.1+cu121`; the controller is
version-agnostic. Weights: `zai-org/CogVideoX-2b` (was `THUDM/CogVideoX-2b`), placed at
`/root/viewdit/weights/CogVideoX-2b`.

## 1. Row format

`results/video/<run>/closed.jsonl`, one JSON object per line:

```json
{"prompt": 0, "seed": 3000, "N": 13, "op": "tea_gap",
 "rel": 0.1104, "nfe": 13, "steps": [0,1,2,3,5,7,9,12,15,19,23,27,29]}
```

`rel` is the relative L2 of the terminal latent against the all-exact run of the same
(prompt, seed). `nfe` equals `N` by construction, so all comparisons are at matched NFE.

## 2. Regenerating each artifact

All commands run from `repro/`.

### 2.1 Core result (CogVideoX-2B, 17 frames, 16 samples)

```bash
python analyze/paired.py --name "core: tea_gap vs reuse" \
  --cand results/video/mpc_20261021a:tea_gap \
  --base results/video/mpc_20261021a:reuse
```

Expected: N=10 `+0.103 / win 1.00 / 2SEM True`; N=13 `+0.034 / 1.00 / True`;
N=17 `+0.048 / 1.00 / True`.

### 2.2 Extended budgets N=12 and N=16

```bash
python analyze/paired.py --name "extended budgets" \
  --cand results/video/save1:tea_gap \
  --base results/video/save1:reuse
```

Expected: N=12 `+0.017 / 0.75`; N=16 `+0.044 / 1.00`.

### 2.3 Control ablation: with and without the gap gate

```bash
python analyze/paired.py --name "no gate (tea_mp) vs reuse" \
  --cand results/video/mpc_20261021a:tea_mp \
  --base results/video/mpc_20261021a:reuse

python analyze/paired.py --name "gate (tea_gap) vs reuse" \
  --cand results/video/mpc_20261021a:tea_gap \
  --base results/video/mpc_20261021a:reuse
```

Expected: without the gate the correction is **worse** than reuse (−0.240 / −0.029 /
−0.037); with the gate it is better (+0.103 / +0.034 / +0.048). This is the ablation that
gives the control component its independent significance.

### 2.4 Pricing diagnostics (gap probe, 8 samples)

```bash
python analyze/gap_analyze.py results/video/g0a_gap_20261020/runs.jsonl
```

Prints Spearman(score, terminal error) per gap for the oracle output error, the solver
prior and the online model, plus calibration ratios.

### 2.5 Single-skip probe (16 samples, 192 rows)

```bash
python analyze/d_identifiability.py results/video/g0a_costmodel_20261015/runs.jsonl
```

Reported numbers: oracle ordering 0.85 within-run; online price vs terminal error 0.33;
solver prior 0.40; selector improves to 0.65.

### 2.6 Two-signal skip-budget simulation

```bash
python analyze/two_signal_sim.py results/video/g0a_costmodel_20261015/runs.jsonl
```

Expected share of terminal error at k=2..8 for the oracle order, the prior and the online
model (Table in the ablation section of the paper).

### 2.7 Full ablation catalogue

```bash
python analyze/archive_runs.py results/video ABLATIONS.md \
  results/video/mpc_20261021a/closed.jsonl
```

Regenerates `ABLATIONS.md`: every run, paired against `reuse`, including the negative
ones (schedulers, price gating, branch decomposition, trust/κ/bias sweeps, compute-saving).

## 3. Re-running an experiment (optional)

```bash
python code/mpcache_controller.py --frames 17 --nprompt 8 --seeds 3000,3001 \
  --Ns 10,13,17 --planners reuse,tea_mp,tea_gap --gate-age 2 \
  --out results/video/<new_run>
```

Flags that define the reported configuration: `--gate-age 2` (default), `--trust -1`
(β gate), `--kappa 1.0`, `--rls-gamma 0.5`, `--bias 0`, `--reserve-last 1`.

Development protocol: 17 frames, 720×480, 30 DDIM steps, CFG 6, prompts 0–7,
seeds 3000/3001. Held-out prompts are 8–15 (`--prompt-offset 8`).

## 4. What is *not* here

- The 100-prompt production runs (49 frames, CogVideoX-2B) live on the H20 cluster; see
  `h20/RUNBOOK.md` for the queue that produces them.
- Wan2.1 and HunyuanVideo traces are produced by the H20 runners and are not in this
  directory yet.
- The baseline reproduction table (TeaCache / TaylorSeer / RA-CFGCache, 100 prompts on
  CogVideoX-2B and Wan2.1-1.3B) was produced with the public RA-CFGCache harness and is
  summarised in the paper; its raw rows are under `results/` on the H20 cluster.
