# Main-table TODO (updated 2026-10-23)

Status legend: `[x]` done · `[~]` partial / needs re-run · `[ ]` todo

## 1. Matrix

Backbones: CogVideoX-2B (49f, DDIM-50, CFG 6), Wan2.1-T2V-1.3B (81f, UniPC-50, shift 3, CFG 5),
HunyuanVideo-13B (49f, 50 steps).
Protocol: 100 VBench prompts from `resources/prompts/prompt_video.txt`, seed = prompt index,
negative prompt `animation`, batch one. Reference = the 50-step full run of the same prompt and seed.

| arm | Wan-81 | Cog-49 | Hunyuan |
|---|---|---|---|
| Full (reference) | [x] ours 100 + RA 100 | [ ] | [ ] |
| Ours (MPCache) | [~] tea_gap 100 done, **method now uses ex1 on uniform grids** | [ ] | [ ] |
| TeaCache | [x] RA harness ~100 | [ ] | [ ] |
| TaylorSeer | [ ] | [ ] | [ ] |
| RA-CFGCache-Slow | [ ] | [ ] | [ ] |
| RA-CFGCache-Fast | [ ] | [ ] | [ ] |
| metrics | [~] ours-vs-ourfull only | [ ] | [ ] |

**Why "Ours" on Wan changes.** The schedule produced by the flow solver is uniform (gaps all 4),
so `r = age / previous gap = 1` and the base forecast is a full first-order extrapolation. The
controller uses the base forecast alone on a uniform grid (Section 3.4, "Which forecaster to
trust"), which is exactly the `ex1` arm. Wan therefore needs one arm: `--save-op ex1`.

Cog is unaffected: no Cog schedule between N=8 and N=17 is uniform, so `--auto-base 1` never
fires and every existing Cog number stands (`autobase16` verifies this bit-for-bit).

## 2. Runs (4 x H20)

### 2.1 Wan (13 steps, ~0.6 h per arm)

```bash
P=/root/miniconda3/envs/racfg/bin/python
cd /root/viewdit
for op in ex1 taylor ra_slow ra_fast; do
  for i in 0 1; do
    CUDA_VISIBLE_DEVICES=$i setsid nohup $P mpcache_wan.py \
      --seed-from-prompt --negative "animation" --solver unipc --sched sigma \
      --no-full --save-video --save-op $op --save-n 13 \
      --nprompt 50 --prompt-offset $((i*50)) \
      --out results/wan_arms/$op > logs/wan_${op}_p$i.log 2>&1 < /dev/null &
  done
done
```

`taylor/ra_slow/ra_fast` are **not** our controller: they are the RA harness
(`models/wan/sample.py --taylor_method taylor` and `--mode CFGCache` with the two thresholds).
If the arms are launched through the RA harness, keep `--limit 100` and shard with
`--start_index` instead of `--prompt-offset`.

### 2.2 CogVideoX-2B, 49 frames (~0.9 h total)

```bash
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i setsid nohup $P experiments/video/mpcache_controller.py \
    --frames 49 --nprompt 25 --prompt-offset $((i*25)) --seed-from-prompt \
    --Ns 13 --planners reuse,tea_gap --auto-base 1 \
    --out results/video/cog49_p$i > logs/cog49_p$i.log 2>&1 < /dev/null &
done
```

Baselines for Cog run through the RA harness (`models/cogvideox/sample.py`), 49 frames,
720x480, DDIM-50, CFG 6, `--limit 100`.

### 2.3 HunyuanVideo-13B (the long pole: Full ~8.7 h, 13-step arm ~2.3 h)

Start the Full arm first on two GPUs, everything else fits behind it:

```bash
for i in 0 1; do
  CUDA_VISIBLE_DEVICES=$i setsid nohup $P mpcache_hunyuan.py \
    --seed-from-prompt --negative "animation" --no-full --save-video \
    --save-op original --nprompt 50 --prompt-offset $((i*50)) \
    --out results/hy_full > logs/hy_full_p$i.log 2>&1 < /dev/null &
done
```

If the wall clock is short, run Hunyuan on 50 prompts and say so in the caption ("50 prompts"),
or drop its TaylorSeer arm; the fidelity deltas are paired either way.

## 3. Evaluation (once videos exist)

```bash
python /root/RA-CFGCache/scripts/eval.py \
  --ref results/ra_wan/wan_original_full/videos \
  --cmp results/wan_arms/<arm>/videos \
  --out results/metrics/wan_<arm>.csv
```

Every arm is scored against the **same** reference videos of its backbone (RA Full for Wan,
our full50 for Cog, ours for Hunyuan) so that the LPIPS/SSIM/PSNR columns are paired.

## 4. Order of work (critical path first)

| slot | GPU 0-1 | GPU 2-3 |
|---|---|---|
| 0-2.4 h | Hunyuan Full | Wan ex1 -> TaylorSeer -> RA-Slow -> RA-Fast |
| 2.4-3.3 h | Hunyuan Full | Cog Full -> Cog ours |
| 3.3-4.0 h | Hunyuan Full | Cog TeaCache -> TaylorSeer -> RA-Slow -> RA-Fast |
| 4.0-6.3 h | Hunyuan Full | Hunyuan ours |
| 6.3-8.7 h | Hunyuan Full | Hunyuan TeaCache |
| 8.7-11 h | Hunyuan RA-Slow | Hunyuan TaylorSeer (optional) |

## 5. Ablations already fixed on V100 (no H20 time needed)

Budget curve N=8..17, gate age 1/2/3, trust, full-rank model, kappa, bias, RLS, per-token mask,
ex1 counterfactual on Cog, auto-base integrity. All on the VBench prompt set, prompts 0-15,
seed 3000, archived in `repro/ABLATIONS.md`.
