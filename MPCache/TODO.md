# Hand-off — last updated 2026-11-21 03:15

Repository is in sync with `origin/h20-run`. Everything below is measured and pushed; nothing
is a placeholder.

## The paper's claim, in one paragraph

A training-free cache makes two decisions that the literature keeps apart: where to place the
exact calls, and what a skipped step returns. Placement is positional and steep (a head of
sixteen of fifty steps carries about three quarters of the sensitivity). The replacement has a
law: adding a *damped* part of the one-step difference of the model output (damping about one
half) lowers LPIPS by a third, while scaling that difference by the gap destroys the result.
Re-planning placement at every step cannot help, because a global error scale does not reorder a
fixed-budget allocation.

## The numbers, frozen

### Main table (RA-CFGCache pipeline, the competitor's own harness)

| setting | n | baseline | ours |
|---|---:|---|---:|
| CogVideoX-2B 49f | 100 | 0.1554 / 0.8973 / 29.17 | **0.1300 / 0.9247 / 31.63** |
| Wan2.1-1.3B 81f | 100 | 0.1685 / 0.8686 / 27.91 | **0.1527 / 0.8853 / 29.27** |
| CogVideoX-2B 17f | 100 | CFG 0.1587, Tea 0.2786, Taylor 0.4917 | **0.1454** |

### Replacement law (plain diffusers harness, our own; head-16 + six tail steps = 22 calls)

CogVideoX-2B, 49 frames, 100 prompts:

| what a skip returns | LPIPS | SSIM | PSNR |
|---|---:|---:|---:|
| the last exact output | 0.1549 | 0.8771 | 27.55 |
| one-step difference, damping 0.25 | 0.1214 | 0.9100 | 29.46 |
| one-step difference, damping 0.5 | **0.1041** | 0.9293 | 31.25 |
| one-step difference, damping 1.0 | 0.1176 | 0.9273 | 31.53 |
| the same, scaled by the gap | 0.4417 | 0.6129 | 15.84 |

Same sweep at 17 frames, 32 prompts: 0.1556 / 0.1239 / **0.1084** / 0.1131 / 0.1250 / 0.4419.

**The two harnesses do not share absolute numbers.** The main table lives in the RA pipeline; the
replacement law lives in plain diffusers. Never mix rows from the two in one comparison.

### Negative results

| claim | evidence |
|---|---|
| feedback cannot help with a global error scale | budgeted DP is invariant to it; three implementations, one degenerate |
| single-skip gains do not add up | R^2 = 0.59; 29% of samples exceed the additive prediction |
| gap-scaled extrapolation destroys quality | 0.4417 at 49f, 0.4419 at 17f, and TaylorSeer itself is 0.4917 at 17f |

## What is missing, in priority order

0. **The two qualitative figures (highest value, about 30 minutes on one box).** Everything is
   built and tested; only the generation is missing, because every video on this laptop comes
   from an older method (gap-scaled extrapolation, warmup-4 schedule) and using them would
   misrepresent the paper.

   ```bash
   RA_REPO=/root/RA-CFGCache COG=/root/viewdit/weights/CogVideoX-2b \
     bash MPCache/sourcecode/scripts/latest/make_qualitative.sh
   ```

   Produces `qual_A.pdf` (Original / TeaCache / RA-CFGCache / Ours, all inside the RA harness,
   matching Table 1) and `qual_B.pdf` (Original / plain reuse / damped 0.5 / scaled by gap,
   inside the diffusers harness, matching Table 2). Three prompts, four columns, 49 frames,
   four GPUs in parallel.

   The composer takes red circles once we have looked at the frames:
   `--circle row,col,x,y,r` in fractions of a cell. Insertion points: `qual_A.pdf` goes next to
   Table 1 as a `figure*`, `qual_B.pdf` next to Table 2. Caption drafts:

   > **A.** The same prompt and seed under the policies of Table 1, all produced by one
   > harness. Our schedule holds the first sixteen steps exact; the baselines decide with a
   > threshold. Speedups in parentheses are measured, not quoted.

   > **B.** The same schedule, four replacements. Plain reuse and the gap-scaled correction
   > blur or break the moving detail; the damped correction keeps it.

1. **Alpha 0.75 at 49 frames.** The videos exist on the 2026-11-20 machine (100/100 at
   `/root/rp2/e075`); the evaluation did not finish before the box went away. Re-run
   `evaluate_reuse.sh` against `/root/rp/full` if that box returns, otherwise regenerate.
2. **TaylorSeer in the replacement comparison at 49 frames.** The law's strongest claim is that
   a gap-scaled predictor fails; that is currently supported by our own scaled variant plus
   TaylorSeer at 17 frames only.
3. **HunyuanVideo 17 frames.** Adapter written, never exercised. Deferred by decision; belongs in
   the final version, not before.
4. **A figure for the damping law.** One line, three curves (damping 0.5, damping 1.0, gap-scaled)
   against LPIPS; the table alone undersells it.

## How to resume

```bash
# 1. environment (torch 2.10.0 / torchvision 0.25.0 is required; 2.5.1 dies with SIGFPE)
#    see the deploy script used on 2026-11-20; weights: zai-org/CogVideoX-2b and
#    Wan-AI/Wan2.1-T2V-1.3B-Diffusers into /root/viewdit/weights

# 2. one arm of the replacement comparison, sharded over four GPUs
for sid in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$sid python reuse_policy.py --model /weights/CogVideoX-2b \
    --prompts prompts_video.txt --frames 49 --policy extrap --alpha 0.5 \
    --offset $((sid*25)) --limit 25 --out out/e05/s$sid
done

# 3. evaluate every arm against the full reference
PYTHON=python evaluate_reuse.sh out/full out latest e025 e05 e10 scaled

# 4. compile the paper
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Pitfalls that have cost time before

- **Sharding needs the global seed.** `--offset` shifts both the prompt and the seed; without it
  shards produce different trajectories and cannot be compared to the reference.
- **File names carry the global prompt index**, so a recursive evaluation matches shards by
  basename. Do not rename outputs.
- **The boxes are recycled without notice.** Push results as soon as an arm finishes; do not
  batch a night of runs and push in the morning. Two nights of numbers were lost that way.
- **GitHub push from this network fails with SSL timeouts intermittently.** Retry; the commit is
  local and safe.
- **Clone the RA repository's hooks before comparing both harnesses**, and remember the RA
  pipeline needs its own environment (see its README).
