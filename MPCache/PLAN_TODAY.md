# Plan for the next machine session

Three experiments and one writing pass. Each has a decision rule, so the result is useful even if
it comes out negative. Total about four hours of machine time plus an hour of writing.

## A. A qualitative figure that shows something (highest value, ~25 min on four GPUs)

**Why.** The grid we generated at twenty-two calls is visually indistinguishable across methods, so
it cannot go in the paper. At a tighter budget the baselines are known to break, and our schedule is
known to hold, so the same figure becomes evidence instead of decoration.

**What.** Six prompts with motion or faces, 49 frames, four arms inside the RA pipeline:
`original`, `TeaCache`, `CFGCache`, and ours at **twelve calls** (fewer calls than the baselines
take, which makes the comparison favourable to them, not to us).

**Command.**

```bash
RA_REPO=/root/RA-CFGCache COG=/root/viewdit/weights/CogVideoX-2b N=6 \
  bash MPCache/sourcecode/scripts/latest/make_qualitative.sh
# ours-at-twelve needs its own schedule: 0..7,12,18,26,36,49 (12 calls)
MPC_SCHED="0,1,2,3,4,5,6,7,12,18,26,36,49" \
  bash RUN/demo_cogvideox.sh --mode CFGCache --gpus 3 --limit 6 --num_frames 49 \
  --model_path "$COG" --output_dir /root/qualA/ours12
```

**Decision rule.** If any baseline shows a visible artefact at a frame we can point to, take the
figure, circle it, and put it next to Table 1. If all four still look the same, drop the qualitative
figure from the paper entirely and say so in the appendix; a figure that shows nothing is worse than
no figure.

## B. Does the damping law survive on Wan at a budget where reuse is mild? (~30 min)

**Why.** The Wan sweep failed at twenty-one calls, but there every arm sat at 0.40 and the schedule
dominated. The law may simply live in the regime where reuse is not catastrophic, and that regime is
reachable by spending more calls.

**What.** Wan2.1 at 81 frames, a thirty-call schedule, alpha in {0, 0.5, 1.0}, eight prompts, one
reference. Same script as before (`reuse_wan.py`, plain diffusers harness).

**Decision rule.** If the inverted U appears at thirty calls, the law is scoped by *budget* rather
than by backbone, and the paper can say so precisely. If it stays monotone, the law is a Cog result
and the appendix keeps the current wording.

## C. Finish the 49-frame damping curve (~20 min)

Alpha 0.75 at 49 frames, 100 prompts, to complete the five-point curve that currently has four.

```bash
python reuse_policy.py --model /weights/CogVideoX-2b --prompts prompts_video.txt \
  --frames 49 --policy extrap --alpha 0.75 --offset $((sid*25)) --limit 25 --out out/e075/s$sid
```

**Decision rule.** None; it fills a hole in a table that already has a clear optimum.

## D. Submission-readiness pass (no GPU, ~1 hour)

- Every number in the prose against `experiments_results/main_table.json`.
- Figure and table numbering, and every `\ref` resolving.
- Related work: keep the three families, drop any sentence that only says a competitor's component
  is decorative.
- Claims audit: each sentence in the abstract must map to a measurement or a derivation.

## Order

Do **A** first (it decides whether the paper has a qualitative figure at all), then **B** (it decides
the scope paragraph), then **C**, then **D** while the GPU is idle.
