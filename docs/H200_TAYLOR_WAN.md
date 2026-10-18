# H200 runbook: the single deferred row (`wan`, TaylorSeer, n=100)

Everything else is completed on the V100.  This row is deferred because the Taylor cache alone is
90-135 GiB (see `STATUS.md`, "Wan TaylorSeer: measured footprint and deferral to H200"), which is
3-4x the V100's 31.7 GiB and 3.5-5x the host's free RAM.  H200 (141 GiB) is the capacity class the
paper used; a 96 GiB H20 is **not** guaranteed to fit and must be measured before committing.

## Preconditions

- Clean pinned upstream (`RA-CFGCache` at `e17d6a7`); do **not** copy the V100 tree.  The v100
  profile has the text-encoder parking, Wan VAE tiling and an adapted pipeline; the h20 profile
  refuses such a tree on purpose (`apply_repro_patch.py` checks for the marker).
- `racfg` conda env with torch/CUDA wheels that support the card; bf16 (paper precision).
- Wan2.1-T2V-1.3B-Diffusers weights and the 100-prompt VBench list
  (`repro/upstream/RA-CFGCache/resources/prompts/prompt_video.txt`, 100 non-empty lines).
- No CPU offload, ever.  If it OOMs, preserve the log and report the failure.

## 1. Environment

```bash
export VIEWDIT=/root/viewdit
export RACFG_PROFILE=h20
export RACFG_DTYPE=bfloat16
export CPU_OFFLOAD=0
export ALLOW_GIT_PUSH=0
bash /root/viewdit/repro/setup_h20.sh
```

`setup_h20.sh` clones, installs and downloads; review it before running and check the disk.  Confirm
afterwards: `nvidia-smi` shows the card, `torch.cuda.is_bf16_supported()` is true, and the prompt
file has exactly 100 non-empty lines:

```bash
python3 -c "from pathlib import Path;print(sum(bool(x.strip()) for x in Path('/root/viewdit/repro/upstream/RA-CFGCache/resources/prompts/prompt_video.txt').read_text().splitlines()))"
```

## 2. Smoke: two consecutive prompts, then watch the peak

```bash
bash /root/viewdit/repro/run_repro.sh wan 0 2 taylorseer
```

While it runs, sample the real footprint (this is the go/no-go measurement):

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv -l 5
```

- Peak < ~130 GiB and both videos valid -> proceed.
- OOM -> write it down (step, block, allocated) and stop.  Do not reduce frames, resolution,
  `max_order` or `interval`; that changes the method and the row can no longer be paired with the
  published number.  Report "not reproducible on this card" instead.

Also check the log for the three things the V100 profile used to hide: `release_text_encoder=0`,
no `pipe.enable_tiling` for the VAE, and the model resident on the GPU for the whole loop.

## 3. Full row

```bash
CHUNKS="wan:0:100:taylorseer" bash /root/viewdit/repro/queue_overnight.sh
```

The runner skips prompts whose `video_%06d.mp4` already exists, so re-running is safe and does not
create a second sample for the same global id.  Results land in `results/repro_h20/wan/taylorseer/
gpu_only_v2/p000-100/` — the h20 root, never mixed with the V100 tree.

## 4. Aggregate and report

```bash
RACFG_PROFILE=h20 /root/miniconda3/envs/racfg/bin/python /root/viewdit/repro/aggregate.py \
  --model wan --n 100 --device cuda:0
```

Report the row with a hardware column: it is an H200 (bf16) result next to V100 (fp16) rows, so it
is not a like-for-like speedup comparison.  The quality numbers are still comparable because they
are evaluated against a same-hardware, same-seed Vanilla50 reference — if the H200 run does not also
regenerate Vanilla50 for those 100 prompts, the paired reference must come from that run, not from
the V100 one.

## 5. What success looks like

- `results/repro_h20/wan/taylorseer/gpu_only_v2/p000-100` has 100 valid videos.
- Peak memory recorded, no CPU offload in the log, no silent protocol change.
- A table row that can be footnoted as "H200, bf16, TaylorSeer N=3 O=1, interval 3, max_order 1,
  first_enhance 3" and compared against RA-CFGCache Table 1.
