# H20 deployment playbook (MPCache)

Every H20 allocation is a fresh machine: environment, weights, and code must be restored.
This page records what has to be rebuilt and the traps we already hit, so a redeploy is a
single script run plus one smoke test.

## Split of work

| Hardware | Models |
|---|---|
| V100-32GB | **CogVideoX-2B only** (17-frame development protocol; 49-frame runs also fit) |
| H20-96GB | Wan2.1-T2V-1.3B, HunyuanVideo (13B), and the 100-prompt production runs |

Never schedule Wan or HunyuanVideo on the V100: 13B does not fit in 32 GB, and the Wan
TaylorSeer variant needs about 90 GB.

## One-shot deploy

```bash
scp repro/bootstrap_h20.sh root@<h20>:/root/
ssh root@<h20> 'bash /root/bootstrap_h20.sh /root/viewdit'
```

The script creates the `racfg` conda env (python 3.10), installs the pinned packages,
pulls the three weight sets through the mirror with retries, and runs a one-sample smoke
test. It is idempotent: a reclaimed machine can be rebuilt by re-running it.

## Pinned environment (reference: working V100 host)

| Package | Version |
|---|---|
| python | 3.10 |
| torch | 2.10.0+cu128 |
| diffusers | 0.37.0 |
| transformers | 5.2.0 |
| numpy | 2.x |

`HunyuanVideoPipeline` is available from diffusers 0.37; no extra model code is needed.

## Weights (approximate size)

| Model | Size | Source |
|---|---|---|
| CogVideoX-2b | 13 GB | `THUDM/CogVideoX-2b` |
| Wan2.1-T2V-1.3B-Diffusers | 27 GB | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` |
| HunyuanVideo | ~60 GB | `hunyuanvideo-community/HunyuanVideo` |

Total ≈ 100 GB. Check `df -h /root` first; the V100 host has 328 GB free.

## Traps already hit

1. **System python lacks numpy.** Always call `/root/miniconda3/envs/racfg/bin/python`,
   never bare `python`.
2. **HuggingFace is unreachable, the mirror is not.** `curl https://huggingface.co` returns
   000; `hf-mirror.com` returns 200. Export `HF_ENDPOINT=https://hf-mirror.com`.
3. **Mirror handshakes time out mid-download.** Downloads are resumable: re-run the same
   `snapshot_download` instead of deleting the directory.
4. **`huggingface-cli` may be missing** even when `huggingface_hub` is installed; call
   `snapshot_download` from python instead of the CLI.
5. **Long runs die with the ssh session.** Use
   `setsid nohup ... > logs/<name>.log 2>&1 < /dev/null &`, and verify with `ps -eo pid,cmd`.
6. **`pgrep -f mpcache_controller` matches the ssh command itself** and reports a false
   "RUNNING". Check with `ps -eo pid,etimes,cmd | grep ...` instead.
7. **The RA-CFGCache reproduction harness lived on the previous H20 host** and is not on the
   V100. When redeploying, either restore that checkout or plan to re-implement the
   baselines for a new backbone.
8. **Disk and VRAM checks before a long queue:** `df -h /root`, `nvidia-smi`, and one
   smoke sample per model.

## After deploy: smoke checklist

1. One CogVideoX-2B sample at `--Ns 13 --planners reuse,tea_gap` prints two rows without NaN.
2. Peak VRAM for the model you are about to queue is below the card size
   (`torch.cuda.max_memory_allocated()` is printed by the controller).
3. Output JSONL grows (`wc -l results/video/<run>/closed.jsonl`).
4. Kill test: `kill <pid>` stops the process and leaves the JSONL intact (the controller is
   append-only).

## Standing queues

| Queue | Hardware | Command |
|---|---|---|
| Cog development, 16 samples, N=10/13/17 | V100 | `--frames 17 --nprompt 8 --seeds 3000,3001 --Ns 10,13,17 --planners reuse,tea_mp,tea_gap` |
| Cog production, 49 frames, 100 prompts | H20 | `--frames 49 --nprompt 100 --seeds <0..99> --Ns ...` |
| Wan / Hunyuan reproduction + caching | H20 | model-specific runner (to be written) |
