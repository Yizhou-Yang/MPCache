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
git clone -b h20-run https://github.com/Yizhou-Yang/MPCache.git /root/MPCache
bash /root/MPCache/h20/bootstrap_h20.sh /root/viewdit
```

The script installs miniconda if absent, accepts the conda channel TOS, creates the `racfg`
env (python 3.10), installs the pinned packages **in the only order that works**, pulls the
weight sets through the mirror with retries, and ends with real-content validation rather
than "it ran". It is idempotent: a reclaimed machine is rebuilt by re-running it.

## What a fresh allocation actually misses (2026-10-27)

A new H20 container starts as bare `/root`: **no miniconda, no weights, no
`/root/viewdit`, no `/root/RA-CFGCache`, no reference videos**. Budget ~1 h before the
first GPU-second of real work:

| Item | Cost | Note |
|---|---|---|
| miniconda + `racfg` env | ~10 min | fresh install refuses env creation until `conda tos accept` |
| torch 2.5.1+cu121 + diffusers stack | ~15 min | see P13 for why not cu124/cu128 |
| Wan 27 GB + Cog 13 GB via hf-mirror | ~40 min | ~10 MB/s per stream, parallel helps a little |
| RA Full reference videos | 2.2 h on 4 cards | **not in git**, see P21 |

## Pinned environment (H20, driver 535)

| Package | Version | Why |
|---|---|---|
| python | 3.10 | reference env |
| torch | **2.5.1+cu121** | driver 535 rejects cu124/cu128 wheels (P13) |
| torchvision | 0.20.1 | matches torch |
| diffusers | 0.37.0 | Wan/Cog/Hunyuan pipelines used by both runners |
| transformers | 4.57.1 | 5.2.0 needs the newer torch; 4.57.1 is what ran |
| numpy | 1.26.4 (`<2`) | lpips / older skimage code paths |
| nvidia-cublas-cu12 | **12.4.5.8, installed LAST** | P1 + P18 |

`HunyuanVideoPipeline` is available from diffusers 0.37; no extra model code is needed.
CogVideoX weights moved to `zai-org/CogVideoX-2b`; `THUDM/CogVideoX-2b` still resolves
(the script tries both, zai-org first).

## Weights (approximate size)

| Model | Size | Source |
|---|---|---|
| CogVideoX-2b | 13 GB | `zai-org/CogVideoX-2b` (fallback `THUDM/CogVideoX-2b`) |
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
   V100. Refetch it at the pinned commit:
   `git clone https://github.com/yiming-l21/RA-CFGCache && git checkout e17d6a7`.
8. **cuBLAS is silently downgraded by any later pip** → four SIGFPE (-8) processes and no
   traceback (P18). It must be the last install; verify with `pip list | grep cublas`.
9. **`--frames` defaults to 33.** A command without `--frames 81` produces a 33-frame video
   that still evaluates, just not against the 81-frame reference (P19).
10. **A flat-colour video is the failure signature of the text-embedding bug** (P16). `rel`
    looks healthy while the pixels are a blue gradient, so never trust `rel` as a liveness
    check — decode a frame and look at `mean/std`.
11. **Disk and VRAM checks before a long queue:** `df -h /root`, `nvidia-smi`, and one
    smoke sample per model.

## After deploy: smoke checklist

1. `pip list | grep cublas` → `12.4.5.8`; the three SIGFPE shapes multiply without exit -8.
2. One Wan sample at `--frames 33 --Ns 50` writes a video whose **pixel std > 40** (a flat
   frame means P16/P17 regressed). The bootstrap script does this and fails loudly.
3. `rope`/`rel` sanity: `sanity roundtrip 0.0`, one CFG step ~1.7 s at 33 frames / ~6.4 s at
   81 frames on H20.
4. Peak VRAM for the model you are about to queue is below the card size
   (`torch.cuda.max_memory_allocated()` is printed by the controller).
5. Output JSONL grows (`wc -l results/video/<run>/closed.jsonl`).
6. Kill test: `kill <pid>` stops the process and leaves the JSONL intact (the controller is
   append-only).

## Standing queues

| Queue | Hardware | Command |
|---|---|---|
| Wan main table: ex1 → tea_gap → our full → eval | H20 4× | `bash /root/viewdit/queue_wan.sh` |
| RA Full reference (only if the mp4s were not restored) | H20 4× | `bash /root/viewdit/run_ra_full.sh` |
| Cog development, 16 samples, N=10/13/17 | V100 | `--frames 17 --nprompt 8 --seeds 3000,3001 --Ns 10,13,17 --planners reuse,tea_mp,tea_gap` |
| Cog production, 49 frames, 100 prompts | H20 4× | needs `--save-video/--no-full/--negative/--video-dir` ported into `mpcache_controller.py` first (TODO 3.2) |

Both queue scripts are `setsid nohup`-safe and chain stages on process exit, so the cards do
not idle between arms.
