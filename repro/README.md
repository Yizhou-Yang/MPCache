# Competitor reproduction: TeaCache / TaylorSeer / RA-CFGCache

Goal: reproduce **RA-CFGCache Table 1** (NeurIPS 2026, arXiv 2609.36433) on our single V100-32GB,
so that our method (MPCache) is compared to the three competitors under exactly their protocol.

Why one harness for all three: RA-CFGCache's Table 1 is the only recent, top-venue table that
reports TeaCache and TaylorSeer on CogVideoX-2B / Wan2.1-T2V-1.3B with code. Its repo ships the
TeaCache / Taylor(Seer) re-implementations used for that table, so reproducing inside it gives us
three competitors with one protocol. Official TeaCache / TaylorSeer repos are kept as submodules
for cross-checks only.

## Upstream (git submodules, pinned)

| path | repo | commit | license |
|---|---|---|---|
| `upstream/RA-CFGCache` | yiming-l21/RA-CFGCache | `e17d6a7` | GPL-3.0 |
| `upstream/TeaCache` | ali-vilab/TeaCache | `7c10efc` | Apache-2.0 |
| `upstream/TaylorSeer` | Shenyi-Z/TaylorSeer | `704ee98` | see repo |

Our only change to upstream: `patches/apply_v100_patch.py` replaces hard-coded `torch.bfloat16`
with `RACFG_DTYPE` (default `bfloat16`, i.e. unchanged). V100 has no bf16, so we run fp16
(CogVideoX-2B is an fp16 checkpoint anyway). Schedules, thresholds, proxies, calibration tables
(`calibration/*_rho_cfg*.npz`) are untouched.

## Protocol (paper Sec. 4 + App. B.1)

| | CogVideoX-2B | Wan2.1-T2V-1.3B |
|---|---|---|
| video | 49f, 720×480, 8 fps | 81f, 832×480, 16 fps |
| sampler | 50 steps (pipeline default) | 50 steps UniPC, flow_shift 3.0 |
| CFG | 6.0, negative prompt "animation" | 5.0, negative prompt "animation" |
| prompts | 100 from VBench `all_dimension.txt` (`resources/prompts/prompt_video.txt`) | same |
| seed | 0 + global prompt index | same |
| TeaCache | τ = 0.2 | τ = 0.15 |
| TaylorSeer | N = 3, O = 1, first_enhance 3 | same |
| RA-CFGCache | Slow τ = 0.6 / Fast τ = 1.4 (TeaCache-style proxy) | Slow τ = 0.4 / Fast τ = 0.6 |
| metrics | LPIPS(alex) / SSIM / PSNR vs Vanilla T=50 same prompt+seed; latency, speedup | same |

## Targets (RA-CFGCache Table 1, H200 bf16)

CogVideoX-2B: Vanilla25 0.4751/0.5445/13.61 @1.86× · TeaCache 0.2040/0.7538/21.48 @2.42× ·
RA-Slow 0.0968/0.8746/26.93 @2.39× · RA-Fast 0.2052/0.7614/21.89 @3.13× (TaylorSeer not reported).

Wan2.1-1.3B: Vanilla25 0.4371/0.5658/15.87 @1.93× · TeaCache 0.1661/0.7955/23.08 @2.44× ·
TaylorSeer 0.3632/0.6187/16.96 @2.13× · RA-Slow 0.0993/0.8778/28.03 @2.56× · RA-Fast 0.1409/0.8386/26.12 @2.99×.

Expected deviations: absolute latency (V100 vs H200); small fidelity shifts from fp16 vs bf16.
Speedup and the ranking of methods should match.

## Usage (server)

```bash
bash repro/setup_server.sh                          # conda env `racfg`, upstream, Wan weights
bash repro/run_repro.sh cogvideox 0 1 vanilla50 teacache   # smoke test
bash repro/run_repro.sh cogvideox 0 32              # all 6 rows on prompts [0,32)
bash repro/run_repro.sh cogvideox 32 100            # rest; chunks keep global seeds
conda activate racfg && python repro/aggregate.py --model cogvideox --n 32
```

Outputs: `results/repro/<model>/<method>/p<start>-<end>/videos/video_<idx>.mp4`,
logs in `results/repro/<model>/_logs/`, tables in `results/repro/<model>/_merged/table_n*.md`.

## Status

See `STATUS.md`.
