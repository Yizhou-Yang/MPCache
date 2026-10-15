# Reproduction status

| date | step | state |
|---|---|---|
| 2026-10-15 | harness, V100 dtype patch, run/aggregate scripts, pinned submodules | done |
| 2026-10-15 | server env `racfg` (diffusers 0.37 / torch 2.10 / transformers 5.2) + Wan2.1-1.3B weights | in progress |
| 2026-10-15 | smoke test (1 prompt): CogVideoX vanilla/TeaCache/TaylorSeer/RA-Slow, Wan vanilla/TeaCache | queued |
| – | CogVideoX-2B prompts [0,32) all 6 rows | pending (after G0a) |
| – | Wan2.1-1.3B prompts [0,32) all 6 rows | pending |
| – | prompts [32,100) both models | pending |

Results tables will be appended here as `## <model>, n=<N>`.
