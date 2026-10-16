# MPCache reproduction checkpoint

Generated: 2026-10-16T16:16:59.981843

GPU-only generation; V100 fp16 differs from paper H200 bf16.
Legacy RA Slow/Fast outputs used tau=0.4 and are excluded; cfg_tau_v2 reruns required.
Video counts are unique global IDs, not overlapping chunk totals.
Timing uses chunk averages on a shared GPU and remains provisional.

## cogvideox

- vanilla50: 16 valid unique videos
- vanilla25: 16 valid unique videos
- teacache: 16 valid unique videos
- taylorseer: 1 valid unique videos
- racfg_slow: 16 valid unique videos
- racfg_fast: 16 valid unique videos

## cogvideox: V100 fp16 vs RA-CFGCache Table 1 (H200 bf16), requested n=16

| Method | videos | 质量 pair n | timing pairs | 状态 | LPIPS↓ ours / paper | SSIM↑ ours / paper | PSNR↑ ours / paper | Speedup ours / paper | Latency(s) ours / paper |
|---|---|---|---|---|---|---|---|---|---|
| Vanilla (T=50) | 16 | 16 | 16 | reference-self | 0.0000 / 0.0000 | 1.0000 / 1.0000 | ∞ / ∞ | 1.00× / 1.00× | 443.2 / 40.48 |
| Vanilla (T=25) | 16 | 16 | 16 | evaluated | 0.4585 / 0.4751 | 0.5437 / 0.5445 | 12.97 / 13.61 | 2.70× / 1.86× | 164.2 / 21.79 |
| TeaCache | 16 | 16 | 16 | evaluated | 0.1962 / 0.2040 | 0.7414 / 0.7538 | 20.08 / 21.48 | 3.83× / 2.42× | 115.8 / 16.72 |
| TaylorSeer (N=3,O=1) | 1 | 1 | 0 | evaluated | 0.5934 / – | 0.2304 / – | 11.82 / – | –× / –× | – / – |
| RA-CFGCache-Slow | 16 | 16 | 16 | evaluated | 0.0819 / 0.0968 | 0.8829 / 0.8746 | 26.57 / 26.93 | 4.03× / 2.39× | 110.0 / 16.96 |
| RA-CFGCache-Fast | 16 | 16 | 16 | evaluated | 0.1884 / 0.2052 | 0.7507 / 0.7614 | 20.86 / 21.89 | 5.58× / 3.13× | 79.5 / 12.91 |

- V100 fp16 vs H200 bf16：仅作论文对照，不是原数复现。
- RA 两行仅纳入 cfg_tau_v2 输出：旧 CLI 阈值未绑定运行时、实际均为 tau=0.4，旧结果无效。
- TaylorSeer 仅纳入 gpu_only_v2 输出；旧 Cog n=16 使用 CPU offload，排除正式全 GPU 对标。
- Wan 使用 GPU VAE tiling/slicing，未证明与论文未分块解码等价。
- Latency/speedup 使用所选 source 的 chunk-average，共同 timing IDs 上加权；并非精确 per-prompt 计时，shared-GPU latency 为 provisional。
- 未执行统计门检验，不能宣称统计门通过；reference-self 为恒等参照，未调用质量评估器。
- 质量 pair n 为实际评估数（reference-self 为恒等配对数）；pending/not-evaluated 的指标缺失，不填 0。
- source 审计及配对指纹：n16/<method>/merge_manifest.json、pair_manifest.json。


## wan

- vanilla50: 16 valid unique videos
- vanilla25: 16 valid unique videos
- teacache: 16 valid unique videos
- taylorseer: 0 valid unique videos
- racfg_slow: 16 valid unique videos
- racfg_fast: 16 valid unique videos

## wan: V100 fp16 vs RA-CFGCache Table 1 (H200 bf16), requested n=16

| Method | videos | 质量 pair n | timing pairs | 状态 | LPIPS↓ ours / paper | SSIM↑ ours / paper | PSNR↑ ours / paper | Speedup ours / paper | Latency(s) ours / paper |
|---|---|---|---|---|---|---|---|---|---|
| Vanilla (T=50) | 16 | 16 | 16 | reference-self | 0.0000 / 0.0000 | 1.0000 / 1.0000 | ∞ / ∞ | 1.00× / 1.00× | 878.6 / 93.32 |
| Vanilla (T=25) | 16 | 16 | 16 | evaluated | 0.4397 / 0.4371 | 0.6001 / 0.5658 | 16.10 / 15.87 | 1.90× / 1.93× | 461.8 / 48.41 |
| TeaCache | 16 | 16 | 16 | evaluated | 0.1203 / 0.1661 | 0.8675 / 0.7955 | 24.98 / 23.08 | 2.40× / 2.44× | 366.1 / 38.24 |
| TaylorSeer (N=3,O=1) | 0 | 0 | 0 | pending | – / 0.3632 | – / 0.6187 | – / 16.96 | –× / 2.13× | – / 43.79 |
| RA-CFGCache-Slow | 16 | 16 | 16 | evaluated | 0.0805 / 0.0993 | 0.8960 / 0.8778 | 28.82 / 28.03 | 2.54× / 2.56× | 346.1 / 36.48 |
| RA-CFGCache-Fast | 16 | 16 | 16 | evaluated | 0.1177 / 0.1409 | 0.8584 / 0.8386 | 26.73 / 26.12 | 2.97× / 2.99× | 296.2 / 31.25 |

- V100 fp16 vs H200 bf16：仅作论文对照，不是原数复现。
- RA 两行仅纳入 cfg_tau_v2 输出：旧 CLI 阈值未绑定运行时、实际均为 tau=0.4，旧结果无效。
- TaylorSeer 仅纳入 gpu_only_v2 输出；旧 Cog n=16 使用 CPU offload，排除正式全 GPU 对标。
- Wan 使用 GPU VAE tiling/slicing，未证明与论文未分块解码等价。
- Latency/speedup 使用所选 source 的 chunk-average，共同 timing IDs 上加权；并非精确 per-prompt 计时，shared-GPU latency 为 provisional。
- 未执行统计门检验，不能宣称统计门通过；reference-self 为恒等参照，未调用质量评估器。
- 质量 pair n 为实际评估数（reference-self 为恒等配对数）；pending/not-evaluated 的指标缺失，不填 0。
- source 审计及配对指纹：n16/<method>/merge_manifest.json、pair_manifest.json。
- TaylorSeer (N=3,O=1)：没有共同视频 ID


