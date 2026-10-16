# MPCache reproduction checkpoint

Generated: 2026-10-16T15:49:41.340782

GPU-only generation; V100 fp16 differs from paper H200 bf16.
Legacy RA Slow/Fast outputs used tau=0.4 and are excluded; cfg_tau_v2 reruns required.
Video counts are unique global IDs, not overlapping chunk totals.
Timing uses chunk averages on a shared GPU and remains provisional.

## cogvideox

- vanilla50: 16 valid unique videos
- vanilla25: 16 valid unique videos
- teacache: 16 valid unique videos
- taylorseer: 0 valid unique videos
- racfg_slow: 16 valid unique videos
- racfg_fast: 16 valid unique videos

# CogVideoX n=16：协议审计后的当前结果

旧 TaylorSeer offload 行已排除；其他指标沿用 13:01 完成的配对评测，未声称重新评测。

| Method | GPU-valid n | LPIPS | SSIM | PSNR | provisional speedup | status |
|---|---|---|---|---|---|---|
| vanilla50 | 16 | 0.0 | 1.0 | Infinity | 1.0 | reference-self |
| vanilla25 | 16 | 0.4584805123622016 | 0.5437056039035233 | 12.97303591953715 | 2.699005535628376 | evaluated |
| teacache | 16 | 0.19620437793224593 | 0.7413866911851392 | 20.082814917936215 | 3.8278590121174974 | evaluated |
| taylorseer | 0 | – | – | – | – | excluded-offload |
| racfg_slow | 16 | 0.08190712441538214 | 0.8828604822441619 | 26.5715724205408 | 4.027635405307162 | evaluated |
| racfg_fast | 16 | 0.18841392894712639 | 0.7506675884417184 | 20.858118053494888 | 5.5761880197783125 | evaluated |

- V100 fp16 vs H200 bf16：仅作论文对照，不是原数复现。
- RA 两行仅纳入 cfg_tau_v2 输出：旧 CLI 阈值未绑定运行时、实际均为 tau=0.4，旧结果无效。
- Wan 使用 GPU VAE tiling/slicing，未证明与论文未分块解码等价。
- Latency/speedup 使用所选 source 的 chunk-average，共同 timing IDs 上加权；并非精确 per-prompt 计时，shared-GPU latency 为 provisional。
- 未执行统计门检验，不能宣称统计门通过；reference-self 为恒等参照，未调用质量评估器。
- 质量 pair n 为实际评估数（reference-self 为恒等配对数）；pending/not-evaluated 的指标缺失，不填 0。
- source 审计及配对指纹：n16/<method>/merge_manifest.json、pair_manifest.json。
- TaylorSeer 仅纳入 gpu_only_v2 输出；旧 Cog n=16 使用 CPU offload，排除正式全 GPU 对标。


## wan

- vanilla50: 16 valid unique videos
- vanilla25: 16 valid unique videos
- teacache: 16 valid unique videos
- taylorseer: 0 valid unique videos
- racfg_slow: 16 valid unique videos
- racfg_fast: 14 valid unique videos
