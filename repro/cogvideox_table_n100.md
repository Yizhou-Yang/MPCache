## cogvideox: v100 float16 vs RA-CFGCache Table 1 (H200 bf16), requested n=100

| Method | videos | 质量 pair n | timing pairs | 状态 | LPIPS↓ ours / paper | SSIM↑ ours / paper | PSNR↑ ours / paper | Speedup ours / paper | Latency(s) ours / paper |
|---|---|---|---|---|---|---|---|---|---|
| Vanilla (T=50) | 100 | 100 | 100 | reference-self | 0.0000 / 0.0000 | 1.0000 / 1.0000 | ∞ / ∞ | 1.00× / 1.00× | 399.7 / 40.48 |
| Vanilla (T=25) | 100 | 100 | 100 | evaluated | 0.4612 / 0.4751 | 0.5446 / 0.5445 | 13.97 / 13.61 | 2.43× / 1.86× | 164.3 / 21.79 |
| TeaCache | 100 | 100 | 100 | evaluated | 0.2015 / 0.2040 | 0.7503 / 0.7538 | 21.21 / 21.48 | 3.45× / 2.42× | 115.8 / 16.72 |
| TaylorSeer (N=3,O=1) | 100 | 100 | 100 | evaluated | 0.4044 / – | 0.5607 / – | 15.13 / – | 2.78× / –× | 143.7 / – |
| RA-CFGCache-Slow | 100 | 100 | 100 | evaluated | 0.1031 / 0.0968 | 0.8589 / 0.8746 | 26.15 / 26.93 | 3.63× / 2.39× | 110.1 / 16.96 |
| RA-CFGCache-Fast | 100 | 100 | 100 | evaluated | 0.2094 / 0.2052 | 0.7477 / 0.7614 | 21.44 / 21.89 | 5.03× / 3.13× | 79.4 / 12.91 |

- V100 float16 vs H200 bf16：仅作论文对照，不是原数复现。
- RA 两行仅纳入 cfg_tau_v2 输出：旧 CLI 阈值未绑定运行时、实际均为 tau=0.4，旧结果无效。
- TaylorSeer 仅纳入 gpu_only_v2 输出；旧 Cog n=16 使用 CPU offload，排除正式全 GPU 对标。
- Wan 使用 GPU VAE tiling/slicing，未证明与论文未分块解码等价。
- TaylorSeer 行在编码后把空闲的文本编码器停到 CPU（非 CPU offload；去噪与缓存全程在 GPU）；h20 profile 无此外挂。
- Latency/speedup 使用所选 source 的 chunk-average，共同 timing IDs 上加权；并非精确 per-prompt 计时，shared-GPU latency 为 provisional。
- 未执行统计门检验，不能宣称统计门通过；reference-self 为恒等参照，未调用质量评估器。
- 质量 pair n 为实际评估数（reference-self 为恒等配对数）；pending/not-evaluated 的指标缺失，不填 0；partial-n 表示该行配对样本少于表内 n，不构成完整行；partial-timing 表示质量配对完整但共同有效计时不足 n，同样返回失败。
- source 审计及配对指纹：n100/<method>/merge_manifest.json、pair_manifest.json。
