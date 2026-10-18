## cogvideox: h20 bfloat16 vs RA-CFGCache Table 1 (H200 bf16), requested n=100

| Method | videos | 质量 pair n | timing pairs | 状态 | LPIPS↓ ours / paper | SSIM↑ ours / paper | PSNR↑ ours / paper | Speedup ours / paper | Latency(s) ours / paper |
|---|---|---|---|---|---|---|---|---|---|
| Vanilla (T=50) | 100 | 100 | 100 | reference-self | 0.0000 / 0.0000 | 1.0000 / 1.0000 | ∞ / ∞ | 1.00× / 1.00× | 138.8 / 40.48 |
| Vanilla (T=25) | 100 | 100 | 100 | evaluated | 0.4704 / 0.4751 | 0.5350 / 0.5445 | 14.08 / 13.61 | 1.89× / 1.86× | 73.2 / 21.79 |
| TeaCache | 100 | 100 | 100 | evaluated | 0.2122 / 0.2040 | 0.7392 / 0.7538 | 21.22 / 21.48 | 2.64× / 2.42× | 52.6 / 16.72 |
| TaylorSeer (N=3,O=1) | 100 | 100 | 100 | evaluated | 0.4047 / – | 0.5626 / – | 15.38 / – | 2.36× / –× | 58.8 / – |
| RA-CFGCache-Slow | 100 | 100 | 100 | evaluated | 0.0990 / 0.0968 | 0.8661 / 0.8746 | 26.89 / 26.93 | 2.63× / 2.39× | 52.8 / 16.96 |
| RA-CFGCache-Fast | 100 | 100 | 100 | evaluated | 0.2092 / 0.2052 | 0.7493 / 0.7614 | 21.88 / 21.89 | 3.74× / 3.13× | 37.1 / 12.91 |

- H20 bf16 与论文精度一致，但硬件为 H20 而非 H200：质量指标可与论文对照，latency/speedup 不可当作论文原数。
- RA 两行仅纳入 cfg_tau_v2 输出：旧 CLI 阈值未绑定运行时、实际均为 tau=0.4，旧结果无效。
- TaylorSeer 仅纳入 gpu_only_v2 输出；旧 Cog n=16 使用 CPU offload，排除正式全 GPU 对标。
- h20 profile 未启用任何显存外挂：VAE 未分块、文本编码器全程常驻，与 upstream 一致。
- Latency/speedup 使用所选 source 的 chunk-average，共同 timing IDs 上加权；并非精确 per-prompt 计时，shared-GPU latency 为 provisional。
- 未执行统计门检验，不能宣称统计门通过；reference-self 为恒等参照，未调用质量评估器。
- 质量 pair n 为实际评估数（reference-self 为恒等配对数）；pending/not-evaluated 的指标缺失，不填 0；partial-n 表示该行配对样本少于表内 n，不构成完整行；partial-timing 表示质量配对完整但共同有效计时不足 n，同样返回失败。
- source 审计及配对指纹：n100/<method>/merge_manifest.json、pair_manifest.json。
