## wan: h20 bfloat16 vs RA-CFGCache Table 1 (H200 bf16), requested n=100

| Method | videos | 质量 pair n | timing pairs | 状态 | LPIPS↓ ours / paper | SSIM↑ ours / paper | PSNR↑ ours / paper | Speedup ours / paper | Latency(s) ours / paper |
|---|---|---|---|---|---|---|---|---|---|
| Vanilla (T=50) | 100 | 100 | 100 | reference-self | 0.0000 / 0.0000 | 1.0000 / 1.0000 | ∞ / ∞ | 1.00× / 1.00× | 323.4 / 93.32 |
| Vanilla (T=25) | 100 | 100 | 100 | evaluated | 0.4204 / 0.4371 | 0.5700 / 0.5658 | 15.85 / 15.87 | 1.96× / 1.93× | 165.3 / 48.41 |
| TeaCache | 100 | 100 | 100 | evaluated | 0.1681 / 0.1661 | 0.7887 / 0.7955 | 22.84 / 23.08 | 2.53× / 2.44× | 128.1 / 38.24 |
| TaylorSeer (N=3,O=1) | 100 | 100 | 100 | evaluated | 0.3496 / 0.3632 | 0.6283 / 0.6187 | 17.10 / 16.96 | 2.47× / 2.13× | 130.9 / 43.79 |
| RA-CFGCache-Slow | 100 | 100 | 100 | evaluated | 0.0941 / 0.0993 | 0.8820 / 0.8778 | 28.11 / 28.03 | 2.65× / 2.56× | 122.1 / 36.48 |
| RA-CFGCache-Fast | 100 | 100 | 100 | evaluated | 0.1352 / 0.1409 | 0.8420 / 0.8386 | 26.14 / 26.12 | 3.14× / 2.99× | 102.9 / 31.25 |

- H20 bf16 与论文精度一致，但硬件为 H20 而非 H200：质量指标可与论文对照，latency/speedup 不可当作论文原数。
- RA 两行仅纳入 cfg_tau_v2 输出：旧 CLI 阈值未绑定运行时、实际均为 tau=0.4，旧结果无效。
- TaylorSeer 仅纳入 gpu_only_v2 输出；旧 Cog n=16 使用 CPU offload，排除正式全 GPU 对标。
- h20 profile 未启用任何显存外挂：VAE 未分块、文本编码器全程常驻，与 upstream 一致。
- Latency/speedup 使用所选 source 的 chunk-average，共同 timing IDs 上加权；并非精确 per-prompt 计时，shared-GPU latency 为 provisional。
- 未执行统计门检验，不能宣称统计门通过；reference-self 为恒等参照，未调用质量评估器。
- 质量 pair n 为实际评估数（reference-self 为恒等配对数）；pending/not-evaluated 的指标缺失，不填 0；partial-n 表示该行配对样本少于表内 n，不构成完整行；partial-timing 表示质量配对完整但共同有效计时不足 n，同样返回失败。
- source 审计及配对指纹：n100/<method>/merge_manifest.json、pair_manifest.json。
