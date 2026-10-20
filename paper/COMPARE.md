# 论文对比与精修工作区

对标对象：**Consistency Models**（Song, Dhariwal, Chen, Sutskever；ICML 2023；arXiv 2303.01469）。
选它的理由：同属扩散加速方向、同用 ICML 模板、版式被广泛参照；缓存线的竞品（TeaCache / TaylorSeer / RA-CFGCache）都不在 ICML，格式上没有可比性。

## 目录

```
对比/
├── ConsistencyModels_ICML2023_2303.01469.pdf   # 基准
├── MPCache_main.pdf                            # 我们当前的论文（最新构建）
├── versions/                                   # 每个版本的快照
│   └── MPCache_v1_20261020.pdf
├── src -> ~/Desktop/DPCache/paper              # LaTeX 源码（软链）
└── COMPARE.md                                  # 本文件：对比表 + 版本日志
```

## 工作流（后续所有对比和精修都在这里）

1. 改源码：`cd ~/Desktop/对比/src && $EDITOR sections/*.tex`
2. 编译：`make`（tectonic，产物在 `build/main.pdf`）
3. 快照：`cp build/main.pdf ~/Desktop/对比/MPCache_main.pdf`
   `cp build/main.pdf ~/Desktop/对比/versions/MPCache_vN_YYYYMMDD.pdf`
4. 在本文件记录：改了什么、为什么、页数/版面变化。

## 版面对比（2026-10-20，v1）

| 维度 | Consistency Models | MPCache（v1） | 状态 |
|---|---|---|---|
| 表标题位置 | 表格上方 | 表格上方 | ✅ 已对齐 |
| 图注长度 | 2–3 行 | 2 行 | ✅ |
| 内部备忘入正文 | 无 | 原有一行 “Draft status”，已删 | ✅ |
| 章节命名 | Intro / Background / Method / Experiments / Related / Discussion | Intro / Related / Method / Experiments | ⚠️ Related 前置，属叙事选择，暂不改 |
| 局限性 | 单列一节 | 实验节末 `Limitations.` 段 | ✅（形式略轻） |
| 图的数量 | 5–6 张（含首页 teaser） | 1 张（方法循环图） | ⚠️ 待加 teaser |
| 摘要长度 | ~150 词 | ~170 词（v3 精简） | ✅ |
| 参考文献 | icml2026.bst | icml2026.bst | ✅ 同模板 |
| 页数 | 9 页正文 + 附录 | 8 页（含参考文献） | ✅ 符合 ICML 8 页正文约束 |
| Overfull box | — | 0 | ✅ |

## 版本日志

**v13 — 2026-10-20**
- 摘要结论句改为目标主张（RA 式）："Experiments on CogVideoX-2B and Wan2.1-T2V-1.3B demonstrate improved efficiency--fidelity trade-offs over TeaCache, TaylorSeer, and RA-CFGCache at matched exact-evaluation budgets."
- 写作流程修正：草稿里先写目标、后补实验是正常的；护栏从正文移到 **源码注释**（不进 PDF）：CLAIM PENDING + 提交前必须过的门（中位 ≥25%、胜率 ≥0.75、>2SEM、双 backbone）+ 双盲匿名提醒。
- 页数仍 8，0 overfull。


**v12 — 2026-10-20**
- 摘要末尾补“实验对象 + 竞品 + 代码”三要素，对标 RA-CFGCache 摘要结尾的写法：模型 CogVideoX-2B / Wan2.1-T2V-1.3B，竞品 TeaCache / TaylorSeer / RA-CFGCache，100 条 VBench prompt，matched exact-evaluation budget。
- **没有写 "demonstrate improved"**：目前控制器还没有 matched-budget 结果（G0a/G0b 均 FAIL），写成“已验证”是造假。现在的写法是 "the baselines are reproduced here, and the controller's matched-budget results are pending in this draft"，并留了 FINAL FORM 的 LaTeX 注释，一条命令即可替换。
- 代码地址 `https://github.com/Yizhou-Yang/MPCache` 已加；**投稿前必须换成匿名镜像**（ICML 双盲），注释里已标。
- 页数仍 8，0 overfull。


**v11 — 2026-10-20**
- 摘要方法段改为 RA-CFGCache 式的“组件命名 + 一句话职责”：`State-Conditioned Forecasting` / `Solver-Aware Pricing` / `Residual Check` / receding-horizon planner，各一句，取代原来偏实现的描述。
- 同步把 3.3–3.5 的节标题改成与摘要一致：State-Conditioned Forecasting、Solver-Aware Pricing、Budgeted Receding-Horizon Planning，避免“摘要叫一个名字、正文叫另一个”。
- 页数仍 8，0 overfull。

**v10 — 2026-10-20**
- 摘要过渡句按“后果句”重写：`a cheap-looking skip can still return a value whose error later steps amplify`，并加入影响句 `A skip now carries an explicit cost before it is taken rather than being discovered after`。


**v8 — 2026-10-20**
- 摘要过渡句按用户指定方向重写：删去 "The two are not independent. A skip can be priced well yet forecast poorly, and the forecast moves the trajectory…"（29 词的解释性句子），改为直接给出来源与方法定位："Inspired by model-predictive control, we propose MPCache, which uses one small local model of the denoising field to unify the two."
- 好处：节奏从"解释问题"切到"提出方法"，读者更早看到 MPC 这一差异化定位；"unify the two" 仍隐含耦合，但不再展开论证（耦合论证移到 Introduction 第二段）。
- 摘要 ~140 词；页数仍 8，0 overfull。


**v7 — 2026-10-20**
- 摘要第二句瘦身：原句 35 词、带三项破折号列表（"when to skip, what to return, and whether the skip was worth it"），且与下一句对不上（下一句只谈 price/forecast 两件事）。
- 改为 24 词、只讲两个动作："…but decides when to skip and what to return with separate rules."；随后加一句 5 词的桥："The two are not independent."，再落到后果，衔接从"并列"变成"因果链"。
- 摘要总计 ~150 词；页数仍 8，0 overfull。


**v6 — 2026-10-20**
- 摘要从 190 词压到 ~150 词；首句改为 "every step runs the denoiser twice"（去掉 "of the fifty" 的冗词）。
- 可读性：拆掉原来的嵌套长句（"because … and …"），改为一句一个意思；平均句长从 ~27 词降到 ~19 词；"three pieces … specified independently" 改为更具体的 "three decisions—when to skip, what to return, and whether the skip was worth it—are made by separate heuristics"。
- 结尾保留 "online identification becomes a testable requirement rather than an assumption"，作为全文的诚实姿态。
- 页数仍 8，0 overfull。


**v5 — 2026-10-20**
- 摘要开头改为四拍结构，对标 RA-CFGCache 摘要的写法：① 领域与成本（50 步 × CFG 双分支）→ ② 现有解法（training-free caching 复用缓存/预测）→ ③ 缺口（三个部件独立指定：proxy 决定何时跳、forecaster 决定跳步返回什么、propagation prior 决定值不值）→ ④ 后果（可能“定价对、预测差”，且预测改变轨迹）→ ⑤ We propose。
- v4 的问题是从“现有方法”直接起手，缺了背景与成本两拍，读起来像半路开始。
- 页数仍 8，0 overfull。


**v4 — 2026-10-20**
- 摘要开头重写：v3 的 "from local change alone, without a model of how the denoiser responds" 被指出两个问题——术语未落地、且暗含“无人建模响应”的过度主张（RA-CFGCache 有传播模型、RACER 有闭环、ERTACache 有误差累积）。
- 新写法把 proxy 具体化为三类（input change / temporal derivatives / calibrated propagation prior），把缺口收窄为结构性事实：三者被独立指定，因此可能“定价对但预测差”，且预测会改变被定价的误差；最后以一个研究问题而非否定式断言收束。
- 页数仍 8，0 overfull。


**v3 — 2026-10-20**
- 摘要首句重写：原来是 60+ 词的双分句长句（"…by replacing selected denoiser evaluations…but deciding what to predict and when to refresh are tightly coupled"），读者读不到“问题是什么”。现在拆成两句、共 40 词：第一句直接点缺口（只凭局部变化预测跳步、定价，没有“去噪器如何响应”的模型），第二句才说两个决策为何耦合。
- 原第二句（时间外推 vs 传播感知控制器）冗长，其信息已被新首句覆盖，删除。
- 页数仍 8，0 overfull。


**v2 — 2026-10-20**
- 重写 `Limitations.`：第一句直接写 **“MPCache has no validated result.”**，不再用 “specified but not yet validated” 这种长句；第二句给硬数字（G0a Spearman 0.33 vs 先验 0.95；G0b N=13 +9.1%、N=10 −22%）；随后才是 H20≠H200、白帧 prompt、基线预算不统一三条次要限制。
- 页数仍 8，0 overfull。


**v1 — 2026-10-20**
- 表标题从表格下方移到上方；改的是生成器 `tools/make_baseline_table.py`，重跑仍一致。
- 删除首页 “Draft status” 内部备忘（诚实性说明已在 Introduction 的 “design contributions, not claims” 中）。
- 章节 `Experimental Protocol` → `Experiments`。
- 新增 `Limitations.` 段（控制器未验证、H20≠H200、白帧 prompt、基线预算不统一）。
- 图 1 图注由 4 行压到 2 行。
- 结果：8 页，0 overfull，无缺引用。

## 待办

- [ ] **首页 teaser**：左panel 时间外推（历史点 + 延伸虚线）、右panel 状态修正（偏离预测路径 + Ĵ 拉回），配 “same budget, different question”。加上后预计仍 8 页。
- [ ] 控制器 pilot 出结果后，补方法对比表（matched NFE，vs reuse/ex1/TeaCache）。
- [ ] 视篇幅决定是否把 Related Work 移到 Experiments 之后（对齐 CM 的叙事顺序）。
