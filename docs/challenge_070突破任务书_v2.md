# Challenge 0.70+：时序运动表示与迁移学习任务书 v2

> **状态：待人工审阅。本文的创建不授权训练、outer evaluation 或发布。**
> 审阅通过后，实施者按本文任务清单逐项执行；使用 `superpowers:executing-plans`，如另获授权拆分子任务，可使用 `superpowers:subagent-driven-development`。不得让技能默认流程越过本文的人工审阅门。

**Goal：** 在不破坏正式版本的前提下，寻找能够将相同官方事件匹配口径下的 development F1 从 `0.6514285714` 提升到 **`>= 0.7000`** 的 challenger；更高 F1 是后续目标，不承诺实验一定成功。

**Architecture：** 保留现有 macro/micro candidate generation 与 Context-v1 event-stack。增加一个覆盖完整候选区间、保留时间信息的轻量运动分支，比较 scratch 与 FD 预训练；第一版仅用一个有明确语义的分支概率接入 verifier，避免跨折潜在 embedding 坐标不一致和高维拼接。

**Tech Stack：** 现有 Python / NumPy / SciPy / PyTorch / scikit-learn / LightGBM 环境；优先复用 `D:/Anaconda3/envs/bme/python.exe`。不安装新框架、不升级依赖、不改生产架构。

**Spec 与依据：** 本文是独立的新任务书，基于人工认可的本轮分析、[上一版任务书](challenge下一阶段任务书.md)、[原始 challenge 规范](challenge设计任务书.md)及仓库证据。旧任务书保留为历史记录。本文获批后，在本轮 representation 研究范围内优先于旧版冲突条款；评估、交付和发布保护规则没有被解除。

---

## 0. 执行者先读：权限与交付边界

### 0.1 本轮现在只交付文档

本文件初始状态为 `DRAFT_REQUIRES_REVIEW`。不得因为看到文件而自动启动实验。后续人工明确批准后，默认授权范围为 **P0 → P1 → P2 → P3**，且每阶段必须满足进入条件。

P4 full outer 和 P5 promotion 是两个独立人工授权点。P3 即使 inner F1 达到 0.85，也不能自动进入 P4。P4 达到 0.70，也不能自动覆盖 release 或 dist。

### 0.2 Global Constraints

- 正式 incumbent：`160afaf81debf1ee`；正式 F1：`0.6514285714`；TP/GT/pred：`114/153/197`。
- 不改 `release/event_stack_incumbent.json`、`models/event_stack/160afaf81debf1ee/**`、`dist/**`、正式 submission 和生产推理接口。
- 不重算新的 sealed-outer 指标，不读取 outer GT 来选择模型、样本、阈值、checkpoint、采样规则或停止点。
- 不用 E01/E03.5/E04/Data Sprint/R1 的 outer-derived 行表生成训练、验证或负样本清单。
- 不改变官方 GT、eligible truth 定义、IoU `>= 0.25` 匹配器及其一对一匹配语义。
- 不对 fold、subject、meal 长度、场景设置专属 threshold 或专属模型。
- 不重新启动 Context-v2/v3、hard-negative 权重搜索、candidate-control 放宽、adaptive LR/LGBM fusion、GYRO5 标量扩展、PPG/HRV。
- 不做 Transformer/GRU/CNN/TCN 架构大比武，不同时更改 representation、标签策略和候选策略。
- 保持 main 上小步提交；只提交本任务文件，保留用户已有改动；不得 `git add .` 后盲目提交。
- 每个完整阶段更新研究摘要、状态与资源账本；正式 README 的发布指标不变。
- 使用子代理时限 terra/luna、中等思考强度；最多一个实施者加一个独立审查者。GPU 训练只由主执行者调度。不能因模型不可用而静默切换 Astra。

### 0.3 五个优先审查的失败模式

1. **形式 OOF、实际跨层泄漏：** 验证受试者不得出现在模型任何训练依赖祖先中，而不只是最终分类器。
2. **伪时序：** 输入覆盖完整 candidate；timestamp、顺序、间隔、missing mask 不得被索引插值或 top-k 池化悄悄抹掉。
3. **虚高 probe F1：** class balance、always-positive 基线和 candidate/event 指标必须分开报告。
4. **计算失控：** 拟合数量、步数、缓存和墙钟预算在训练前算清；不因 nested 拓扑昂贵而偷用 checkpoint。
5. **证据越权：** diagnostic、selection score、outer development、independent validation 四种证据严格分栏。

---

## 1. 当前证据与目标解释

### 1.1 正式基线

| 指标 | 当前值 |
|---|---:|
| TP | 114 |
| FP | 83 |
| FN | 39 |
| Precision | 0.578680 |
| Recall | 0.745098 |
| Event F1 | 0.651429 |

`F1 = 2 * TP / (153 + TP + FP)`。以下只是目标量级计算，不是预测、oracle 或注册实验结果：

| TP | FP | 相对 incumbent | F1 |
|---:|---:|---|---:|
| 114 | 58 | 不丢 TP，减少 25 FP | 0.701538 |
| 120 | 68 | 增加 6 TP，减少 15 FP | 0.703812 |
| 128 | 83 | 增加 14 TP，FP 不变 | 0.703297 |
| 120 | 47 | 增加 6 TP，减少 36 FP | 0.750000 |

优先追求“补回一部分真实事件，同时去掉一批不同性质的 FP”，而不是全局降低 threshold。

### 1.2 已有尝试应如何解释

| 证据 | 观测 | 本轮决策 |
|---|---|---|
| E02 Context-v2 | inner 0.76560 → 0.76567；少 13 FP、少 8 TP | 暂停这套信息的继续扩展 |
| E03 hard-negative | inner 0.76198，低于 control | 不重新搜索权重 |
| E03.5 | 取消 cap 救回 1 个目标 GT，新增 21 final FP | 不粗暴放宽 candidate control |
| E04 | reachable FN 31；near-boundary 3；Context 明显压制 0；LR/LGBM 分歧 15；未解决 14，类别非互斥 | 不预设 Context rescue 或 representation 已达上限 |
| E05 GYRO5 | inner 0.76510；净增 2 TP 和 4 FP | 当前标量追加方案失败，不等于短尺度无信息 |
| E06 adaptive fusion | inner 0.75671；跨层 OOF 拓扑仍须核查 | 不重启融合搜索；不宣称所有 fusion 均无效 |
| R1-003 | candidate F1 0.69375；scalar 0.76712；AUC 约 0.54 | 该 probe 失败，不足以排除时序表示或 FD |

结果来源位于 `outputs/challenger_70/` 下相应报告、`experiment_registry.jsonl` 及 `scripts/research/temporal_rep/run_r1.py`。P0 对引用值做一次结构化校验，不重跑实验；不能把实现中止编号计为额外算法负证据。

### 1.3 R1/E06 的证据整改，不改写历史

P0 输出 `evidence_review.json`，逐项写明 `confirmed / not_confirmed / insufficient_evidence`，附源码路径和依赖链。待核查事项：

- R1 的 population 来自 E04 outer-derived TP/FP/FN 表，不能作本轮合法 inner 数据入口。
- R1 的 top-k mean 丢弃 token 顺序；它仅测试局部 token 加无序聚合，而非完整事件序列。
- R1 中心最多 120 秒不等于覆盖整个 meal；按样本序号插值不等于 timestamp resampling。
- R1 每折 30 次全批量更新缺乏足够的收敛诊断，不能直接推断信息不存在。
- R1 的 145 positive / 83 negative 集合上，always-positive candidate F1 约 0.7775；与正式 event F1 无可比性。
- E06 的第二层 OOF 必须检查底层训练依赖，不能以“每行自己的底层模型排除了该 subject”代替完整嵌套隔离。
- E05 的不同 arm 重新选择 blend/policy，因此 churn 不是冻结 policy 下的纯 feature 因果归因。

旧结果、旧文件和旧状态保持原样。新 ledger 可以注明 `not_reusable_for_current_gate`，不得回填伪造为“当时已知”。本轮不重新审计整个生产 release，也不未经证据宣布 incumbent 作废。

---

## 2. 研究假设与明确不做的事

### 2.1 三个可分别失败的假设

- **H1 局部表示：** 5 秒 ACC/GYRO 片段中存在现有统计特征未保留的可迁移动作信息。
- **H2 时间组织：** 这些片段在整个候选中的顺序、间隔和分布能帮助区别 meal 与持续/碎片化非 meal 活动。
- **H3 互补性：** 时序分支能在现有 event-stack 之外提供增量事件判别能力。

H1/H2 的 probe 成功不等于 H3 成功。H3 失败不自动说明 raw IMU 没有信息。scratch 失败不自动禁止 FD；FD 失败也不妨碍有证据的 scratch 分支进入 P3。

### 2.2 本轮默认路线

`数据与依赖整改 → 完整时间轴输入 → 小型时序分支 + FD 对照 → 单个概率接入现有 verifier → 严格 inner gate → 人工 outer 审阅`。

内部 SSL、人工补标训练、PU learning、边界回归、多模型 ensemble、Transformer、额外身体部位传感器均不在默认授权范围。若本轮失败，只提交证据与下一份 amendment 建议，不自动执行这些路线。

---

## 3. 评估与数据防火墙

### 3.1 四种证据标签

| 类型 | 可用于什么 | 不可用于什么 |
|---|---|---|
| 历史 sealed diagnostic | 解释失败、形成宽泛 hypothesis | 样本筛选、训练、阈值选择、模型选择 |
| 新 inner probe | 在 outer-train 内选择表示配置 | 新 outer headline |
| 新 inner event selection score | 筛选一个值得人工审阅的 challenger | 宣称独立泛化或已达 0.70 |
| 另行批准的 outer development | 与现有开发基线对照 | 宣称未被研究过程使用过的独立测试 |

现有 outer 已参与多轮错误分析，必须继续称 development evidence。最终泛化结论以独立受试者或组委会隐藏测试为准。

### 3.2 合法 candidate 群体

每个 outer fold k：只读取 `outer_train_subjects[k]` 的 raw/session、标签和可证明合法的派生证据。保留现有 outer 分组身份，不重新找“更好”的 split。

每个 inner held-out 集合 V：

- 训练范围 T = outer-train \ V。
- V 的 macro/micro 概率、candidate 区间、116D 特征及运动分支预测，其 learned ancestors 全部只来自 T 或合法 FD 数据。
- candidate 按 incumbent 的确定性算法产生，禁止加入 GT 裁切的“完美正例”。
- 训练候选标签使用 incumbent 的既有 label helper 与 IoU 规则；训练多候选正例标签和最终一对一 event matcher 不混用。
- validation 上评价全体合法候选，不能只取错误 atlas 中的 TP/FP/FN。
- 候选训练集须包含正常正例、普通负例和自然产生的困难负例；不得以 sealed FP 名单定向抽样。
- 训练采样若受资源限制，采用按 subject→class→candidate 的固定随机种子采样；仅影响训练 minibatch，不删 validation rows，不按 held-out 分数采样。

### 3.3 两层学习器必须真正嵌套

P3 的一个 inner validation fold 必须遵守：

```text
outer train
├─ T：当前 inner training subjects
│  ├─ U0/U1/U2：T 内的 3 个 subject-disjoint 子折
│  │  ├─ 在 T\Uj 内训练底层模型及运动分支
│  │  └─ 仅预测 Uj，形成训练 verifier 所需 OOF 特征/分支概率
│  ├─ 拼接 T 的合法 OOF 116D + branch probability
│  ├─ 在这些行上训练 LR / constrained LGBM verifier
│  └─ 底层模型和运动分支在全部 T 上 refit，用于预测 V
└─ V：inner validation subjects
   └─ candidate、116D、branch probability、verifier score 均无 V 训练祖先
```

内层子折依然按 subject 划分；不得把 session、candidate、token 随机拆分代替 subject isolation。

训练 verifier 的底层 OOF 各行允许来自不同拟合模型，但**只合并有共同语义的 116D 特征与单一 branch probability**，不直接合并不同 encoder 的潜在坐标。不得把各折 16D embedding 当成天然对齐的同一坐标系。

同样，不能先在全部 outer-train 训练 encoder，再生成 inner OOF；也不能先生成 outer-train 全局底层 OOF，再不检查祖先关系就分割给新的 meta learner。

候选生成、base score、Context 特征中的所有 learned producer 也在上述防火墙内。缓存不满足就 cache miss；预算不够就报告阻塞，不能降级隔离标准。

### 3.4 Threshold 的含义写清

各 inner held-out prediction 必须 end-to-end 无本 subject 训练祖先。随后在该 outer-train 的合并 inner OOF 预测上选择 event threshold，得到的是 **inner selection score**，不是 threshold 独立验证分数。

不为每个 inner held-out subject 或 subgroup 单独选阈值。没有额外声明的 calibration holdout 时，不得把这个 gate 包装成 unbiased nested validation。

### 3.5 Provenance 最小字段

每个 artifact 记录：schema/code/config hash、source file fingerprint、train subject hash、score subject hash、parent artifact IDs、normalization fit subjects、external dataset/version/license、seed、训练步数、dtype、sensor units、preprocessing version。

除了哈希，应保留可审计的本地 subject 列表。哈希用于比对，不代替验证集合交集。为每个 score 递归收集 learned ancestors，断言与本次 held-out subject 集合无交集。

纯确定性、逐 session 的无标签原始切片可复用物理文件，但数据访问器必须仅暴露当前授权 subject。任何全数据统计归一化、PCA、SSL 都不是这种可无条件复用的缓存。

---

## 4. P0 — 证据整改、预算与预注册

**进入条件：** 人工批准本文件。**本阶段不训练模型。**

### 4.1 必做清单

- [ ] 核对 git status、当前 commit、正式指针及保护文件摘要；复用 `src/pipeline/challenger/protection.py`，不重复扫描整套 raw 数据。
- [ ] 创建本轮独立目录 `outputs/challenger_70/representation_v2/`，不覆盖旧 `temporal_rep/`。
- [ ] 核查第 1.3 节证据并写 `evidence_review.json` 与 `protocol.md`。
- [ ] 建立 source allowlist、subject manifest、父依赖图；确认现有 candidate/116D cache 哪些可安全复用。
- [ ] 盘点已有 FD 预处理及 checkpoint；读取其实际训练身份、输入定义、权重来源、许可。旧 checkpoint 无法证明 provenance 时不得复用。
- [ ] 固定新实验配置和计算上限，写 `plan.json`；只允许在任何结果产生前依据实际硬件做一次预算修订，并保留 diff。
- [ ] 建立 `budget.json`，计算每个 arm 的拟合次数，而不是只估算单模型时间。
- [ ] 文档中核对当前硬件、磁盘和 Python 依赖，不安装软件、不启动后台长期任务。

### 4.2 状态

- 全部通过：`P0_READY`。
- split、provenance 或许可无法确认：`BLOCKED_PROVENANCE_OR_LICENSE`。
- 预计成本超过第 9 节上限：`BUDGET_REVIEW_REQUIRED`。

后两种状态停止并报告，不自动改变架构、数据源或预算。

---

## 5. P1 — 完整事件时间轴与最小可行训练检查

### 5.1 输入合同（首版固定）

- 六通道：ACC xyz、GYRO xyz。保留原始单位，明确单位转换；只在训练分区拟合 robust channel scaling。
- 5 秒 token；stride 5 秒；25 Hz、125 个时间点。无 window/rate sweep。
- token 从 candidate 起点按固定网格覆盖至终点，尾段 padding+mask；不只截中心 120 秒，不裁掉长事件的两端。
- ACC/GYRO 分别按真实 timestamps 映射到同一时间网格；禁止用数组行号代替时间，禁止把各重叠 token 串起来假装连续信号。
- 降采样前使用固定抗混叠处理；在连续有效 span 内处理，不跨传感器 dropout 滤波或插值。
- 超过 200 ms 的观测间隙不插值。token 有效覆盖率低于 0.8 则整个 token 标为无效；阈值固定，不按结果调整。
- 无效 token 保留位置、时间和 mask，不从序列中删除。全无效 candidate 保留其行，走第 5.4 节 fallback。
- scaling 只在有效样本上拟合和应用；无效采样点在 scaling 后填 0，token mask 与 sample mask 独立保留。局部池化必须排除无效位置，不能让 padding 长度成为伪动作特征。
- 不跨 session 或跨 candidate 取样，不借用候选外未来数据补齐。
- 使用已验证的重力对齐算法时只采用该样本可访问的确定性信号，ACC/GYRO 应用同一个旋转矩阵；不能重力对齐两次。
- P1 记录并固定重力估计所需数据范围、失败条件与 fallback；如现有对齐函数无法可靠复用，停止审阅，不能静默改为未对齐输入。
- 首版不做随机旋转、channel dropout、时间扭曲等 augmentation，以减少自由度。

25 Hz 等参数是本任务首版工程预注册选择，不是已证实的最优值。如原始有效采样率、FD 输入或现有物理单位不满足这些条件，P1 在训练前报告冲突，而不是自动换一组参数。

### 5.2 覆盖完整候选的轻量结构

局部 encoder 固定使用现有 SmallTokenTCN 结构的受控移植：6 输入通道、hidden 24、3 个 residual blocks、dilation 1/2/4、token embedding 16D。不直接依赖旧 R1 population/cache 代码。

事件级结构固定为：

```text
candidate 的全部 5s tokens
  → 逐 token TCN，得到 16D（有效 token 单独批处理）
  → 按 candidate 的真实起止时间分成 64 个等时长 bin
  → 每 bin：mean(16) + max(16) + relative center + relative width
            + valid coverage + valid-token fraction = 36D
  → 36→32 pointwise projection
  → 3 个小型时序 residual blocks，kernel=3，dilation=1/2/4
  → masked mean(32) + masked max(32)
  → linear 64→16 + ReLU → linear 16→1 → sigmoid
  → branch_probability
```

token 按中心 timestamp 分配 bin；不跨 bin 人工拼接。empty bin 全零，单独 mask；每个卷积 block 后重新 mask 无效 bin。coverage 按 bin 真实有效时间比例计算，不用 token 数代替时间覆盖。

固定 64 bin 是有意的有损时间压缩：保留全区间粗粒度顺序，但长事件的细间隔会被压缩。报告 candidate duration、每 bin 时宽分布及长事件影响，不能宣称精确保留所有长程细节。

padding 不参与 mean/max；masked max 的全空输入必须给有限值，不得产生 NaN/-inf。原始 gap 不能因删除无效 token 而被拼接成连续动作。

局部 encoder 与事件聚合器总 trainable parameters 必须 `<100,000`。任意训练 arm 都使用同一结构，不因结果不好自动加层。

### 5.3 训练合同

- 损失：candidate bag 的 binary cross entropy；不把 meal 内每个 token 都标成 intake。
- 优化器：AdamW，learning rate `1e-3`，weight decay `1e-4`，gradient clipping norm `1.0`；不搜索。
- batch：8 candidate bags，subject→class→candidate 分层抽样，类别存在时正负各半；单类训练分区直接报告不可用，不能伪造另一类。
- 训练目标：每次拟合固定 `2,000 optimizer steps`，不采用 outer/inner held-out early stopping；每 100 steps 保存训练 loss、梯度范数和实际样本计数。
- seed 首轮 `20260924`；确认轮 `20260925` 仅在 P3 满足主 gate 后使用。FD checkpoint 固定，不按目标数据选 seed。
- memory microbatch/gradient accumulation 可以减小，但必须保持有效 batch 和 optimizer step 数；算法超参数不变。
- 不足 2,000 steps 的预算截停标记 `INCOMPLETE_BUDGET`，不是算法 negative result。
- 只保留 final checkpoint 和一个 rolling recovery checkpoint，不按验证分数保存 best。

这是初始固定预算，不保证一定收敛。若最后 400 steps 的训练 BCE 均值仍比之前 400 steps 降低超过 10%，标记 `CONVERGENCE_UNRESOLVED`，只能请求人工复核，不自动加 steps，也不能宣布表示无效。

### 5.4 无有效运动输入

不得丢弃这个 candidate 或用 GT 决定 fallback。单分支 probe 对该行输出当前训练分区的候选阳性比例并标记 invalid；P3 最终预测对该行直接使用 **同一 T 分区重新拟合的 control verifier score**。

这个路由只依赖预注册的 signal validity，不依赖预测置信度。所有 arm 评价同一 candidate ledger；报告 fallback 数量与 subgroup 分布。

### 5.5 只用训练侧的 smoke checks

- [ ] 人工构造不规则 timestamps、重复点、gap、跨 session、全空、短尾段样本，逐一测试。
- [ ] 合成两个 token 集合相同、顺序不同的事件；无序 control 必须置换不变，时序网络必须具有可表达不同输出的能力。
- [ ] 用合成顺序任务验证能学习顺序，不能只检查 tensor shape。
- [ ] 在某个 T 的 16 个固定训练 bag 上验证梯度与小样本记忆能力；不读取任何 V 指标。诊断权重销毁，不得用于正式 arm。
- [ ] 对 CPU/CUDA embedding/score 做有限值和 tolerance 检查；决策距离阈值超过 `1e-5` 时输出一致，边界单独记录。
- [ ] checkpoint round-trip 后 CPU 输出可复现；无 GPU 时可推理。

输出 `P1_READY` 或 `IMPLEMENTATION_BLOCKED`；失败不进入 P2，不用更大网络掩盖问题。

---

## 6. P2 — 合法 inner 上的 Scratch / FD 表示对照

### 6.1 四个 arm，一种架构

| Arm | token encoder | 事件聚合器 | 用途 |
|---|---|---|---|
| Q0 | frozen random | 在 T 上训练 | 随机非线性表示对照 |
| Q1 | scratch，在 T 上训练 | 在 T 上训练 | 目标数据独立学习 |
| Q2 | FD pretrained，冻结 | 在 T 上训练 | 迁移表示本身的价值 |
| Q3 | FD pretrained，在 T 上微调 | 在 T 上训练 | 迁移加微调价值 |

Q0/Q1 使用配对初始化；Q2/Q3 使用同一 FD checkpoint。batch 序列与目标训练步数一致。FD 预训练耗时单独报告，不能宣称总训练预算完全相同。

scratch 不通过也必须允许在预算内评价 Q2/Q3。若 FD 无许可或无有效标签，停止 FD arm 并说明；不得自动下载替代数据集。Q1 仍可独立给出结果。

### 6.2 FD 任务与使用条件

数据标识：`doi:10.48804/CN8VBB`，参考 [官方数据介绍](https://research.wur.nl/en/datasets/replication-data-for-eating-speed-measurement-using-wrist-worn-im/)。P0 保存实际版本与许可证据；比赛允许外部数据不等于可以忽略数据许可，不自动重新分发 FD 原始数据或权重。

优先采用已有可核验的 FD episode/bite 预处理，不引用历史 `+.035` 作为新结果。

- 若存在可靠 intake 点/区间标注：5s token 中心±0.5s 内有 intake 中点为 positive；token 全区间与 intake 区间及其前后各 2.5s 扩张均不相交为 negative；其余 ambiguous 不用于 source token 分类。
- 若仅有 meal interval 标签：使用 meal bag supervision 和同一事件分支，不能生成伪 bite labels；这属于任务类型变更，训练前提交简短 amendment，不静默执行。
- FD participant 以身份分组，固定 80% train / 20% validation，seed `20260924`；同一人的不同 session/左右手不能跨组。
- source pretraining 固定 `10,000 optimizer steps`，batch 128 tokens，AdamW 参数同 P1；source normalization 只用 source-train。固定 final checkpoint，FD validation 只做健康检查，不选最佳 epoch。
- source/target 单位、重力对齐和采样管线一致。source scaling 随 checkpoint 固定，Q2/Q3 不另用目标全数据统计替换。Q0/Q1 scaling 只拟合各自 T。
- source-only checkpoint 在确认无人群重叠、未用目标数据选取后可跨目标 folds 共享；目标 fine-tuned checkpoint 绝对不能跨不相容 training subject 集共享。

无法确定 participant 身份、source 标签语义或单位时输出 `FD_DATA_REVIEW_REQUIRED`，不自行猜测。

### 6.3 Probe 评价

在 5 个 outer-train wrapper 中，分别沿用固定 4 个 inner subject folds；所有 arm 的 V、candidate ledger 与标签完全一致。只训练运动分支，不提前训练新 event verifier。

报告：sample prevalence、candidate 数、always-positive/negative baseline、average precision（AP）、ROC-AUC、log loss、Brier、固定 0.5 threshold 的 candidate precision/recall/F1。candidate F1 不作为唯一 gate，不调 probe threshold。

另外报告各 wrapper 的合并 inner OOF、各 subject 的指标与缺失原因。AP/AUC 的单类 subject 是 undefined，不强行写 0 或 1。wrapper 间共享受试者，不能当作 5 个独立实验计算显著性。

### 6.4 P2 工程 gate

预注册主排序指标为 5 个 wrapper 的 macro mean AP；并列（差小于 `1e-6`）选 inference 更便宜者，再按 Q1、Q2、Q3 顺序打破平局。

某个 Q1/Q2/Q3 可进入 P3，要求：

1. macro mean AP 比 Q0 高至少 `0.02`；
2. 至少 4/5 wrapper 的 AP 高于 Q0；
3. macro mean log loss 不比 Q0 高超过 `0.02`；
4. 无 split/provenance/input/预算不完整问题，无未解决的训练收敛告警。

这些是控制试验数量的工程 gate，不是统计显著性判定。只有一个符合者进入 P3；多个符合者按上述规则选一个，不把所有 arm 都跑完整 event challenger。

FD 优于 scratch 的单独结论要求：最佳 FD arm 相对 Q1 macro mean AP `>= +0.01` 且 4/5 wrapper 正向。未达到只能叫“未证明迁移增益”，不能阻止合格 Q1 进入 P3，也不能将 FD 宣传为已获益。

无 arm 通过：`P2_NO_SUPPORTED_BRANCH`。这只是本版配置无支持，不写“时序路线不可能”。AP 改善但 gate 未过：记录后停止，不换 pooling/window/seed 追分。

### 6.5 顺序依赖诊断不选模型

只对选定 branch，在 inner V 上做固定 seed 的 embedding 顺序置换：保持时间 bin 的位置/mask，置换有效 bin 的运动表示，查看 AP/log loss 变化。它属于扰动诊断，不是独立 challenger，也不产生 event F1。

无序 mean/max control 必须置换不变；时序分支置换无损失则不得声称 H2 已被证实，但 H1/H3 仍可能成立。此结果不用于选择第二个架构或新参数。

---

## 7. P3 — 单分支概率的事件级增量验证

### 7.1 固定接口与三个输出

只运行 P2 选中的一种训练方式，冻结 architecture、pretraining、采样、normalization、steps、loss 与 seed。

- **C0 exact-protocol control：** 116D → 原 LR / constrained LGBM。
- **C1 challenger：** 116D + `temporal_branch_probability` → 同样的 LR / constrained LGBM。
- **C2 branch-only diagnostic：** 直接对 branch probability 做相同 event decoder 评价；不增加新模型、不得据此切换主 arm。

117D 顺序固定，最后一维就是概率。首版不加入 GYRO5、PCA、16/32/128D embedding、额外 quality 特征或 meta fusion。单一概率是刻意的复杂度控制；若其增量无效，不能据此宣称高维表示也无效。

### 7.2 Policy 冻结的准确含义

复用当前 incumbent 算法和网格，但不能用携带 held-out labels 选择结果的参数冒充无泄漏固定 policy。

- candidate generation 所需可学习 admission 设置，仅在每次训练范围 T 的内部 OOF 上选择；对相应 V 的 C0/C1/C2 使用完全相同的结果。
- 保存 raw、admitted、post-NMS ledger 及 hash；arm 之间不得重新挑 admission threshold/NMS/subject cap。
- 最终 event decoder 的 NMS/cap 规则及值由 C0 的合法 inner selection 固定，所有 arms 共享；没有规则位置自由度。
- LR/LGBM 的超参数与 blend weight grid 沿用 incumbent，具体值写入 plan。C0 先选一个 blend weight，C1 使用同一值，禁止 C1 另做 blend sweep。
- **允许 C0/C1 分别在合并 inner OOF 上选择一个最终 event threshold**，因为分数尺度不同；选择器、网格和 tie policy 必须相同。
- 同时提供使用 C0 threshold 的 C1 固定阈值诊断，但不得择优报告。主结果始终是上述分别 calibration 的版本。

最终 threshold 和 C0 policy 在 pooled OOF 上选择，因此本节数字仍叫 selection score。上面 T 内的 candidate 设置与 pooled OOF 上的 final event policy 属于不同层级，必须在 `policy_manifest.json` 分栏记录。

### 7.3 不直接复用旧 0.76560 当 control

本轮 end-to-end OOF 边界收紧后，控制成绩可能改变。必须重新生成同协议、同 candidate universe 的 C0，与 C1 配对。

记录旧 control 的差异来源：候选身份、训练 subjects、base score、归一化、OOF 层级和 policy。不能调参“修复”到恰好 0.76560。

旧分数只作历史参考；新旧不同不自动证明 release 错误。若涉及生产协议本身的重大问题，单独报告并停止扩展范围。

### 7.4 主要与次要指标

主要：官方 matcher 下 pooled inner event F1、TP/FP/FN、PPV/recall。153 个 GT 在不同 outer-train wrapper 中重复出现时，必须报告重复计数规则；不能把 612 条暴露数说成 612 个独立 GT。

强制 event accounting：

- GT 按稳定 truth key：TP retained/recovered/lost；
- FP 按稳定 candidate key：FP retained/removed/introduced；
- 匹配重分配、同一 GT 的候选替换单独列，不能当成多次救回；
- short/medium/long recall、candidate recall、预测数、invalid-input fallback 数；
- 每 wrapper F1/TP/FP；每 subject F1 的 median/IQR/10th percentile；improved/tied/worsened subjects；
- GT=0 且 pred=0 的 subject F1 记 undefined 并报告数量；同 subject 多 wrapper 重复结果不能当独立 subjects，另给 subject×wrapper 表和按 subject 平均后的描述统计；
- runtime、吞吐、peak VRAM/RAM、checkpoint size、cache size、CPU 推理开销。

P2/P3 阶段只用各自 inner 的 FP/FN 作 subgroup，不回查 sealed E04 的具体 candidate 身份。

### 7.5 P3 gate：相对改进为主，绝对值作为历史参照

令新配对 control 为 F0，challenger 为 F1，`delta=F1-F0`。所有 gate 先要求 provenance、训练完整性和保护检查通过。

**强 gate：**

1. `delta >= 0.020`；
2. 至少 4/5 wrapper F1 严格提高；最差 wrapper 下降不超过 `0.020`；
3. pooled TP 不下降且 FP 减少，或 TP 增加且 `新增净 FP / 新增净 TP <= 1.0`；若 TP 轻微下降，要求下降数不超过 `floor(0.01 * pooled_GT)`，且净减少 FP 至少为净损失 TP 的 3 倍。三种情况满足其一即可；
4. short recall 下降不超过 `0.020`；
5. 没有超出资源限额或未解决的实现/协议问题。

若满足强 gate 但绝对 inner F1 `<0.790`，输出 `P3_STRONG_RELATIVE_REVIEW_REQUIRED`，说明 control protocol 改变；不静默调整绝对门槛。

若同时满足强 gate 且 inner F1 `>=0.790`，输出 `P3_PRIMARY_STRONG_PASS`，仅允许执行预注册确认 seed，不允许 outer。

**中等 gate：** `delta >=0.010`、4/5 wrapper 正向，但不满足强 gate：`P3_REVIEW_REQUIRED`，停止人工审阅，不自动增加 seed。

其余有效完成结果：`P3_INNER_GATE_FAILED`。不跑第二个 encoder、不反复改 threshold grid、不以 subgroup 亮点代替主 gate。

### 7.6 一次确认 seed

只有 `P3_PRIMARY_STRONG_PASS`，且剩余预算允许，才能对同一 C1 配置运行 seed `20260925`。控制 C0 可在 identity 完全一致时复用。

确认要求：相对 C0 `delta >=0.010`，至少 4/5 wrapper 正向，short recall 不下降超过 0.020，TP/FP 成本满足同一规则。

不挑两次里最好 seed；未来 outer 默认采用首轮 seed。两次都记录，不平均后掩盖失败，不自动做 seed ensemble。确认失败输出 `P3_SEED_INSTABILITY_REVIEW_REQUIRED`；预算不足输出 `P3_CONFIRMATION_INCOMPLETE`。

全部通过：`READY_FOR_HUMAN_OUTER_REVIEW`，提交材料后停止。

---

## 8. P4/P5 — 只定义后续标准，本轮不授权

### 8.1 P4：一次明确批准的 full outer

人工批准必须指定唯一 config hash、seed、selection procedure、资源上限。outer 预测开始前锁定全部选择；只允许每个 outer-train 内已经注册的自动选择流程，不允许人工逐折修改。

正式比较必须重用当前 eligible truth 与官方 matcher，报告 pooled F1 和完整 TP/FP accounting。达到 `>=0.7000` 才能写“本次开发基准突破 0.70”；inner、oracle、candidate F1 一律不能满足这个目标。

outer 低于 0.70：如实记录，不以同一 outer 结果挑 seed、改阈值或追加训练后再跑。`0.65143 < F1 <0.70` 只能叫部分提升，是否发布另行决定。

达到 0.70 也不等于独立泛化已证实；同时报告受试者间分布。如需置信区间，按 subject 成组 bootstrap，不能按 candidate 独立抽样。

### 8.2 P5：人工批准后才 promotion

未来 promotion 另立执行清单：模型固化、production schema/version、完整推理链、CPU/GPU parity、可执行文件、`-device cpu/gpu` 接口、dist-only 集成验证、README、manifest、许可证、回滚材料。

P0–P3 不提前迁移到 `src/pipeline`，不提前构建 dist，不更新正式 headline。无法合法分发的新权重不得进入 dist。

---

## 9. 资源预算与限额：不能用节省计算换泄漏

以下是待人工随本文批准的**首轮默认硬上限**，不是当前设备性能断言。P0 用硬件盘点、合成输入或训练侧无标签前向 throughput 估算核对，不执行 optimizer step；明显不适配就暂停审阅。

| 项目 | 首轮上限/规则 |
|---|---|
| 网络架构 | 1 种；总参数 <100,000 |
| 初筛 arms | Q0–Q3；FD 条件不成立不新增替代 arm |
| 进入完整 P3 的表示 | 最多 1 个 |
| seeds | 1 primary + 最多 1 条件确认；不做第三 seed |
| outer calls | 0 |
| 运行中的 GPU 训练 | 最多 1 个进程 |
| PyTorch/DataLoader | workers ≤4；CPU 总活跃计算线程默认 min(8, 可用逻辑核数−2)，至少 1 |
| LightGBM/BLAS | 总预算内分配，不叠加多层 n_jobs=-1 |
| 新增 GPU 占用时间 | 合计 ≤24 GPU 小时；所有成功/失败/重试计入 |
| 新增计算墙钟 | 合计 ≤48 小时；人工等待不计，CPU fallback 仍计 |
| 新增持久缓存 | ≤12 GiB；source raw/session 共用存量不复制 |
| 新增 checkpoint | 合计 ≤2 GiB；每拟合只保留 final+rolling |
| 项目盘剩余空间 | 应 ≥20 GiB 且 ≥当前计划最大临时占用的 2 倍，否则暂停 |
| GPU/RAM | 预计占用不超过当前可用量 80%，不得主动挤占其他训练 |
| 单失败配置重试 | 同类实现错误最多 2 次；第三次人工复核 |

固定训练步数与硬资源限额同时生效：不能在预算耗尽后继续，也不能把截停 arm 当有效负结果。

缓存至少按 session/span/token 起点去重；不为每个 fold 复制同一 raw token，不把整个 session 常驻 GPU。token encode 可 chunk，event bins 流式累计；任何 memory 优化必须通过等价性测试。

### 9.1 拟合数量必须先算

5 outer-train wrappers × 4 inner folds = 20 个 T/V 分区。

- P2 最多 4 arms ×20 =80 个目标分支拟合，另加 1 次 source-only FD 预训练。
- P3 选定 arm 每 T 需要 3 个 subfold branch 拟合 +1 个 refit；同 identity 的 P2 refit 可以复用。因此最多新增 60 个 primary branch 拟合。
- 确认 seed 最多新增 80 个 branch 拟合。
- base macro/micro 与 verifier 拟合另计，不藏在上述数目之外；确切数量写入 dry-run 计划。

默认预算可能不支持全部阶段；这不是允许减少隔离层数的理由。允许安全地复用完全相同祖先、配置和训练 subjects 的模型；不允许跨 partition 偷复用。预计无法完成 P2 时，在训练前就提交预算冲突，不先消耗一半算力再报告。

### 9.2 实施者自身工作效率

按需读取摘要和 JSON 字段，禁止反复把数百 KB provenance ledger 全量打印到上下文。每个阶段只跑相关 tests，不反复做 full regression、dist build 或 100GB raw parsing。

子代理仅处理独立、有界任务，例如 gap/mask 单测审查或 provenance 图审查；不得让多个代理同时改同一 runner、同时训练或各自开超参数试验。每次交接包含文件、接口、禁止项、验收测试和当前 commit。

---

## 10. 文件组织、接口与缓存生命周期

### 10.1 新增文件布局

```text
scripts/research/representation_v2/
  __init__.py
  contracts.py          # schema、类型、配置、状态枚举
  provenance.py         # subject dependency graph 与 cache identity
  dataset.py            # 合法 population、timestamps、mask、source/target 适配
  models.py             # token encoder、64-bin sequence head
  train.py              # 固定预算训练、checkpoint、资源计量
  crossfit.py           # end-to-end T/V/U 隔离和 score 对齐
  evaluate.py           # probe/event metrics、accounting、gate
  run.py                # 唯一 CLI：preflight / cache / probe / event / report
tests/research/
  test_representation_v2_contracts.py
  test_representation_v2_dataset.py
  test_representation_v2_models.py
  test_representation_v2_crossfit.py
  test_representation_v2_evaluate.py
outputs/challenger_70/representation_v2/
  plan.json
  protocol.md
  evidence_review.json
  budget.json
  state.json
  summary.md
  manifests/
  cache/
  checkpoints/
  p1_input_audit/
  p2_probe/
  p3_event/
  cleanup_manifest.json
```

不要预先创建无内容脚本；这些是模块职责边界，按任务落实。若同职责可复用现有纯函数，优先 import，不复制生产代码。旧 `scripts/research/temporal_rep/` 不作为本轮入口，不破坏其历史可复现性。

### 10.2 对外数据合同

`CandidateKey = (session_id, start_ms, end_ms, source_id)`；同区间多 source 的 canonical union identity 使用 incumbent helper，额外保留 producer/config hash，不自行改变去重规则。

`Partition` 含 `outer_fold, inner_fold, subfold, train_subjects, score_subjects, parent_ids`。

`TokenBag` 含 `candidate_key, values[N,6,125], sample_mask[N,125], token_mask[N], token_center_ms[N], start_ms, end_ms, coverage`。

`BranchPrediction` 含 `candidate_key, probability, valid_input, model_id, normalization_id, provenance_id`。

`ScoreLedger` 一行一 candidate，含 subject/session、fold roles、GT linkage（仅当前授权范围）、116D schema hash、C0/C1 score、policy ID；输出行必须按 key join，不能假设两个数组顺序自然一致。

### 10.3 缓存 identity 与删除

identity 至少包含 raw fingerprint、preprocessing/schema、sampling、mask规则、单位/坐标、partition、train subjects、model/config/seed。纯 raw token cache 不含拟合统计；normalized/embedding/score cache 必须含相应 fitted artifact identity。

每次成功阶段后：删除本轮失败中间 tensor、孤立分片、重复 checkpoint 和临时导出；保留 plan、split/provenance、metrics、score ledger、错误日志和最终必要权重。不能因结果不好删除负证据。

删除前必须列出精确绝对路径、大小、依赖和可重建方法；验证位于本轮 `representation_v2/cache` 或 `checkpoints` 允许目录内。不按模糊 glob 删除整个项目、原始数据、FDdatasets、共享 session cache、incumbent 或 dist。

跨轮存量缓存清理只提出清单，未经核对不删除。`cleanup_manifest.json` 记录删除内容、字节数、保留证据与恢复方法。不要为“工作树干净”强制回滚用户改动。

---

## 11. 分任务实施清单与验证

以下 CLI 是要实现的合同，当前不代表命令已经存在。所有训练命令须在人工批准及上游 gate 后运行。默认 Python 使用前先检查环境。

### Task 1：P0 contracts 与防火墙

**文件：** `contracts.py`、`provenance.py`、`test_representation_v2_contracts.py`；只读参考 `src/pipeline/challenger/protection.py`、`src/pipeline/crossfit.py`。

**接口：** `validate_partition(partition)`、`assert_no_heldout_ancestors(artifact_id, heldout_subjects, graph)`；后者递归检查父节点，不只检查叶节点。

- [ ] 先写失败测试：直接 subject overlap、grandparent overlap、未知 parent、错误 source allowlist、旧 R1 cache identity 均拒绝。
- [ ] 实现最小 schema/图检查与 preflight；任何缺失 provenance 按拒绝处理。
- [ ] 增加反例：child 不含 subject s，但 grandparent 含 s，必须抛出异常。
- [ ] 测试通过后输出 P0 dry-run，不训练；提交本任务代码与测试。

```python
import pytest
from scripts.research.representation_v2.provenance import assert_no_heldout_ancestors

def test_transitive_subject_leak_is_rejected():
    graph = {
        "base": {"fit_subjects": ["s1"], "parents": []},
        "meta": {"fit_subjects": ["s2"], "parents": ["base"]},
    }
    with pytest.raises(ValueError, match="s1"):
        assert_no_heldout_ancestors("meta", {"s1"}, graph)
```

### Task 2：P1 timestamp/token 与完整覆盖

**文件：** `dataset.py`、`test_representation_v2_dataset.py`；参考 `src/pipeline/preprocessing/timeline.py` 与既有重力对齐测试，不绕过其约定。

**接口：** `build_token_bag(session, candidate, preprocess_config) -> TokenBag`、`pool_time_bins(token_embeddings, bag, bins=64)`。

- [ ] 先测不等间隔时间点必须按实际时间插值；250ms gap 不补点；两 session 不拼接。
- [ ] 10 分钟 candidate 的开始、中段、结束都应出现在 token index，不能止于中心 120 秒。
- [ ] 全空、少于 5 秒、重复 timestamps、ACC/GYRO 不同时间轴、NaN、单位不符分别有明确拒绝或 mask 行为。
- [ ] 坐标旋转一致性与 gravity fallback 测试通过后才生成最小 cache。
- [ ] 测 cache key：改变 rate、gap阈值、source fingerprint 后必须 miss；规范化 fit subjects 变化时不得复用 normalized cache。
- [ ] 提交代码/测试和 P1 input audit；不要提交大 token 数据。

### Task 3：P1 模型与训练健康

**文件：** `models.py`、`train.py`、`test_representation_v2_models.py`。

**接口：** `MotionBranch.forward(values, masks, times, intervals) -> logits[B]`；checkpoint 必须含 config、weights、normalization、provenance 和 optimizer step。

- [ ] 先测 shape、参数上限、全空 finite 输出、padding 不改变结果、checkpoint round-trip。
- [ ] 实现局部 encoder、64-bin 有序 head 和 masked pooling，不引入第二种架构。
- [ ] 实现合成顺序任务与小训练集 smoke，不读 validation。
- [ ] 测 optimizer step 计数、预算截止、resume identity；同类错误重试不重置总预算。
- [ ] 验证 CPU/CUDA tolerance、chunk encode 等价性；输出 P1 状态并提交。

### Task 4：P2 FD 与 probe

**文件：** `dataset.py` 的 FD adapter、`train.py`、`crossfit.py`、`evaluate.py`、`run.py`，对应测试。

**接口：** `fit_branch(train_partition, arm, config)`、`predict_branch(model, score_partition)`、`evaluate_probe(ledger)`；任何函数不接受 outer error atlas 参数。

- [ ] 先测 FD participant split、ambiguous label 排除、source-only checkpoint 可复用边界。
- [ ] 使用 spy estimator 记录每次 fit subject，断言 P2 各 V 在所有拟合祖先中缺席。
- [ ] 先完成 dry-run 成本表；超过预算停止，不启动 80 fits。
- [ ] 在批准范围运行固定 Q0–Q3，按第 6 节生成唯一选择结果；没有 gate 就没有 P3。
- [ ] 保存 all-positive baseline、所有 arm 和失败日志；更新 summary、清理中间文件、提交代码与轻量结果。

### Task 5：P3 全链路 crossfit 和事件配对

**文件：** `crossfit.py`、`evaluate.py`、`test_representation_v2_crossfit.py`、`test_representation_v2_evaluate.py`；只读复用 `src/pipeline/challenger/replay.py` 的官方匹配入口。

**接口：** `fit_inner_stack(T, selected_branch_config)`、`predict_inner_stack(stack, V)`、`evaluate_event_pair(control_ledger, challenger_ledger, truths)`。

- [ ] 先构造 E06 型错误依赖图，确认测试能抓到 meta held-out label 经底层训练进入 meta-training features 的路径。
- [ ] 验证 T 内 U0/U1/U2 的 source/normalization/candidate/encoder/refit 身份；缺任何祖先即失败。
- [ ] 修改某个 V 的标签，重建其评分过程：candidate/feature/score hashes 必须不变，只有事后的 metrics 能变化；pooled threshold selection 另属选择阶段。
- [ ] 输入 ledger 打乱行序，按 key join 后结果不变；重复 key、漏行直接报错。
- [ ] C0/C1 admitted candidate hash、policy control 参数一致；加入 branch 禁止悄悄重新选择 cap/NMS。
- [ ] 官方 matcher fixture 一致；TP/FP accounting 恒等式闭合；无效输入 fallback 精确等于 C0。
- [ ] 运行唯一 primary P3；严格按 gate 决定确认 seed 或停止；提交报告与状态。

### Task 6：收口与人工 handoff

**文件：** `summary.md`、`state.json`、`budget.json`、`cleanup_manifest.json`、experiment registry 新条目及 README 研究进度段。

- [ ] registry 仅追加，experiment ID 使用 `RV2-P2-001`、`RV2-P3-001` 等新命名；diagnostic/probe 正式 outer performance 字段必须为 null。
- [ ] 核查保护快照与工作树，无 dist/release/model 变化。
- [ ] 清理本轮孤立缓存，保留复现最低材料和全部负结果。
- [ ] 报告完成阶段、未完成项、阻塞、确切使用预算、是否具备下一阶段资格。
- [ ] 在 `READY_FOR_HUMAN_OUTER_REVIEW` 或其他终止状态停止，不自行批准下一阶段。

### 11.1 建议命令合同

```powershell
& D:/Anaconda3/envs/bme/python.exe -m pytest tests/research/test_representation_v2_contracts.py -q
& D:/Anaconda3/envs/bme/python.exe -m pytest tests/research/test_representation_v2_dataset.py tests/research/test_representation_v2_models.py -q
& D:/Anaconda3/envs/bme/python.exe -m pytest tests/research/test_representation_v2_crossfit.py tests/research/test_representation_v2_evaluate.py -q
& D:/Anaconda3/envs/bme/python.exe -m scripts.research.representation_v2.run preflight --plan outputs/challenger_70/representation_v2/plan.json
```

CLI 后续阶段使用 `cache / probe / event / report` 子命令，必须自动检查 `state.json` 和 plan hash。不得提供绕过 gate 的 `--force` 或隐藏 outer 开关。每个任务提交前运行相应 tests、`git diff --check`，只 add 明确文件。

---

## 12. 输出与最终审阅模板

### 12.1 最低结果集合

每个 P2/P3 正式 attempt 保存：

- `plan.json`、`manifest.json`、`split_manifest.json`、`provenance.json`；
- `runtime.json`、`training_health.json`、`cache_inventory.json`；
- `candidate_ledger.parquet` 或现有依赖支持的等价压缩表，不为格式安装新库；
- `fold_metrics.json`、`subject_metrics.json`、`aggregate_metrics.json`；
- P2：`probe_controls.json`、`transfer_comparison.json`、`order_perturbation.json`；
- P3：`policy_manifest.json`、`event_accounting.json`、`inner_gate.json`；
- `report.md`，明确本阶段没有 outer evaluation 或 release change。

### 12.2 人工报告必须回答

1. 当前正式发布是否仍是 0.6514285714？保护快照是否一致？
2. 新 control 与旧 0.76560 的口径是否相同？若不同，具体差在哪里？
3. scratch、FD frozen、FD fine-tuned 哪个有支持？支持是 candidate 还是 event 级？
4. 时序顺序本身是否有证据，还是仅局部动作表示有用？
5. C1 相对 C0 的 TP retained/recovered/lost、FP removed/introduced 分别多少？
6. 至少 4/5 wrapper 是否同向？受试者分布、短餐和最差 wrapper 如何？
7. source/target/inner/meta 层是否都有可核验隔离？有没有复用不合法 checkpoint？
8. 用了多少训练 fits、steps、GPU/墙钟时间和磁盘？剩余预算多少？
9. 失败代表实现/收敛未解决、局部表示无支持、迁移无支持，还是 event 互补性无支持？
10. 当前状态是否仅允许人工审阅？不允许用“准备好了”代替明确状态。

### 12.3 最终纪律

**0.70 是真实事件级开发目标，不是必须通过调参凑出的数字。**

**缓存可以共享，训练身份不能共享；逐行 OOF 不等于跨层无泄漏。**

**不再用 scratch 失败禁止迁移，但也不预设迁移一定有效。**

**保留事件的时间结构，不把无序池化失败写成时序学习失败。**

**预算不足就停止；证据不足就承认；任何 outer 与 promotion 都留给人工授权。**
