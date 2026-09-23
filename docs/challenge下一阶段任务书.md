# BMEcontest-2 下一阶段 Challenge 任务书

## Representation Phase

### Temporal Motion Encoder → FD Transfer → Event-level MIL → Existing Event-Stack

---

# 0. 任务性质

本阶段正式结束以下低成本优化路线：

* Context-v2 / Context-v3
* hard-negative weighting
* candidate-control relaxation
* NMS / subject-cap tuning
* adaptive LR/LGBM fusion
* GYRO5 scalar augmentation
* GYRO5 residual
* handcrafted periodicity expansion
* subject-relative normalization
* PPG / HRV
* fold-specific intervention

这些路线已有足够负证据，不再继续微调。

本阶段研究目标变为：

> 寻找当前 63D macro + 47D micro + 116D event representation 尚未表示的 candidate 内部时序运动信息。

---

# 1. 当前正式 incumbent

正式 incumbent 始终保持：

* run key：`160afaf81debf1ee`
* strict subject-disjoint nested 5-fold development F1：`0.6514285714`
* TP / eligible truth / predictions：`114 / 153 / 197`

不得修改：

* `release/event_stack_incumbent.json`
* `models/event_stack/160afaf81debf1ee/**`
* `dist/**`
* 正式 release
* 正式 submission
* production README headline

本阶段默认不授权 full outer。

---

# 2. 历史证据与本阶段 hypothesis

已知事实：

1. Context-v1 曾产生最大一次可靠 information jump。
2. GYRO5 说明 5 s 短尺度包含信息，但简单 scalar statistics 无法稳定转化为 event-level收益。
3. FP 中大量属于：

   * long continuous movement
   * low-frequency long movement
   * fragmented high activity
4. 单纯运动强度不是 meal-specific：

   * 部分 introduced FP 的 GYRO5 variance 甚至显著高于 recovered TP。
5. 过去 TCN / deep-score 在严格协议审计后仍有结构性正证据。
6. FD Episode pretraining 曾得到跨折一致正增益。
7. 旧深度 ranker 失败的重要原因是 proposal coverage，而当前 event-stack 已解决大部分 proposal / candidate coverage 问题。

因此本阶段核心 hypothesis：

> 当前剩余错误的主要缺失信息，不是“5 秒运动有多强”，而是“短尺度运动片段如何在 candidate 内随时间组织”。

具体表现可能包括：

* intake-like burst
* burst 间隔
* repeated reach-mouth-return morphology
* intermittent high-motion evidence
* sustained non-meal motion
* local movement concentration
* temporal ordering

---

# 3. 总体架构原则

禁止推翻当前 event-stack。

新 representation 必须作为附加 evidence branch 接入 incumbent。

目标架构：

```text
Raw ACC + GYRO
      │
      ▼
5 s motion token encoder
      │
      ▼
token sequence
      │
      ▼
event-level temporal / MIL aggregation
      │
      ▼
learned temporal evidence
      │
      ├──────────────┐
      │              │
existing 116D        │
Context-v1           │
      │              │
      └──── verifier ┘
             │
             ▼
        incumbent-style
        event decision
```

禁止：

```text
raw IMU → giant neural network → final event
```

---

# 4. 本阶段分四个 Gate

严格按以下顺序：

## R0 — Infrastructure / cache check

只确认可复用数据。

## R1 — Temporal Token Representation Probe

确认 learned 5 s token 是否产生超越现有 scalar statistics 的信息。

## R2 — FD Transfer Probe

比较 scratch 与 FD-pretrained encoder。

## R3 — Event-level MIL Challenger

将最优冻结 encoder 接入 inner-only event-stack。

## R4 — Optional Combination / refinement

只有 R3 强通过才讨论。

本任务默认：

**最多执行到 R3。**

禁止自动 outer。

---

# 5. 计算预算原则

本阶段必须继续遵守：

**Cache first.**

优先复用：

* parsed IMU cache
* session segmentation
* candidate intervals
* fold assignments
* incumbent candidate ledger
* eligible truth
* Context-v1 cache
* incumbent OOF event scores

禁止每次实验重新：

* parse 全部 100 GB raw
* 重提 macro
* 重提 micro-v1
* 重提 Context-v1
* rebuild dist
* clean-machine
* full regression suite

新计算仅限：

> candidate 周边短尺度 ACC+GYRO representation。

---

# 6. R0 — 数据与缓存准备

先检查当前仓库是否已有：

* raw/session cache
* gravity-aligned ACC/GYRO
* candidate timestamp mapping
* candidate → session identity
* fold identity
* GT linkage
* FD preprocessed dataset

如果已有：

直接复用。

如果缺失：

只建立最小必要 cache。

建议：

`outputs/challenger_70/temporal_rep/cache/`

内容：

* candidate IMU slice index
* subject/fold mapping
* event labels
* FD token index
* normalization metadata
* cache manifest

不要复制完整 raw 数据。

---

# 7. 输入信号

第一版只使用：

* ACC x/y/z
* GYRO x/y/z

优先使用已有 gravity-aligned representation。

不要加入：

* PPG
* heart-rate features
* Context-v1 feature vector
* macro/micro probabilities

这些留到 event aggregation 层。

Token encoder 首先只研究：

> motion representation。

---

# 8. Token 时间尺度

第一版固定：

**5 s token**

原因：

Data Sprint 已证明 5 s 是当前最有信息的短尺度。

不要重新 sweep：

* 3 s
* 4 s
* 6 s
* 8 s
* 10 s

避免重新打开 window-search 空间。

可以使用：

* 5 s window
* stride 2.5 s 或 5 s

但 stride 必须 preregister。

---

# 9. Token preprocessing

优先保持简单：

* timestamp-resampled IMU
* gravity alignment
* per-channel robust scaling

任何 learned normalization 必须 fold-contained。

禁止：

全数据 global mean/std。

---

# 10. R1 — Token Encoder Probe

第一版 encoder 必须轻量。

推荐：

### Option A

small 1D CNN

### Option B

small dilated TCN

只允许其中一种作为 primary。

不要同时做 CNN / TCN / GRU / Transformer benchmark。

推荐首选：

**small dilated TCN**

---

# 11. 网络规模

目标参数量：

约：

`10^4 – 10^5`

建议：

* 3–5 convolution / residual blocks
* hidden channels 16–32
* kernel 3–5
* simple global pooling
* token embedding 16–32D

禁止：

* > 1M 参数
* Transformer
* BiLSTM stack
* large GRU
* giant CNN

---

# 12. R1 supervision

第一版不要直接用整个 meal interval 的每个 5s window 强标为 positive。

原因：

meal event 内大量时段并非真实 intake gesture。

优先采用：

## Weak / bag-aware supervision

正 candidate：

一组 5s motion tokens

负 candidate：

另一组 5s motion tokens

目标：

让正 candidate 中至少部分 token 获得高 intake-like evidence。

可以采用：

* top-k pooling
* max / softmax pooling
* attention pooling

第一版优先：

**top-k mean pooling**

因为：

* 简单
* 稳定
* 低容量
* 不增加复杂 attention 参数

---

# 13. R1 不直接优化完整 event F1

R1 是 representation probe。

目标：

确认 learned temporal embedding 是否包含现有 scalar representation 没有的信息。

允许使用：

* bag classification
* candidate classification
* pairwise separation

但必须 subject-disjoint。

R1 不产生正式 challenger headline。

---

# 14. R1 必须有三个 controls

### Control A

现有简单 GYRO5 statistics。

### Control B

random lightweight temporal encoder。

即：

保持 architecture，但 encoder frozen random。

### Challenger

trained temporal encoder。

这样可以回答：

> 提升是否真的来自学习时序结构，而不是参数维度增加。

---

# 15. R1 主要评价

不以 ROC-AUC 单独决定。

至少同时看：

* TP candidate vs FP candidate
* reachable FN vs FP
* fold consistency
* learned representation incremental value

可以使用小型 fold-contained Logistic probe：

```text
existing compact features only
vs
existing compact features + temporal embedding
```

重点：

> temporal embedding 是否提供 incremental information。

---

# 16. R1 成功 Gate

只有同时满足：

1. trained encoder 明显优于 random encoder；
2. learned temporal evidence 在至少 4/5 folds 方向一致；
3. reachable FN 更趋向 TP；
4. final FP 不出现同等程度提升；
5. incremental probe 明显优于 GYRO5 scalar control；

才进入 R2/R3。

否则：

`TEMPORAL_REPRESENTATION_NOT_SUPPORTED`

停止。

---

# 17. R2 — FD Transfer Probe

只有 R1 显示 learned temporal representation 有价值时运行。

核心问题：

> FD intake-gesture pretraining 是否使 learned motion encoder 更稳定、更可迁移？

---

# 18. FD 使用原则

使用已存在合法预处理链路。

保持：

* 数据许可
* participant-level isolation
* no competition validation leakage

FD 数据只能用于：

* encoder pretraining

不能：

* 用 competition outer labels 选择 FD checkpoint。

---

# 19. R2 必须比较

### A — random initialization

competition-data-only training。

### B — FD pretrained

FD pretraining → competition inner fine-tune。

### C — frozen FD encoder

FD pretrained encoder frozen，只训练 aggregation / head。

这样区分：

* transfer representation value
* fine-tuning value
* architecture本身 value

---

# 20. FD pretraining task

优先使用与 ingestion motion 直接相关的目标。

例如：

* intake gesture vs background
* bite-like episode classification
* contrastive intake/background discrimination

不要为了本阶段发明复杂 self-supervised objective。

优先使用已有已验证 Episode pretraining链路。

---

# 21. FD Gate

FD-pretrained 必须：

* 多 fold 稳定优于 scratch；
* 不能只改善单 fold；
* 不明显伤害 short-event evidence；
* candidate-level evidence方向符合 meal-like motion hypothesis。

历史 `+.035` 只能视为 motivation。

不能直接复用旧分数作为当前证据。

---

# 22. R3 — Event-Level MIL Challenger

只有：

* R1 通过；
* R2 中选出一个明确 encoder config；

才运行正式 inner-only challenger。

---

# 23. R3 encoder 冻结原则

进入 R3 前：

选择一个 encoder config：

* architecture
* token length
* hidden size
* pooling
* pretraining mode

全部冻结。

不要在 event-level F1 上重新搜索 encoder architecture。

---

# 24. Event representation

对每个 candidate：

```text
token1
token2
token3
...
tokenN
```

通过 MIL / temporal pooling 得到：

`temporal_event_embedding`

第一版控制维度：

`16–32D`

---

# 25. Event aggregation

第一版推荐：

* embedding mean
* embedding max
* embedding std
* top-k pooled embedding

或一个固定的 top-k temporal pooling。

禁止第一版使用：

* Transformer event aggregator
* BiLSTM
* large attention stack

如果简单 pooling 已经有效，再讨论复杂模型。

---

# 26. 接入 incumbent 的方式

第一版：

```text
existing 116D
+
16–32D temporal embedding
```

但为了避免 E05 类似的 destructive global retraining，必须同时设置：

### Control A

incumbent 116D

### Control B

temporal embedding only

### Challenger C

116D + temporal embedding

目标：

确认 temporal representation 与 incumbent information 是互补的。

---

# 27. 可选 branch-score 方案

如果直接拼接产生高 churn，可以预注册一个替代：

```text
existing verifier
+
temporal branch score
```

而不是直接 concat 全 embedding。

但：

不能看到结果后再临时决定。

必须在 R3 `plan.json` 中指定 primary 和 optional fallback。

---

# 28. Candidate generation 保持不变

R3 第一版禁止：

* 改 macro candidate
* 改 micro candidate
* 改 NMS
* 改 subject cap
* 改 admission threshold

只验证：

> 新 temporal representation 是否改善现有 candidate 的 event discrimination。

---

# 29. Verifier model

为了隔离 representation 效果：

优先保持 incumbent：

* LogisticRegression
* constrained LightGBM
* incumbent-style blend

不要同时升级 verifier architecture。

---

# 30. R3 preregistration

正式执行前创建：

`outputs/challenger_70/experiments/R3-001/plan.json`

必须记录：

* encoder config
* pretraining state
* embedding dimension
* token stride
* pooling
* verifier feature schema
* LR config
* LGBM config
* fusion procedure
* threshold procedure
* inner splits
* seeds
* runtime limit
* success gate
* abort conditions

运行后不得修改。

---

# 31. R3 leakage requirements

必须严格保证：

* competition encoder fine-tuning subject-disjoint；
* base token encoder不见 held-out subject；
* event verifier不见 held-out subject；
* normalization fold-contained；
* FD participant split独立；
* outer labels不参与任何 architecture selection；
* no sealed-outer feature selection。

---

# 32. R3 primary metric

Primary：

**event-level aggregate inner OOF F1**

secondary：

* TP
* FP
* FN
* precision
* recall
* per-fold F1
* short recall
* medium recall
* long recall
* candidate recall
* prediction count

---

# 33. 必须做 event accounting

相对于 exact inner incumbent control：

* TP retained
* TP recovered
* TP lost
* FP retained
* FP removed
* FP introduced

重点：

> 是否减少 E05 那种 destructive churn。

---

# 34. 重点 subgroup

报告：

* short meals
* reachable FN
* final FP
* fold0
* fold2

但禁止：

* subgroup-specific threshold
* fold-specific training

只观察统一模型是否改善弱区域。

---

# 35. R3 Strong Gate

当前参考 inner：

`≈0.7656`

Strong pass 建议：

`>=0.790`

且：

* 至少 4/5 folds 改善；
* aggregate TP/FP trade-off明显改善；
* TP recovered > TP lost；
* FP removed > FP introduced，或 TP gain 足以补偿；
* mechanism 与 temporal representation hypothesis一致；
* 无明显 runtime explosion；
* no leakage。

---

# 36. Moderate Gate

如果：

`0.780–0.790`

并且：

* 4/5 folds 正向；
* event accounting 很干净；

输出：

`TEMPORAL_REP_REVIEW_REQUIRED`

不自动 outer。

---

# 37. Fail Gate

如果：

`<0.780`

或：

* 2/5、3/5折随机方向；
* improvement完全来自单 fold；
* destructive churn严重；

则：

`TEMPORAL_REP_INNER_GATE_FAILED`

停止当前架构。

---

# 38. 不允许 near-miss chasing

如果：

`0.788`

不要自动：

* 多加层
* 换 kernel
* 换 hidden size
* 改 pooling
* 加 attention
* 改 threshold

记录。

人工审核。

---

# 39. 计算预算

第一版应设硬预算。

建议：

* single encoder architecture
* <=3 random seeds
* no architecture sweep
* no large hyperparameter search

如果 GPU 可用：

可用 GPU。

最终 deployment 仍需 CPU inference 可行。

---

# 40. Runtime要求

记录：

* FD pretraining time
* competition fine-tune time
* token extraction time
* event inference overhead
* serialized model size

目标：

最终仍能保持合理 CPU推理。

暂不要求维持当前 18 s/day。

但不接受数量级爆炸。

---

# 41. Internal self-supervised learning

本轮不优先。

只有当：

* scratch temporal encoder 有信号；
* FD transfer 无明显收益；

才允许人工决定是否进入：

`R2B — Internal self-supervised IMU pretraining`

不自动执行。

可能方向：

* masked segment reconstruction
* temporal contrastive learning
* augmentation consistency

但不属于当前任务默认范围。

---

# 42. Label-noise-aware training

暂缓。

当前已知自由生活标签可能有漏标。

但不要同时改变：

* representation
* supervision
* negative policy

否则无法隔离增益来源。

只有 temporal representation 本身成立后，再讨论：

* confidence-aware negative sampling
* PU learning
* label smoothing

---

# 43. 不重新使用 adaptive fusion

R3 中：

不要重新启动 E06 adaptive meta-fusion。

固定 incumbent-style fusion即可。

---

# 44. 不重新使用 GYRO5 scalar block

不要将 E05 6D block额外加入新 temporal representation。

否则无法判断 learned representation本身是否有效。

若 learned encoder已包含 gyro信息：

它应自行学习。

---

# 45. 推荐 research 输出路径

统一：

`outputs/challenger_70/temporal_rep/`

建议：

```text
cache/
r1_token_probe/
r2_fd_transfer/
experiments/R3-001/
summary.md
manifest.json
```

---

# 46. 测试分层

R1/R2：

只跑：

* dataset split tests
* subject isolation
* normalization
* tensor shape
* deterministic cache tests

R3：

增加：

* OOF provenance
* event matcher regression
* feature-schema compatibility

禁止：

* full dist validation
* clean-machine
* submission tests

除非 production code 被触碰。

---

# 47. Experiment registry

正式 R3 attempt：

写入：

`outputs/challenger_70/experiment_registry.jsonl`

R1/R2 diagnostic 不伪装成 formal event challenger。

失败记录保留。

---

# 48. 代码隔离

研究代码优先：

`scripts/research/temporal_rep/`

模型 prototype：

`outputs/challenger_70/temporal_rep/models/`

不要提前修改：

`src/pipeline`

只有未来 promotion 才迁移 canonical implementation。

---

# 49. R1/R2 最终状态

必须输出之一：

`TEMPORAL_REPRESENTATION_NOT_SUPPORTED`

`TEMPORAL_REPRESENTATION_SUPPORTED`

`FD_TRANSFER_NOT_SUPPORTED`

`FD_TRANSFER_SUPPORTED`

`TEMPORAL_REP_READY_FOR_R3`

然后按 gate 执行。

---

# 50. R3 最终状态

只能：

`TEMPORAL_REP_INNER_GATE_FAILED`

`TEMPORAL_REP_REVIEW_REQUIRED`

`TEMPORAL_REP_READY_FOR_HUMAN_OUTER_REVIEW`

`INVALID_EXPERIMENT`

不得自动运行 outer。

---

# 51. 如果 R3 强通过

即使：

`inner F1 >=0.80`

也禁止 outer。

最终报告给人工：

* control F1
* temporal-only F1
* combined F1
* TP/FP accounting
* per-fold results
* short-meal effect
* weak-fold effect
* FD vs scratch
* runtime
* leakage audit

由人工决定是否消耗 full outer。

---

# 52. 如果 R3 失败

不要立即：

* 上 Transformer
* 上更大 TCN
* 上 GRU
* 多尺度模型 ensemble
* 超参数 sweep

先判断失败原因：

### Representation not learned

→ stop temporal branch.

### Representation有 candidate-level information但 event-level不增益

→ 调查 aggregation / supervision，而不是加模型尺寸。

### FD transfer负迁移

→ 保留 scratch结果，停止 FD。

---

# 53. 最值得观察的机制

本阶段最重要的不是单纯 F1。

要回答：

> learned representation 是否能区分“持续高运动的假阳性”和“间歇 intake-like motion 的真实 meal”？

如果 answer = yes：

这是目前最可能的第二次 information jump。

---

# 54. 成功时理想 event accounting

不要求精确达到，但理想趋势：

```text
TP recovered > TP lost
FP removed > FP introduced
4/5 or 5/5 folds positive
```

并且：

* short FN 改善
* long-continuous FP 减少
* low-frequency-long FP 减少

---

# 55. Outer预算继续严格保护

当前仍然只有一个正式 outer headline：

`0.6514285714`

不要因为研发时间延长就反复跑 outer。

下一次 outer 应该只给：

> 在 inner 中出现明确 representation-level improvement 的候选。

---

# 56. 本阶段最核心的问题

不是：

> 哪个神经网络最强？

而是：

> 现有事件级统计特征没有保留的“运动序列结构”，能否被一个很小的 learned encoder稳定捕获？

---

# 57. 模型选择原则

模型越小越好。

如果：

small TCN = 0.795

large Transformer = unknown

优先 small TCN。

比赛目标是：

* 泛化
* 可复现
* CPU可运行
* 证据链清楚

不是模型复杂度。

---

# 58. 本阶段的核心对照

始终保持：

```text
Incumbent representation
vs
Temporal representation
vs
Incumbent + Temporal representation
```

以及：

```text
scratch
vs
FD pretrained
```

这四个对照比大规模调参重要。

---

# 59. 最终科研问题

本任务最终必须回答三个问题：

1. 5 s raw ACC+GYRO 的 learned temporal representation 是否包含 GYRO5 statistics 之外的稳定信息？
2. FD intake-gesture pretraining 是否提高这种 representation 的跨主体泛化？
3. 这种新表示能否作为额外 evidence，把 event-level inner F1 从约 `.7656` 明显推向 `.79+`？

只有三个问题都给出正证据：

才值得消耗下一次 outer。

---

# 60. 最重要执行原则

**Stop tuning the same information.**

**Learn the information that scalar features discard.**

**Preserve the incumbent event-stack.**

**Use deep learning only as a representation tool, not as a replacement pipeline.**

**Transfer learning must beat scratch.**

**Candidate-level improvement is not enough; event-level improvement is mandatory.**

**No outer until a real information jump appears.**

正式 incumbent 始终保持：

`0.6514285714 / 114 / 153 / 197`

直到人工明确批准新的完整 outer evaluation。
