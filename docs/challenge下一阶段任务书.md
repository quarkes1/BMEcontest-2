# BMEcontest-2 下一轮 Challenge 任务书

## E05.1 Decision Churn Audit → E05-R Frozen GYRO Residual

### Low-cost / Cache-first / Incumbent-frozen

---

# 0. 当前正式状态

正式 incumbent 保持：

* run key：`160afaf81debf1ee`
* strict subject-disjoint nested 5-fold development F1：`0.6514285714`
* TP / eligible truth / predictions：`114 / 153 / 197`

当前分支：

`research/f1-070-challenger`

已完成：

* E00 incumbent reproduction
* E01 Error Atlas
* E02 Context-v2：failed
* E03 hard-negative：failed
* E03.5 candidate-control audit
* E04 verifier audit
* Data Characterization Sprint
* E05-GYRO5
* E06-FUSION

当前 Tier 1 状态：

`NO_TIER1_CHALLENGER_PASSED`

---

# 1. 已知 E05 结果

E05-GYRO5：

* Control inner F1：`0.7655986509`
* Challenger inner F1：`0.7651006711`
* TP Δ：`+2`
* FP Δ：`+4`
* only `2/5 folds` improved

但 decision churn 明显：

* TP recovered：11
* TP lost：9
* FP removed：16
* FP introduced：20

short events：

* TP recovered：5
* TP lost：3

因此：

> GYRO5 不是完全无信息，而是改变了大量 decision，但当前“116D + 6 GYRO5 features 后整体重训 verifier”的方式未改善 TP/FP trade-off。

E06 adaptive fusion：

* Control A：`0.7655986509`
* Control B：`0.7655986509`
* Adaptive：`0.7567114094`

因此 E06 路线正式停止。

不要再继续 adaptive fusion。

---

# 2. 本轮核心研究问题

本轮只回答：

> E05 的 churn 是否存在稳定结构，支持“冻结 incumbent，只让 GYRO5 对有限 candidate 做 residual correction”？

重点不是：

> GYRO5 能否重新训练一个更好的 verifier？

这个问题 E05 已经回答：

**当前方式不能。**

---

# 3. 总体执行顺序

严格按：

`E05.1 → human-style gate → optional E05-R`

执行。

不得跳过 E05.1 直接训练 E05-R。

---

# 4. 本轮最高优先级：节省计算

优先复用：

* E05 inner event diff
* E05 GYRO5 feature cache
* incumbent inner OOF scores
* candidate metadata
* fold assignments
* existing matcher outputs

禁止无理由重新：

* 解析全部 raw TSV
* 提取全部 63D macro
* 提取全部 47D micro
* 提取全部 Context-v1
* 训练 incumbent LR/LGBM
* 跑 sealed outer
* rebuild dist
* full regression suite
* clean-machine validation

---

# 5. 不修改资产

禁止修改：

* `release/event_stack_incumbent.json`
* `models/event_stack/160afaf81debf1ee/**`
* `dist/**`
* official submission
* production README headline

本轮只能新增 research artifact。

---

# 6. Phase A — E05.1 Decision Churn Audit

E05.1：

**DIAGNOSTIC ONLY**

不得：

* 训练新模型
* 调 threshold
* 计算 outer F1
* 创建正式 challenger score
* 修改 event policy

---

# 7. E05.1 population

使用 E05 exact inner event diff 中已有四类事件：

1. `TP_RECOVERED = 11`
2. `TP_LOST = 9`
3. `FP_REMOVED = 16`
4. `FP_INTRODUCED = 20`

必须使用 exact identity mapping。

不得重新定义 population。

---

# 8. E05.1 必须分析的变量

对四类事件至少比较：

### GYRO5

* 5 s gyro variance
* 5 s gyro RMS / energy
* 5 s / 15 s gyro variance ratio
* 5 s local concentration

只使用 E05 已经 preregistered / computed 的 feature。

不要新增 feature search。

### incumbent evidence

* incumbent blend score
* incumbent event threshold
* incumbent score margin
* LR score
* LGBM score

### candidate metadata

* duration
* macro evidence
* micro evidence
* macro/micro composition

### fold

* fold id

---

# 9. 重点变量：incumbent margin

定义：

`margin = incumbent_score - incumbent_event_threshold`

必须重点比较：

* TP recovered margin
* TP lost margin
* FP removed margin
* FP introduced margin

核心问题：

> GYRO5 是否主要帮助 incumbent 原本低置信 / 边界 candidate，同时伤害高置信 incumbent decision？

---

# 10. 需要验证的主要 hypothesis

## H1 — Boundary rescue hypothesis

理想结构：

* TP recovered 多集中在低 margin / near-boundary
* TP lost 多集中在高 margin
* FP removed 多集中在边界
* FP introduced 多集中在较低 incumbent confidence

如果成立：

支持 uncertainty-gated residual。

---

## H2 — Monotonic GYRO signal hypothesis

理想结构：

* TP recovered 的 GYRO5 明显高于 FP introduced
* FP removed 的 GYRO5 明显低于 TP recovered
* 方向至少多数 folds 一致

如果成立：

支持简单 residual correction。

---

## H3 — Churn chaos

如果：

* 四类 GYRO5 分布高度混合
* margin 无结构
* fold 方向不稳定

则：

GYRO5 路线停止。

---

# 11. E05.1 禁止事项

禁止：

* 根据四类 event 人工搜索最佳阈值
* brute-force GYRO threshold
* 100 个 interaction
* outer-derived rules
* classifier benchmark
* UMAP / clustering sweep
* post-hoc feature explosion

这是机制审计，不是调参。

---

# 12. E05.1 最低输出

创建：

`outputs/challenger_70/e05_1_churn_audit/`

至少包含：

* `report.md`
* `churn_population.csv`
* `margin_summary.json`
* `gyro5_summary.json`
* `fold_direction.json`
* `manifest.json`

---

# 13. E05.1 report 必须回答

只回答：

1. TP recovered 是否集中于低 incumbent margin？
2. TP lost 是否主要来自高-confidence incumbent decision？
3. FP removed / introduced 的 GYRO5 是否有明显区别？
4. GYRO5 signal 与 incumbent confidence 是否存在稳定 interaction？
5. 多 fold 是否一致？
6. 是否支持 frozen residual？
7. 是否支持 uncertainty gating？

---

# 14. E05.1 Gate

最终必须输出：

`GYRO_RESIDUAL_SUPPORTED`

或：

`GYRO_RESIDUAL_NOT_SUPPORTED`

---

# 15. Supported 最低要求

不能只因为 recovered TP 有 11 个就通过。

至少满足：

* GYRO5 与 margin 存在可解释 interaction；
* TP recovered 与 TP lost 之间有稳定差异；
* FP removed / introduced 不是完全不可分；
* 至少 3/5 folds 大方向一致；
* 机制不是单 fold 驱动。

若不满足：

输出：

`GYRO_RESIDUAL_NOT_SUPPORTED`

停止整个 GYRO5 路线。

不要运行 E05-R。

---

# 16. Phase B — E05-R Frozen GYRO Residual

只有：

`GYRO_RESIDUAL_SUPPORTED`

时允许执行。

---

# 17. E05-R 核心原则

**冻结 incumbent verifier。**

不得重新训练：

* incumbent LR
* incumbent LGBM
* Context-v1 verifier
* candidate generator

base output 为固定：

`p_incumbent`

E05-R 只学习：

`small correction`

---

# 18. E05-R conceptual form

推荐：

`p_final = p_incumbent + delta`

其中：

`delta = g(GYRO5, incumbent_margin, limited interaction features)`

重点：

> residual 默认为接近 0。

只有有证据时才修正 incumbent。

---

# 19. E05-R feature budget

最多：

**4–6 个输入。**

建议：

* incumbent score
* incumbent margin
* 5 s gyro variance
* 5 s / 15 s gyro variance ratio
* optional gyro concentration
* optional margin × gyro interaction

不要加入：

* 全 116D
* 全 6D E05 block 加大量 expansion
* raw micro representation
* Context-v1 full vector

---

# 20. E05-R model

首选：

`LogisticRegression`

或极小 linear residual model。

禁止：

* LightGBM
* RandomForest
* XGBoost
* neural net
* TCN

目标：

最小容量。

---

# 21. Optional uncertainty gate

若 E05.1 明确支持 H1，可以允许：

> residual 只作用于 incumbent uncertainty region。

但 uncertainty region 必须：

* 在 inner training hierarchy 内选择；
* 不看 held-out inner GT；
* 不看 outer；
* preregister search space。

不要手写 sealed diagnostic threshold。

---

# 22. 推荐 gate 形式

如果使用 uncertainty gate：

概念：

`|margin| > M → delta = 0`

`|margin| <= M → allow residual`

M 只能从一个极小 preregistered inner grid 中选择。

例如：

* 0.05
* 0.10
* 0.15

具体值在 `plan.json` 固定。

---

# 23. E05-R preregistration

创建：

`outputs/challenger_70/experiments/E05R-001/plan.json`

记录：

* hypothesis
* exact inputs
* model
* regularization
* uncertainty gate grid if used
* correction formulation
* threshold policy
* inner folds
* seeds
* gate
* abort criteria

正式运行后不得修改。

---

# 24. E05-R 必须保持 incumbent base prediction OOF

训练 residual 时：

base incumbent score 必须来自：

**OOF prediction**

不能使用：

> base model 在见过该 subject 后产生的 score。

必须验证：

* subject isolation
* score provenance
* no held-out leakage

---

# 25. E05-R Control

必须比较：

### Control A

exact incumbent inner OOF：

`0.7655986509`

或完全相同协议下的 cached control。

### Challenger

Frozen incumbent + residual。

禁止重训 Control A。

只要 cache identity 已验证即可直接复用。

---

# 26. Primary metric

仍为：

**event-level aggregate inner OOF F1**

必须同时报告：

* TP
* FP
* FN
* predictions
* precision
* recall
* fold F1
* short / medium / long recall

---

# 27. E05-R 必须输出 churn accounting

相对 incumbent：

* TP retained
* TP recovered
* TP lost
* FP retained
* FP removed
* FP introduced

特别关注：

> 是否显著降低 E05 的 destructive churn。

---

# 28. E05-R 成功结构

E05 原始 churn：

* recover TP 11
* lose TP 9
* remove FP 16
* introduce FP 20

E05-R 理想目标不是复制这套 churn。

而是类似：

* recover fewer TP
* lose much fewer TP
* introduce much fewer FP

例如：

* TP recovered +5
* TP lost -1
* FP removed 8
* FP introduced 2

这种结构远优于高 churn。

---

# 29. E05-R Gate

### Strong pass

要求：

`inner aggregate F1 >= 0.780`

并且：

* 至少 3/5 folds 改善；
* 最好 4/5；
* TP lost 明显低于 E05；
* FP introduced 明显低于 E05；
* mechanism 与 E05.1 一致；
* no leakage。

### Weak

`0.772–0.780`

输出：

`E05R_REVIEW_REQUIRED`

默认不进入 outer。

### Fail

`< 0.772`

或 fold instability：

`E05R_INNER_GATE_FAILED`

---

# 30. 不允许因接近 gate继续调

如果：

`0.779`

不要自动：

* 扩大 M grid
* 加 feature
* 换 LGBM
* 改 threshold
* 加 10s features

记录结果并停止。

---

# 31. E05-R runtime / compute requirement

本实验应非常低成本。

应尽量只依赖：

* cached incumbent OOF scores
* cached GYRO5 features
* cached candidate/event mapping

若发现需要重新跑完整 feature pipeline：

先调查是否可以从 E05 artifact 复用。

---

# 32. 测试要求

只运行针对性 tests：

* residual feature schema
* OOF provenance
* fold isolation
* score calculation
* event matcher regression

不要运行：

* full clean-machine
* dist parity
* submission suite

除非研究代码触及生产代码。

---

# 33. Experiment registry

正式 E05-R 写入：

`experiment_registry.jsonl`

E05.1 diagnostic 不伪装成 challenger。

失败记录保留。

---

# 34. Outer 禁止

本任务不授权 full outer。

即使：

`E05-R inner = 0.80+`

也只能输出：

`GYRO_RESIDUAL_READY_FOR_HUMAN_OUTER_REVIEW`

---

# 35. 如果 E05-R 失败

GYRO5 simple-statistics 路线正式停止。

不要继续：

* GYRO5-v3
* 10s/30s brute-force
* massive feature engineering
* threshold rescue

下一阶段重新讨论：

**temporal-shape representation**

重点可能是：

* rolling 5 s variance distribution
* within-event concentration
* short-vs-long energy structure
* temporal envelope

但不属于本任务。

---

# 36. 不要恢复 E06

Adaptive fusion 已有明确负证据。

禁止：

* fusion-v2
* LGBM meta-fusion
* larger stacking
* disagreement rescue heuristic

E06 正式关闭。

---

# 37. 最终输出格式

如果只完成 E05.1：

报告：

* churn population
* margin pattern
* GYRO pattern
* fold consistency
* residual hypothesis status

最终状态：

`GYRO_RESIDUAL_NOT_SUPPORTED`

或允许进入 E05-R。

如果完成 E05-R：

输出表：

| Metric         | Incumbent | E05 original | E05-R |
| -------------- | --------: | -----------: | ----: |
| F1             |           |              |       |
| TP             |           |              |       |
| FP             |           |              |       |
| TP recovered   |         — |           11 |       |
| TP lost        |         — |            9 |       |
| FP removed     |         — |           16 |       |
| FP introduced  |         — |           20 |       |
| folds improved |         — |          2/5 |       |

并报告：

* feature count
* runtime
* gate status
* leakage audit

---

# 38. 最终允许状态

只能是：

`GYRO_RESIDUAL_NOT_SUPPORTED`

`E05R_INNER_GATE_FAILED`

`E05R_REVIEW_REQUIRED`

`GYRO_RESIDUAL_READY_FOR_HUMAN_OUTER_REVIEW`

`INVALID_EXPERIMENT`

不得自动继续下一路线。

---

# 39. 本轮判断标准

不要问：

> GYRO5 能不能让模型产生更多 TP？

E05 已经证明能。

真正的问题是：

> 能否利用 GYRO5 的新信息，同时尽量保护 incumbent 已经正确的决策？

即：

**low-churn improvement**

而不是：

**global decision-boundary rewrite**

---

# 40. 最重要执行原则

**Freeze what already works.**

**Use GYRO5 as a correction, not as a reason to relearn everything.**

**Analyze churn before training.**

**Prefer fewer corrected events with better net accounting.**

**If the interaction is not stable, stop GYRO5.**

**Do not spend outer budget on an inner-negative challenger.**

正式 incumbent 始终保持：

`0.6514285714 / 114 / 153 / 197`

直到人工批准下一次完整 outer evaluation。
