# BMEcontest-2 下一轮 Challenge 任务书

## E05-GYRO5 → E06-FUSION → E07-COMBINATION

### Cache-first / Inner-only / Mechanism-isolated

---

# 0. 当前基线与研究状态

当前正式 incumbent：

* run key：`160afaf81debf1ee`
* strict subject-disjoint nested 5-fold development F1：`0.6514285714`
* TP / eligible truth / predictions：`114 / 153 / 197`

当前正式 release、models、dist 均保持冻结。

已完成：

* E00 incumbent reproduction
* E01 Error Atlas
* E02 Context-v2：inner gate failed
* E03 hard-negative：inner gate failed
* E03.5 candidate-control audit
* E04 verifier failure audit
* Data Characterization Sprint

Data Sprint 最终状态：

`MULTIPLE_SIGNALS_FOUND_NEEDS_HUMAN_SELECTION`

已确认两个值得进入 Tier 1 的方向：

### Signal A — 5 s GYRO short-scale representation

最强 proxy：

`5 s gyro_variance`

诊断中位数：

* TP：`56944`
* reachable FN：`39369`
* final FP：`9257`

且：

* TP–FP 方向在 `5/5 folds` 一致；
* reachable FN 更接近 TP；
* temporal-scale signal gate 已通过。

### Signal B — LR/LGBM adaptive fusion

`LR - LGBM >= 0.20`：

* reachable FN：`12 / 31`
* final FP：`9 / 83`
* 方向在 `4 / 5 folds` 一致。

其他数据路线：

* periodicity：未成立；
* subject-relative shift：未成立；
* direct TCN/Transformer：暂不支持；
* hand/scenario：metadata 不足；
* fold0/fold2：主要表现为较高 FP/session density，约其他折 `1.36×`；
* FP dominant modes：

  * long continuous movement：39
  * low-frequency long movement：27
  * fragmented high activity：11

---

# 1. 本轮总体决策

本轮按以下顺序执行：

`E05-GYRO5 → E06-FUSION → E07-COMBINATION`

但每一步都有独立 gate。

禁止因为任务书中写了后续阶段就自动全部执行。

逻辑：

1. 先验证新的信息源：5 s GYRO；
2. 再验证现有信息的自适应融合；
3. 只有两者都独立成立，才允许组合；
4. 整个任务默认禁止完整 outer evaluation。

本轮目标不是直接报告新的正式 F1。

目标是产生：

> 一个或最多两个值得人工批准 full outer 的 inner-only challenger。

---

# 2. 最高优先级原则

## 2.1 Cache first

每一步开始前必须先判断：

> 是否可以用已有 frozen cache 完成？

如果可以：

禁止重新执行：

* raw full parsing
* macro full extraction
* micro-v1 full extraction
* Context-v1 full extraction
* sealed outer replay
* dist rebuild
* release rebuild
* full clean-machine validation

---

## 2.2 只重算真正变化的模块

E05 改变的是：

`5 s GYRO representation`

因此只允许：

* 从已解析 IMU 或已有低层缓存提取新增 5 s GYRO feature；
* 重新训练使用这些新增 feature 的 verifier 部分。

不要重新计算无关模块。

E06 只改变 fusion：

因此若 base OOF scores 已存在：

**禁止重新训练 base LR/LGBM。**

---

# 3. 不得修改的资产

在人工批准 promotion 前，不得修改：

* `release/event_stack_incumbent.json`
* `models/event_stack/160afaf81debf1ee/**`
* `dist/**`
* official submission
* production visual
* incumbent release metadata

研究代码优先放：

`scripts/research/`

或已有等价 research 路径。

研究结果统一放：

`outputs/challenger_70/`

---

# 4. 本轮禁止事项

禁止：

1. full outer evaluation；
2. automatic promotion；
3. 修改 eligible truth；
4. 修改 matcher；
5. 修改 outer fold；
6. 修改 NMS；
7. 修改 admission threshold；
8. 修改 subject cap；
9. 修改 Context-v1；
10. 重新启动 Context-v2/v3；
11. hard-negative v2；
12. PPG；
13. external FD；
14. TCN；
15. Transformer；
16. RNN；
17. AutoML；
18. Optuna / 大规模 Bayesian search；
19. 大规模 window sweep；
20. 大规模 feature ranking；
21. fold-specific threshold；
22. fold-specific model；
23. 根据 sealed outer GT 挑 feature；
24. 删除失败实验；
25. 重用 experiment ID。

---

# 5. E05-GYRO5 — Short-scale GYRO Representation

## 5.1 Hypothesis

当前 micro-v1 主要使用 15 s 尺度。

Data Sprint 表明：

> 5 s GYRO variance 包含现有 15 s representation 未充分表达的短时腕部运动强度信息。

该信息可能：

* 提高部分 reachable FN；
* 区分 long / low-frequency FP；
* 同时改善 recall 与 precision。

---

# 6. E05 不做完整 micro-v2

第一版禁止：

* 5/10/15/30/60 s 全尺度展开；
* 第二套完整 47D micro features；
* 新 candidate generator；
* 新 sequence model。

第一版只做：

> 极小的 5 s GYRO representation augmentation。

---

# 7. E05 feature budget

新增 feature 上限：

**10 个。**

推荐从以下语义中预注册少量 feature：

* 5 s gyro variance
* 5 s gyro RMS
* 5 s gyro energy
* 5 s gyro max magnitude
* 5 s gyro active fraction
* 5 s gyro local peak count
* 5 s gyro concentration
* 5 s / 15 s gyro variance ratio
* local max 5 s variance inside candidate
* local 5 s contrast vs nearby baseline

不要求全部使用。

正式 E05-001 开始前，在：

`plan.json`

中固定 feature schema。

---

# 8. 禁止 outer-derived feature selection

不能因为 Data Sprint 中：

`gyro_variance@5s`

表现最好，就进一步在 sealed outer 上搜索：

* 第几个轴最好；
* 4.5 s 还是 5.5 s；
* variance 还是 std；
* threshold 怎样最好。

5 s 是已有 diagnostic evidence 支持的 hypothesis。

正式 challenger 内的任何进一步选择必须在 inner hierarchy 中完成。

---

# 9. 建议 E05 最小版本

优先：

`E05-001 = incumbent 116D + very-small GYRO5 feature block`

不要修改：

* candidate generation
* candidate-control
* Context-v1
* verifier family
* event policy

目标：

隔离回答：

> 5 s short-scale GYRO information 本身是否有 event-level增益？

---

# 10. E05 模型

保持 incumbent verifier 架构优先。

即：

* LogisticRegression
* constrained LightGBM
* incumbent-style fusion

不要同时改变模型结构。

否则无法区分：

> feature improvement

和：

> model improvement。

---

# 11. E05 preregistration

正式运行前创建：

`outputs/challenger_70/experiments/E05-001/plan.json`

必须记录：

* hypothesis
* feature list
* feature definitions
* window definition
* aggregation definition
* candidate population
* normalization
* LR grid
* LGBM settings
* fusion procedure
* threshold procedure
* inner folds
* seeds
* gate
* abort conditions

一旦正式执行：

不得修改 E05-001。

若 implementation bug：

记录失败；

创建 E05-002。

---

# 12. E05 normalization

任何 learned normalization：

* scaler
* percentile transform
* clipping threshold
* robust center/scale

必须 fold-contained。

禁止：

全数据先算 normalization，再做 CV。

如果 feature 本身可以无需 learned scaling：

优先保持简单。

---

# 13. E05 candidate population

保持 incumbent candidate population。

第一版不要因为有 5 s signal 就增加候选。

原因：

先验证 representation，而不是 candidate recall。

只有 E05 证明 5 s feature 有判别价值但候选 recall 成为明显瓶颈，未来才单独讨论 candidate-generation extension。

---

# 14. E05 primary metric

Primary：

**event-level aggregate inner OOF F1**

保持：

* incumbent eligible denominator
* incumbent event matcher
* incumbent evaluation semantics

同时报告：

* TP
* FP
* FN
* prediction count
* precision
* recall
* short recall
* medium recall
* long recall
* per-fold F1

---

# 15. E05 必须做 event accounting

相对于 exact inner control：

* TP retained
* TP recovered
* TP lost
* FP retained
* FP removed
* FP introduced

尤其检查：

> 是否真的同时改善 reachable FN 与 long/low-frequency FP。

---

# 16. E05 FP mechanism report

只使用已有预定义 FP behavioral modes：

* long continuous movement
* low-frequency long movement
* fragmented high activity

比较 challenger 相对 control：

* removed FP
* retained FP
* introduced FP

禁止为了 E05 结果重新定义 FP taxonomy。

---

# 17. E05 short-meal report

因为 5 s hypothesis 可能与短餐有关，报告：

* short TP recovered
* short TP lost
* short FP change

但：

**不要为 short meal 单独调 threshold。**

---

# 18. E05 inner control

Control 必须是：

> 完全相同 inner protocol 下的 incumbent representation。

不能直接拿历史 `.76560` 数字做唯一 control。

如果已有可信、完全相同 split/cache 的 control artifact，可直接复用。

否则只重跑最小必要 control。

---

# 19. E05 gate

参考 inner：

约 `0.7656`

E05 outer-review 候选门槛：

### Strong pass

`inner aggregate F1 >= 0.780`

并满足：

* 多数 folds 同方向；
* 不是单 fold 驱动；
* TP/FP accounting 支持 GYRO5 hypothesis；
* 无明显 short recall tradeoff；
* 无 leakage。

### Weak / ambiguous

`0.772–0.780`

保留结果，但：

`REVIEW_REQUIRED`

默认不授权 outer。

### Fail

`< 0.772`

或提升几乎完全来自单 fold：

`E05_INNER_GATE_FAILED`

停止扩展 GYRO5。

---

# 20. E05 不允许 feature explosion

如果 E05-001 失败：

禁止立即：

* 加 50 个 GYRO feature；
* 增加 10 s / 30 s / 60 s；
* 上 TCN；
* 大网格搜索。

先停下人工复核。

---

# 21. E05 runtime budget

尽量复用：

* parsed IMU
* candidate windows
* 15 s feature cache
* existing verifier feature cache

只补计算 5 s block。

记录：

* extraction time
* training time
* cache hit rate
* model size
* inference overhead

---

# 22. E05 输出

`outputs/challenger_70/experiments/E05-XXX/`

至少：

* `plan.json`
* `feature_schema.json`
* `inner_gate.json`
* `fold_metrics.json`
* `aggregate_metrics.json`
* `event_diff.json`
* `fp_mode_effect.json`
* `short_meal_effect.json`
* `leakage_audit.json`
* `runtime.json`
* `manifest.json`
* `report.md`

---

# 23. E05 完成后状态

只能输出：

`E05_INNER_GATE_FAILED`

`E05_REVIEW_REQUIRED`

或：

`E05_READY_FOR_HUMAN_OUTER_REVIEW`

但：

**不要跑 outer。**

然后进入 E06 only if allowed below.

---

# 24. E06-FUSION — Adaptive Verifier Fusion

E06 可以在 E05 之后独立执行。

即使 E05 fail：

E06 仍允许，因为两者机制独立。

---

# 25. E06 hypothesis

Data Sprint 表明：

`LR - LGBM >= 0.20`

出现在：

* reachable FN：12/31
* final FP：9/83

且 4/5 folds 方向一致。

Hypothesis：

> frozen LR/LGBM base models 存在 candidate-dependent complementary information，而固定 scalar blend 不能充分利用。

---

# 26. E06 必须尽可能零重算

优先直接使用：

* base OOF LR scores
* base OOF LGBM scores
* existing candidate metadata
* existing inner folds

如果这些已有：

**禁止重新训练 base LR/LGBM。**

---

# 27. E06 controls

必须比较三者：

### Control A

incumbent scalar blend

### Control B

inner-retuned scalar blend

### Challenger C

adaptive meta-fusion

---

# 28. E06 meta-feature budget

最多：

**8–10 个。**

优先只使用：

* `p_lr`
* `p_lgbm`
* `p_lr - p_lgbm`
* `abs(p_lr - p_lgbm)`
* candidate duration
* has_macro
* has_micro
* macro/micro evidence ratio

禁止：

把完整 116D feature 塞进 meta-fuser。

---

# 29. E06 model

第一版只允许：

`LogisticRegression`

小正则网格。

例如：

`C ∈ {0.1, 1, 10}`

正式运行前预注册。

禁止：

* LGBM meta-fuser
* XGBoost
* RF
* neural net

---

# 30. E06 stacking leakage

base-model probability 必须是 OOF。

禁止：

base model 见过该 subject 后的 prediction 被用于 meta-fuser training。

必须有测试验证：

* subject disjoint
* base prediction OOF provenance
* no outer-val leakage
* no cross-fold normalization leakage

---

# 31. E06 primary metric

同 E05：

event-level inner OOF aggregate F1。

必须报告：

* Control A
* Control B
* Adaptive

同时报告 TP/FP/FN 和 fold results。

---

# 32. E06 gate

adaptive fusion 必须同时：

1. 明显优于 Control A；
2. 明显优于 Control B；
3. aggregate inner F1 建议 `>=0.780`；
4. 多 fold 一致；
5. disagreement subgroup improvement 与 hypothesis 一致。

如果：

Control B ≈ Adaptive

则：

`ADAPTIVE_FUSION_NOT_JUSTIFIED`

即使两者都高。

---

# 33. E06 输出

`outputs/challenger_70/experiments/E06-XXX/`

至少：

* `plan.json`
* `meta_feature_schema.json`
* `control_comparison.json`
* `inner_gate.json`
* `fold_metrics.json`
* `event_diff.json`
* `disagreement_effect.json`
* `leakage_audit.json`
* `runtime.json`
* `manifest.json`
* `report.md`

---

# 34. E07-COMBINATION 触发条件

只有当：

* E05 单独通过 strong gate；
* E06 单独通过 strong gate；

才允许 E07。

如果只有一条通过：

不要组合。

保留该单独 challenger 等待人工 outer review。

---

# 35. E07 hypothesis

E05：

提供新的 short-scale GYRO information。

E06：

改善已有 verifier evidence 的融合。

两者理论上属于不同层次：

`representation → base verifier → fusion`

因此可能存在互补。

---

# 36. E07 组合原则

E07 只允许：

`E05 best frozen config + E06 best frozen fusion`

不得重新联合搜索：

* GYRO feature set
* fusion parameters
* threshold
* model grid

即：

**组合，不重新调参。**

---

# 37. E07 control

必须比较：

* incumbent
* E05 alone
* E06 alone
* E05 + E06

否则无法判断组合是否具有真实增量。

---

# 38. E07 gate

只有组合明显优于两条单独方法时，才值得成为 outer champion candidate。

例如：

* E05 = 0.785
* E06 = 0.781
* combo = 0.787

则组合收益太小。

可能优先选择更简单的 E05。

如果：

* E05 = 0.785
* E06 = 0.783
* combo = 0.802

则组合很有价值。

---

# 39. Complexity penalty

如果两个 challenger inner F1 相近：

优先选择：

* feature 更少
* runtime 更低
* dependency 更少
* artifact 更简单
* leakage surface 更小

的方案。

---

# 40. Outer budget

本任务禁止 outer。

即使出现：

`inner F1 > 0.80`

也只能输出：

`READY_FOR_HUMAN_OUTER_REVIEW`

完整 outer 必须由人工单独批准。

---

# 41. 不允许连续 outer 多候选

未来人工批准 outer 时：

优先选一个 champion。

不要：

* E05 outer
* E06 outer
* E07 outer

全部跑一遍再选最好。

如果 E07 明显最优且机制合理：

优先只跑 E07。

如果组合收益很小：

优先跑更简单的单模型 challenger。

---

# 42. Codex 配额优化

本轮明确以降低执行成本为目标。

不要为每个实验重新：

* 浏览整个 repo
* 重写架构说明
* 重查全部 baseline
* 重跑 full tests
* 重建 artifacts

实验开始时：

只确认与当前实验相关的 invariant。

---

# 43. 测试分层

## Tier 0 / research code

只跑：

* directly affected unit tests
* leakage tests
* cache-contract tests

## Tier 1 experiment

增加：

* event matcher regression
* fold isolation
* feature schema consistency

## promotion 前

才运行：

* full test suite
* clean-machine
* parity
* dist/submission validation

---

# 44. 不要重复 immutable audit

E03.5、E04、Data Sprint 已经冻结。

除非其文件发生改变：

禁止重新 hash / replay / validate。

已有证据视为有效依赖。

---

# 45. Experiment registry

所有正式 Tier 1 attempt 写入：

`outputs/challenger_70/experiment_registry.jsonl`

记录：

* ID
* hypothesis
* result
* failure reason
* gate
* commit

失败记录不得删除。

---

# 46. Commit discipline

建议：

### E05

一个 implementation commit

一个 result/artifact commit

### E06

同样

不要每个小统计文件都产生多个无意义提交。

---

# 47. 最终汇报格式

完成后只需给人工一份紧凑表格：

| Experiment | Control F1 | Challenger F1 | TP Δ | FP Δ | Fold stability | Mechanism | Gate |
| ---------- | ---------: | ------------: | ---: | ---: | -------------- | --------- | ---- |
| E05-GYRO5  |            |               |      |      |                |           |      |
| E06-FUSION |            |               |      |      |                |           |      |
| E07-COMBO  |            |               |      |      |                |           |      |

以及：

### E05

* best config
* feature count
* event accounting
* short-meal effect
* FP-mode effect

### E06

* scalar control
* retuned scalar
* adaptive fusion
* disagreement subgroup result

### E07

* 是否有真实组合增益

---

# 48. 最终允许状态

整个任务结束时只能输出以下之一：

`NO_TIER1_CHALLENGER_PASSED`

`GYRO5_READY_FOR_HUMAN_OUTER_REVIEW`

`FUSION_READY_FOR_HUMAN_OUTER_REVIEW`

`COMBINATION_READY_FOR_HUMAN_OUTER_REVIEW`

`MULTIPLE_TIER1_CANDIDATES_NEED_HUMAN_SELECTION`

`INVALID_EXPERIMENT`

不得自行运行 outer。

---

# 49. 如果 E05/E06 都失败

不要继续：

* E05-v10
* E06-v10
* larger models
* threshold hunting

停止并报告：

`NO_TIER1_CHALLENGER_PASSED`

下一阶段才考虑：

-真正的 multi-scale micro-v2；

* candidate temporal representation；
* lightweight TCN。

---

# 50. 本轮研究目标

本轮不是为了穷举。

我们已经有两个数据支持的 hypothesis。

现在只验证：

### Hypothesis A

> 5 s short-scale GYRO information 能否在保持 incumbent pipeline 基本不变的情况下，改善 event-level TP/FP trade-off？

### Hypothesis B

> LR/LGBM 已有互补信息能否通过低容量 adaptive fusion 转化成稳定 event-level收益？

如果成立：

才花 outer 预算。

如果不成立：

快速停止。

---

# 51. 最重要的执行原则

**Do not re-prove what is already proven.**

**Do not recompute what can be cached.**

**Do not combine mechanisms before each works alone.**

**Do not use outer as a development loop.**

**Do not increase model complexity unless new information requires it.**

**Data signal → cheap isolated challenger → inner gate → human review → outer last.**

正式 incumbent 始终保持：

`0.6514285714 / 114 / 153 / 197`

直到人工批准新的 full outer challenger 并完成全部验证。
