# BMEcontest-2 F1 ≥ 0.70 安全优化任务书

## Incumbent-Protected Challenger Research

### 0. 任务性质

这是一次**严格受控的 challenger 优化实验**，不是重构任务，不是清理任务，不是重新设计项目，也不是为了“看起来更先进”而替换模型。

唯一研究目标：

> 在绝对不破坏当前正式发布版本、项目结构、submission、visual、canonical inference、release evidence 和已有 F1=0.6514285714 成绩的前提下，寻找能够在完全相同的 leakage-safe strict subject-disjoint nested-CV 协议下，提高 Episode-level global F1 的新方案。

Stretch goal：

```text
aggregate strict nested-CV F1 >= 0.700
```

但不得为了达到 0.70 而放宽协议、修改分母、窥视 outer validation、使用泄漏信息、针对五个 outer fold 手工调参、修改官方 IoU 规则或牺牲证据可信度。

如果没有方案可靠超过 incumbent：

> 保持当前版本不变，即为成功完成任务。

---

# 1. 开始前必须读取并确认当前仓库

禁止凭历史 prompt 或 README 摘要直接修改代码。

首先检查当前 `main` HEAD，并阅读至少：

```text
README.md

release/event_stack_incumbent.json

src/pipeline/
    event_stack.py
    candidate_control.py
    context_features.py
    crossfit.py
    runner.py
    diagnostics.py
    imu_features.py

src/pipeline/features/
    macro.py
    micro.py

src/pipeline/inference/
    predictor.py

scripts/
    evaluate_event_stack.py
    crossfit_event_stack.py
    release_event_stack.py
    train_event_stack.py
    reproduce_release.py
    build_inference_distribution.py
    build_submission.py

outputs/crossfit/
models/event_stack/160afaf81debf1ee/
tests/
```

同时检查当前 git status、HEAD、已有 branch、当前正式 evidence。

必须从实际代码确认，而不是假设：

* candidate generation；
* macro/micro feature schema；
* Context-v1 schema；
* admission/NMS；
* verifier training；
* inner OOF；
* outer validation；
* event policy；
* eligible denominator；
* diagnostic artifact；
* release promotion；
* Predictor runtime；

当前究竟如何实现。

完成后生成：

```text
outputs/challenger_70/baseline_audit.json
```

记录：

```json
{
  "base_git_sha": "...",
  "incumbent_run_key": "160afaf81debf1ee",
  "incumbent_f1": 0.6514285714285715,
  "tp": 114,
  "eligible": 153,
  "pred": 197,
  "fp": 83,
  "fn": 39
}
```

数值必须从正式 evidence 读取，禁止手写成真源。

---

# 2. 不可破坏区

以下内容在 challenger 被正式证明优于 incumbent 之前，全部视为 READ-ONLY。

## 2.1 正式 release

禁止修改：

```text
release/event_stack_incumbent.json
models/event_stack/160afaf81debf1ee/**
```

禁止覆盖、删除、重新训练后写回 incumbent run directory。

禁止修改 incumbent attestation。

禁止运行任何会无条件更新 incumbent pointer 的命令。

---

## 2.2 dist

本轮研究期间：

```text
dist/inference/**
dist/submission/**
dist/visual/**
dist/event_stack/**
```

全部禁止修改。

不得因为 challenger 实验而 rebuild dist。

只有在最终 challenger：

1. 严格 nested-CV 晋级；
2. 全部 evidence gate 通过；
3. 用户明确批准 promotion；

之后，才能进入独立 release 阶段。

---

## 2.3 当前 canonical ABI

当前正式 ABI：

```text
macro      63
micro      47
verifier   116 = 56 base + 60 Context-v1
```

不得直接修改现有 Context-v1 的数学定义。

不得修改现有 47D micro 特征的数学定义。

不得改变 Predictor 对 incumbent bundle 的行为。

如果实验需要新 schema，必须版本化，例如：

```text
context-v2
micro-v2
verifier schema v3
```

旧 schema 必须继续原样可运行。

---

# 3. 评估协议是最高优先级红线

当前正式结果：

```text
run key = 160afaf81debf1ee

TP       = 114
eligible = 153
pred     = 197
FP       = 83
FN       = 39

PPV      = 0.5786802030
Recall   = 0.7450980392
F1       = 0.6514285714
```

逐折：

```text
fold0  16/23/35   F1 0.5517241379
fold1  30/31/46   F1 0.7792207792
fold2  15/27/33   F1 0.5000000000
fold3  25/32/42   F1 0.6756756757
fold4  28/40/41   F1 0.6913580247
```

任何 challenger 必须使用与 incumbent **完全相同的 outer subjects、eligible truth、IoU matcher 和 aggregate metric**。

不得通过以下方式“提高 F1”：

* 修改 eligible 规则；
* 删除困难 subject；
* 删除困难 session；
* 改 IoU threshold；
* 改 matching；
* 使用 macro/micro 不同的 truth definition；
* 根据 outer-val 选择 threshold；
* 根据五折最终结果手工挑参数；
* 针对 fold0/fold2 单独配置；
* 将 outer-val 用于 normalization；
* 用 outer-val FP/FN 训练 hard-negative model；
* 用 outer-val 选择 Context-v2 features；
* 用 outer-val 训练聚类器后再评估同一 fold；
* 全数据 fit scaler/PCA/UMAP 后再做 CV；
* 使用 full-data session statistics 学习参数；
* 通过 random window CV 替代 subject-disjoint CV。

所有 learned transform 必须 fold-contained。

所有 policy selection 必须 outer-train only。

---

# 4. 优化目标不要抽象成“提高 F1”

当前数学状态：

```text
114 TP
83 FP
39 FN
```

F1：

```text
2*114 / (153+197)
= 0.651428...
```

达到约 0.70 的一种目标状态：

```text
TP = 118
pred = 184
FP = 66

F1 = 236 / 337
   ≈ 0.7003
```

因此本轮将优化任务显式拆成：

```text
Target A:
recover >= 4 useful TP

Target B:
remove >= 17 FP

while avoiding equivalent TP loss
```

这不是硬性要求，而是用于判断实验是否真正作用于瓶颈。

---

# 5. 第一阶段：不要训练新模型，先建立 OOF Error Atlas

这是本轮最重要的第一步。

不要先写 Context-v2。

不要先加模型。

先解释 incumbent 的：

```text
83 FP
39 FN
```

## 5.1 数据来源

只能使用 incumbent 已有 strict nested outer-fold OOF diagnostics。

禁止使用 deployment/full-target prediction 替代 OOF diagnostics。

每一个 error 必须可以追溯到：

```text
outer_fold
subject_id
session_id
candidate/event
GT relationship
macro evidence
micro evidence
Context-v1
verifier score
admission status
final decision
```

---

# 6. FN 必须按失败层级分解

对全部 39 FN，自动分类。

至少分：

```text
FN-C0
GT 根本没有任何 raw candidate 达到 IoU >= 0.25

FN-C1
raw candidate 存在，
但 candidate control / NMS / admission 删除

FN-C2
candidate admitted，
但 verifier/event threshold 删除

FN-C3
最终有预测，
但 event geometry 导致 IoU < 0.25

FN-C4
其他 / ambiguous
```

必须输出：

```text
outputs/challenger_70/error_atlas/fn.csv
outputs/challenger_70/error_atlas/fn_summary.json
```

并至少按：

```text
short <10 min
medium 10–20 min
long >=20 min

outer fold
subject
macro-only
micro-only
macro+micro
```

统计。

尤其必须回答：

> 当前 19 个 short-meal FN 中，有多少是 C0 candidate miss，有多少是 C1/C2/C3？

在这个答案出来前，不允许直接增加 micro model complexity。

---

# 7. FP 必须做 taxonomy

对全部 83 个 strict OOF FP 建立 FP atlas。

每个 FP 至少导出：

```text
outer_fold
subject_id
session_id

start_ms
end_ms
duration

macro support
micro support
macro+micro indicator

verifier score
logistic score if available
lgbm score if available

admission rank
event rank

nearest candidate distance
previous/next evidence

Context-v1 60D

session-relative statistics
```

新增 session-relative diagnostics 可以计算，但此阶段仅用于分析，不进入模型。

---

# 8. FP taxonomy 不允许使用标签训练一个新分类器

第一阶段的 taxonomy 是 diagnostics。

允许：

* descriptive statistics；
* correlation；
* PCA/UMAP，仅用于 visualization；
* k-means/HDBSCAN，仅用于探索；
* histogram；
* rank/percentile；
* TP vs FP feature distributions；
* fold-wise comparison。

但如果 PCA/UMAP/cluster 使用了全部 outer folds：

> 只能用于描述，绝不能成为正式 challenger feature transform。

正式模型中的任何 learned transform 必须重新在每个 outer-train 内 fit。

---

# 9. FP Atlas 必须回答这些问题

至少回答：

### Q1

83 FP 中：

```text
macro-only
micro-only
macro+micro
```

分别多少？

### Q2

FP 是否集中在少数 subject？

输出：

```text
FP / subject
TP / subject
pred / subject
```

不要直接据此做 subject-specific rule。

### Q3

FP 是否集中在某些 session-relative rank？

例如：

```text
candidate score percentile
candidate duration percentile
macro percentile
micro percentile
candidate count/session
```

### Q4

FP 是否存在明显重复事件：

```text
同一活动 episode 被切成多个 prediction
```

### Q5

FP 是否属于：

```text
short spike
long continuous activity
macro-high/micro-low
micro-high/macro-low
dense candidate cluster
isolated candidate
```

### Q6

FP 与 TP 在 Context-v1 哪些 feature 上仍然存在稳定 separation？

必须检查 fold consistency。

---

# 10. 生成 Error Atlas 报告

生成：

```text
outputs/challenger_70/error_atlas/report.md
```

报告必须包含：

```text
Incumbent metrics

39 FN stage decomposition

19 short FN decomposition

83 FP taxonomy

per-fold error profile

per-subject concentration

macro/micro support analysis

candidate geometry analysis

Context-v1 residual separation

recommended experiments
```

此时 Agent 才允许提出 challenger。

---

# 11. Challenger Track A：Context-v2

这是最高优先级实验。

原则：

> Context-v2 是 Context-v1 的严格增量版本，不允许破坏 Context-v1。

新增文件优先：

```text
src/pipeline/context_features_v2.py
```

或者在现有模块内通过显式 version branch：

```python
context_features_version="v2"
```

但：

```python
context_features_version="v1"
```

必须 bit-for-bit 保持 incumbent 行为。

---

# 12. Context-v2 优先研究 session-relative deterministic context

优先考虑以下类别。

## 12.1 Session-relative rank

candidate 相对于本 session：

```text
macro score percentile
micro score percentile
verifier-pre-score percentile if leakage-safe
candidate duration percentile
support width percentile
candidate density percentile
```

禁止使用 final label。

---

## 12.2 Robust normalization

例如：

```text
(x - session median) /
(MAD + epsilon)
```

用于：

```text
macro evidence
micro evidence
activity
candidate duration
candidate density
```

必须 deterministic。

---

## 12.3 Candidate competition

增加：

```text
rank in session
top1-top2 margin
score / session max
score / session median

distance to stronger candidate
number of stronger candidates within ±10/20/30 min
```

目的是识别：

> 某个高分候选到底是真正 meal-like dominant episode，还是大量活动候选中的普通峰。

---

## 12.4 Local/global contrast

比较：

```text
candidate interval
pre-context
post-context
whole-session baseline
```

例如：

```text
local_macro_mean / session_macro_median
local_micro_peak / session_micro_p90
local_density - session_density
```

必须防止除零并显式处理缺失。

---

# 13. Context-v2 的泄漏规则

Context-v2 可以读取同一个 raw session 的无标签 probability stream。

但：

* 不得读取 GT；
* 不得读取其他 subject 标签；
* 不得使用 outer-val 统计学习 normalization 参数；
* session 自身 deterministic median/MAD/rank 允许；
* 如果使用 learned transform，则必须 inner/outer fold-contained。

如果某个 feature 是否泄漏存在疑问：

> 默认不使用。

---

# 14. Context-v2 不允许 feature explosion

第一版新增：

```text
<= 32 features
```

优先少量可解释特征。

不要从 116D 直接扩到 500–1000D。

输出 feature schema：

```text
outputs/challenger_70/context_v2_schema.json
```

每个 feature 写：

```text
name
definition
data dependency
label dependency=false
session-local=true/false
learned=false/true
```

---

# 15. Challenger Track B：Hard-Negative Verifier

只有 Error Atlas 证明 FP 有稳定结构后才启动。

目标不是重新训练一个“大模型”。

先保持：

```text
LogisticRegression
+
restricted LightGBM
```

优先调整：

```text
negative sampling
hard-negative weighting
class weight
C grid
restricted tree parameters
blend
```

所有选择必须 inner OOF。

---

# 16. Hard-negative 规则

允许把 outer-train 中：

```text
high-score false candidates
near-threshold negatives
macro/micro disagreement negatives
dense-cluster negatives
```

作为更高权重训练样本。

但必须注意：

对于 outer fold K：

```text
hard negatives
```

只能来自：

```text
outer-train subjects
```

禁止查看 outer-val FP 再把它加入训练。

必须新增测试证明：

```text
outer-val subject IDs
∩
hard-negative training subject IDs
=
empty
```

---

# 17. Challenger Track C：Short-meal Candidate Recovery

仅在 FN Atlas 证明 short FN 主要是 C0 candidate miss 后启动。

当前正式 micro：

```text
window 15 s
stride 7.5 s
47D ACC+GYRO
gravity aligned
```

保持原实现不变。

新增实验版本：

```text
micro-v2
```

---

# 18. Micro-v2 第一轮只做廉价多尺度研究

优先测试：

```text
5 s
10 s
15 s incumbent
30 s
```

但不要立即四模型 ensemble。

第一步只回答：

> 对 incumbent 未覆盖的 short FN，5/10/30 s 是否产生更多 IoU>=0.25 可达候选？

需要输出 candidate-oracle analysis：

```text
scale
overall candidate recall
short candidate recall
medium candidate recall
long candidate recall
candidate count
candidates / TP
```

如果某个尺度：

```text
short candidate recall ↑
```

但 candidate volume 暴涨且最终 verifier 无法控制：

> 不采用。

---

# 19. Candidate-oracle gate

Micro-v2 在进入 full nested verifier 之前必须满足：

```text
short candidate recall
至少有明确改善
```

并且 candidate count 增长必须可解释。

建议工程门：

```text
short candidate recall + >= 0.08 absolute
```

或者：

```text
recover >= 3 incumbent-unreachable short GT
```

否则停止 Track C。

这是实验门，不是正式竞赛指标。

---

# 20. 不要优化 window AUC

本轮禁止把：

```text
window AUC
```

作为主要 promotion 指标。

最终目标只有：

```text
Episode F1
```

window AUC 可以作为 diagnostics。

一个 window model 即使 AUC 更高，如果：

```text
candidate recall 不升
或
FP volume 增加
或
Episode F1 不升
```

则没有价值。

---

# 21. 暂停以下路线

本轮默认禁止启动：

```text
Transformer
大型 TCN 重构
RNN/GRU 新架构
PPG/HRV
外部 FD 迁移
self-supervised pretraining
GPU-only 模型
端到端 raw deep network
```

不是因为这些方向永远无效，而是：

> 当前 10 天优化窗口下，它们会扩大验证面和回归风险。

只有在 A/B/C 全部证明无收益后，才可以把这些写入 future work。

不要擅自实现。

---

# 22. 实验必须预注册

每个正式 challenger 在运行 outer folds 前生成：

```text
outputs/challenger_70/experiments/<experiment_id>/plan.json
```

至少包含：

```json
{
  "hypothesis": "...",
  "base_run": "160afaf81debf1ee",
  "code_sha": "...",
  "feature_version": "...",
  "allowed_hyperparameter_grid": {},
  "primary_metric": "aggregate_episode_f1",
  "secondary_metrics": [
    "tp",
    "fp",
    "short_recall",
    "fold_f1"
  ],
  "selection_protocol": "outer-train inner-OOF only"
}
```

计划文件生成后才能跑完整 outer evaluation。

防止看到结果后偷偷改变 hypothesis。

---

# 23. 研究阶段与正式阶段必须分开

允许快速研究：

```text
diagnostics
candidate oracle
inner OOF
single-fold smoke test
```

但这些不能称为正式结果。

只有完整：

```text
5 outer folds
strict subject-disjoint
inner selection
aggregate official matcher
```

才能与 `.6514285714` 比较。

---

# 24. Promotion Ladder

每个 challenger 依次经过：

## Gate 0 — correctness

```text
unit tests pass
v1 parity unchanged
incumbent reproduction unchanged
```

失败立即停止。

## Gate 1 — mechanism

必须明确：

```text
它解决 FP 还是 FN？
解决多少？
```

不能只有“F1 好像高了”。

## Gate 2 — inner evidence

inner OOF 必须有合理提升。

如果 inner 明显下降，不进入 full nested。

## Gate 3 — strict five-fold

完整 outer 五折。

## Gate 4 — aggregate improvement

最低要求：

```text
F1 > 0.6514285714
```

但仅超过一点不意味着值得替换 submission。

---

# 25. Replacement Gate

由于当前 submission 已经正式提交，换版标准必须比普通 promotion 更严格。

定义三个等级：

### Research improvement

```text
0.6514 < F1 < 0.67
```

记录结果。

默认不替换 submission。

### Strong challenger

```text
0.67 <= F1 < 0.70
```

进入完整 release review。

### Target challenger

```text
F1 >= 0.70
```

进入 replacement candidate review。

这些不是官方获奖线，只是本项目内部换版风险门。

---

# 26. 不能只看 aggregate F1

必须同时报告：

```text
fold0
fold1
fold2
fold3
fold4

TP
FP
FN

PPV
Recall

short recall
medium recall
long recall

candidate recall
candidate volume

inner mean F1
outer aggregate F1
inner→outer gap
```

特别关注 incumbent 弱折：

```text
fold0 = .552
fold2 = .500
```

如果 challenger 的 `.70` 完全来自 fold1 暴涨，而 fold0/fold2 更差：

> 标记为 unstable，不自动晋级。

---

# 27. Fold stability

新增报告：

```text
mean fold F1
std fold F1
min fold F1
max fold F1
```

以及 incumbent/challenger delta：

```text
delta fold0
delta fold1
...
```

不要求每折都提升，但必须解释退化。

---

# 28. TP/FP accounting 是强制项

对于 challenger 相对 incumbent：

输出：

```text
TP retained
TP newly recovered
TP lost

FP retained
FP removed
FP newly introduced
```

并输出 event-level diff：

```text
outputs/challenger_70/.../event_diff.csv
```

每一个 F1 增益必须可以追溯。

---

# 29. 不允许根据 outer event_diff 再调同一个 challenger

完整 outer 结果产生后：

> 该 experiment 立即冻结。

如果根据它发现新 hypothesis：

创建：

```text
experiment N+1
```

并明确：

```text
derived_from = experiment N
```

不得偷偷修改 N 后重跑并把结果当独立验证。

承认这是开发集上的 repeated experimentation。

---

# 30. Multiple-comparison discipline

因为这 10 天会反复实验，必须记录所有正式 outer runs：

```text
outputs/challenger_70/experiment_registry.jsonl
```

每行记录：

```text
experiment_id
timestamp
git_sha
hypothesis
outer_f1
tp
fp
status
```

不允许只保留最好的 run。

---

# 31. 推荐实验顺序

严格按以下顺序：

```text
E00
Incumbent reproduction

E01
Error Atlas only

E02
Context-v2 session-relative deterministic features

E03
Context-v2 + hard-negative weighting

E04
short-meal micro candidate oracle

E05
仅当 E04 通过 candidate gate：
micro-v2 + best verifier/context

E06
best A/B/C combination
```

不要同时启动十几个方向。

---

# 32. E00 是必须项

第一轮必须重新验证 incumbent：

```text
F1 = 0.6514285714
TP = 114
eligible = 153
pred = 197
```

如果无法复现：

> 停止所有优化。

先解决 reproducibility。

不得在 baseline 不稳定的情况下研究 challenger。

---

# 33. E02 Context-v2 成功标准

优先目标：

```text
FP reduction
```

理想情况：

```text
TP >= 112
FP <= 65
```

或者任何能够明显提高 aggregate F1 的组合。

特别记录：

```text
83 incumbent FP
有多少被移除？

移除 FP 时
误杀多少 incumbent TP？
```

---

# 34. E03 Hard-negative 成功标准

必须证明：

```text
FP reduction
```

不是仅靠提高 threshold。

与一个简单 threshold-only control 比较。

如果 hard-negative 的收益完全可以由改变 event threshold 得到：

> 不声称 hard-negative 有独立价值。

---

# 35. E04 Micro candidate oracle

不训练最终 verifier 前，先输出：

```text
incumbent 15s

vs

5s
10s
30s
union combinations
```

比较：

```text
short GT reachable
overall GT reachable
candidate volume
```

只选择一个最有价值的增量方案进入 E05。

---

# 36. Threshold 与 policy tuning

所有：

```text
verifier threshold
admission threshold
subject cap
NMS IoU
blend weight
event budget
```

必须继续由 outer-train inner OOF 选择。

禁止为 Context-v2 直接使用在五个 outer validation 上表现最好的全局 threshold。

---

# 37. Subject cap 特别审计

由于 FP 可能集中于某些 subject，禁止直接：

```text
“看到某 subject FP 多，所以 cap 调低”
```

subject cap grid 必须预注册并 inner-selected。

同时输出：

```text
TP lost due to cap
FP removed due to cap
```

判断 cap 是否真正有效。

---

# 38. 时间先验暂不扩大

当前 macro 第 63 维存在冻结 time prior。

本轮不要新增：

```text
meal time handcrafted prior
breakfast/lunch/dinner hard rule
```

除非 strict inner OOF 独立证明收益。

尤其禁止根据 outer GT 的 meal-time histogram 构造 prior。

---

# 39. 可解释性要求

Context-v2 每个最终保留 feature 必须能够用一句话解释。

例如：

```text
session_macro_percentile:
候选局部 macro evidence 在本会话全部 macro evidence 中的百分位。
```

不要加入无法解释的几十个组合式 feature。

---

# 40. 测试要求

至少新增：

```text
test_context_v1_unchanged
test_context_v2_schema
test_context_v2_no_labels
test_context_v2_session_isolation
test_context_v2_gap_safe

test_hard_negative_outer_subject_isolation

test_micro_v1_unchanged
test_micro_v2_gap_safe
test_micro_v2_no_cross_session

test_incumbent_reproduction
```

如果修改 runner/crossfit：

必须增加 protocol tests。

---

# 41. Incumbent parity 是硬门

优化 branch 上：

```text
context=v1
micro=v1
```

必须仍然复现 incumbent。

如果由于新增代码导致：

```text
114/153/197
```

发生任何变化：

> 视为 regression。

立即修复。

---

# 42. 禁止顺手重构

本轮严禁：

```text
移动目录
重命名大量模块
格式化整个 repo
升级依赖
改变 requirements
删除历史代码
清理 dist
修改 visual
修改 submission README
修改报告
改变 schema v1
```

如果发现技术债：

记录到：

```text
outputs/challenger_70/deferred_issues.md
```

不要顺手处理。

---

# 43. Branch 要求

创建独立研究 branch，例如：

```text
research/f1-070-challenger
```

不得直接在 main 开发。

每个重要实验使用独立 commit。

建议：

```text
audit/error-atlas
exp/context-v2
exp/hard-negative
exp/micro-v2
```

保持可 cherry-pick / revert。

---

# 44. 缓存

允许新增：

```text
cache/challenger_70/
```

但：

* cache 不得成为唯一 evidence；
* cache key 必须包含 feature version/config；
* 不得让 v2 cache 污染 v1；
* 不得覆盖 incumbent cache contract。

---

# 45. 输出目录

所有本轮实验默认写：

```text
outputs/challenger_70/
```

禁止覆盖：

```text
outputs/crossfit/context_v1_summary.json
```

或任何 incumbent evidence。

---

# 46. 最终 challenger evidence

若出现 best challenger，生成：

```text
outputs/challenger_70/best/
    summary.json
    fold_0.json
    fold_1.json
    fold_2.json
    fold_3.json
    fold_4.json

    event_diff.csv

    fp_removed.csv
    fp_added.csv
    tp_recovered.csv
    tp_lost.csv

    short_meal_analysis.json
    fold_stability.json

    feature_schema.json
    experiment_plan.json
    leakage_audit.md
    comparison.md
```

---

# 47. comparison.md 必须回答

```text
Incumbent:
114 / 153 / 197
F1 .6514285714

Challenger:
TP / 153 / pred
F1 ...

Delta F1
Delta TP
Delta FP

short recall delta

fold-by-fold delta

candidate recall delta

candidate volume delta

inner→outer gap

new feature count

runtime delta
```

以及：

> 为什么它提高了 F1？

不能只写：

> “模型性能更好”。

---

# 48. Runtime gate

比赛 submission 是 CPU。

challenger 最终 inference：

```text
不得要求 GPU
```

允许 CPU runtime 增加，但必须测量。

如果单会话从当前约 18 s 增长：

```text
> 2x
```

必须单独报告并解释。

如果只是 Context-v2 deterministic features，原则上不应出现巨大 runtime 增长。

---

# 49. Memory gate

对真实大 TXT 进行 smoke test。

不能因为新 feature：

```text
复制整段 raw array 多次
建立巨大 NxN matrix
保留所有窗口 Python object
```

导致明显内存退化。

---

# 50. Promotion 禁止自动执行

即使出现：

```text
F1 = .72
```

Agent 也不得自动：

```text
release_event_stack.py
更新 event_stack_incumbent.json
覆盖 models/
rebuild dist/submission
```

必须停下并向用户报告。

Promotion 是单独的人类批准步骤。

---

# 51. 最终汇报格式

完成研究后，给用户一份简洁但完整的报告：

## A. Baseline

```text
git SHA
run key
F1
TP/eligible/pred
```

## B. Error Atlas

```text
39 FN 分层
19 short FN 分层
83 FP taxonomy
```

## C. Experiments

表格：

```text
ID | hypothesis | F1 | TP | FP | short recall | status
```

必须列出失败实验。

## D. Best challenger

```text
aggregate F1
folds
TP
FP
FN
PPV
recall
```

## E. Why it improved

给出 event-level accounting。

## F. Leakage audit

明确回答：

```text
outer-val 是否进入训练？
outer-val 是否进入 normalization？
outer-val 是否进入 threshold selection？
outer-val 是否进入 hard-negative mining？
```

全部必须为 NO。

## G. Regression status

```text
incumbent parity
unit
integration
release tests
```

## H. Recommendation

只能给：

```text
DO NOT PROMOTE
```

或者：

```text
READY FOR HUMAN PROMOTION REVIEW
```

Agent 不得自行 promotion。

---

# 52. 研究优先级

时间有限时严格遵循：

```text
1. E00 incumbent reproduction
2. E01 Error Atlas
3. E02 Context-v2
4. E03 hard-negative verifier
5. E04 short candidate oracle
6. E05 micro-v2
7. E06 combination
```

如果 Context-v2 已达到可靠 `.70+`：

> 不需要为了“把所有想法都做完”继续扩大改动。

进入验证阶段。

---

# 53. Stop conditions

出现以下任一情况立即停止对应路线：

### Stop A

v1 parity 破坏。

### Stop B

发现 leakage。

### Stop C

inner evidence 无改善。

### Stop D

candidate recall 上升但 candidate volume 爆炸，最终 F1 不升。

### Stop E

新模型只提高某一个 outer fold，并显著伤害其余 folds。

### Stop F

收益完全来自修改 eligible/matcher。

### Stop G

为了 +0.005 F1 需要大规模改 canonical runtime。

---

# 54. 本轮最重要的原则

不要为了 0.70 把一个可信的 0.6514 变成一个不可信的 0.71。

本项目已经经历过：

```text
错误时间轴
跨折 bag 泄漏
复核器泄漏
乐观结果作废
```

因此：

> protocol correctness > headline F1

任何无法解释、无法复现、无法严格 subject-disjoint 验证的提升都视为零提升。

---

# 55. 期望的最佳结果

理想结果不是“训练了更复杂模型”。

理想结果类似：

```text
Incumbent
TP 114
FP 83
F1 .6514

Error Atlas:
发现主要 FP 集中于 session-relative 非主导候选，
short FN 中若干属于 micro candidate miss。

Context-v2:
通过 session-relative rank / robust local-global contrast
减少 15 个 FP，仅损失 1 TP。

Micro-v2:
恢复 5 个 short TP，引入 2 个 FP。

Combined:
TP 118
FP 69
pred 187
F1 ≈ .694

或更好：
TP >=118
FP <=66
F1 >=.700
```

这种结果有明确机制、可解释、可复现，也适合后续答辩。

---

# 56. 最后命令

现在开始：

1. 不修改任何文件；
2. 读取当前仓库；
3. 确认 git HEAD 与 incumbent；
4. 运行/验证 E00；
5. 建立 Error Atlas；
6. 输出 E01 报告；
7. 根据真实 error distribution 决定 Context-v2 的**最小** feature set；
8. 再开始实现 challenger。

不要一开始就写新模型。

不要修改 dist。

不要 promotion。

不要覆盖 incumbent。

目标是：

> 在保住可信的 `.6514285714` 的前提下，科学地寻找 `.70+`，而不是用 `.70+` 换掉可信度。
