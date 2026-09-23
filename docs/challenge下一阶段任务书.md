# BMEcontest-2 下一阶段 Challenge 任务书

## Data-first / Cache-first / Low-budget Research Sprint

### 0. 核心目标

本轮不以“尽快跑出一个更高 F1”为首要目标。

首要目标是：

> 用尽可能少的 Codex 配额、CPU 时间和完整训练次数，识别当前 incumbent 尚未利用的稳定数据信息，并据此只启动极少数有机制证据支持的 challenger。

当前正式 incumbent 保持不变：

* run key：`160afaf81debf1ee`
* strict subject-disjoint nested 5-fold development F1：`0.6514285714`
* TP / eligible truth / prediction：`114 / 153 / 197`

当前阶段事实：

* E00 baseline reproduction：完成
* E01 Error Atlas：完成
* E02 Context-v2：inner gate failed
* E03 hard-negative：inner gate failed
* E03.5 candidate-control audit：diagnostic only
* E04 verifier failure audit：diagnostic only
* 尚无新 challenger 完成完整 outer 五折

因此：

**不要重复证明 incumbent 已经证明过的事情。**

---

# 1. 本轮研究哲学

之前阶段重点是：

* correctness
* leakage prevention
* protocol reconstruction
* reproducibility
* release safety

这些基础设施已经建立。

从现在开始，研究流程改成三级预算：

## Tier 0 — Cheap diagnostic / proxy analysis

特点：

* 不训练或只做极小统计模型
* 优先只读 frozen cache
* 不重跑 raw pipeline
* 不重算 macro / micro / Context-v1
* 不重建 dist
* 不做 clean-room
* 不做完整 parity
* 不跑 outer

目标：

快速判断 hypothesis 是否有数据支持。

---

## Tier 1 — Inner-only challenger

只有 Tier 0 有明确机制证据时允许。

特点：

* 复用 frozen candidate / feature / OOF cache
* 只重训发生变化的模块
* 只跑 inner OOF
* 不跑 full outer
* 不修改 incumbent

---

## Tier 2 — Full outer

只有：

* Tier 1 明显通过 gate
* mechanism 与预期一致
* 多 fold 稳定
* 无 leakage

才允许人工决定是否运行。

默认：

**本任务不授权 Tier 2。**

---

# 2. 本轮最高优先级原则

每开始一个实验前必须先回答：

> 这个问题能否只用已有 cache / OOF prediction / candidate ledger 回答？

如果答案是“能”：

**禁止重跑完整 pipeline。**

---

# 3. 不应重复执行的工作

除非发现具体 bug，否则本轮默认跳过：

* incumbent full reproduction
* raw parser 全数据重放
* macro feature 全量重提
* micro feature 全量重提
* Context-v1 全量重提
* sealed outer full replay
* dist rebuild
* submission rebuild
* clean-machine validation
* 350+ test full suite
* artifact hash 全目录重复核验
* README/report 更新
* release manifest 更新

这些工作只在：

1. 真正修改对应代码；
2. promotion 前；
3. 怀疑 regression；

时重新执行。

---

# 4. 仍然不能省略的关键检查

即使是低成本实验，也不能跳过：

* subject isolation
* fold identity
* OOF / in-sample 区分
* eligible truth identity
* matcher identity
* candidate provenance identity
* no outer-label training
* no outer-derived feature selection
* no train/eval row overlap

这些属于协议正确性，不属于“重复验证”。

---

# 5. 建议建立 frozen research cache

若现有数据已经存在，不重复创建。

如果缺少统一研究缓存，可一次性生成：

`outputs/challenger_70/cache/`

建议内容：

* `fold_assignments.*`
* `eligible_truth.*`
* `candidate_ledger.*`
* `verifier_features_v1.*`
* `base_oof_scores.*`
* `incumbent_event_outputs.*`
* `truth_event_metadata.*`
* `candidate_to_truth_links.*`
* `cache_manifest.json`

格式优先：

* parquet
* npz
* feather

避免重复解析大规模 TSV。

---

# 6. Cache invariant

research cache 必须记录：

* source commit SHA
* incumbent run key
* fold assignment checksum
* feature schema version
* candidate policy version
* eligible denominator checksum
* matcher version

只要这些未变：

后续 Tier 0 / Tier 1 可以直接使用 cache。

不要每个实验重新验证整个项目。

---

# 7. 本轮总任务分成两个并行方向

本轮只做：

## Track A

Cheap E04.1 — LR/LGBM disagreement direction

## Track B

Data Characterization Sprint

两者都完成后，才决定下一位 challenger。

本任务默认：

**不训练正式新模型。**

---

# 8. Track A — Cheap E04.1

## 8.1 目的

利用现有：

* `reachable_fn_candidates`
* `tp_reference`
* `fp_reference`
* frozen LR score
* frozen LGBM score
* frozen blend score

回答：

> LR/LGBM disagreement 是否包含稳定且有实际判别价值的方向信息？

不得：

* 重跑 raw data
* 重新训练 LR
* 重新训练 LGBM
* 重跑 Context-v1
* 调 threshold
* 产生新 F1
* 写 challenger result

---

# 9. Track A 最小输出

创建：

`outputs/challenger_70/data_sprint/e04_1/`

至少包含：

* `direction_counts.json`
* `direction_by_fold.json`
* `report.md`

只统计：

`LR - LGBM >= 0.20`

与：

`LGBM - LR >= 0.20`

对：

* matched TP
* final FP
* reachable FN

分别计数。

---

# 10. Track A 不要过度分析

不要再：

* 设计更多 GT-aware taxonomy
* 聚类几十种 subgroup
* 跑 UMAP 大搜索
* 做复杂 statistical modeling
* 发明新的 outer-derived features

只回答：

1. disagreement 哪个方向更多？
2. TP / FP / FN 中方向是否不同？
3. 多 fold 是否一致？
4. 是否值得将来做 adaptive fusion？

如果答案不明显：

立即停止 Track A。

---

# 11. Track B — Data Characterization Sprint

这是本轮重点。

目标：

> 找到当前 63D macro + 47D micro + 116D verifier 尚未充分表达的数据信息。

优先关注：

1. temporal scale
2. repetition / periodicity
3. raw temporal morphology
4. subject-relative behavior
5. dominant / non-dominant differences
6. fold weakness
7. FP behavioral modes

---

# 12. Track B.1 — Temporal Scale Audit

重点研究：

* 5 s
* 10 s
* 15 s
* 30 s
* 60 s

允许：

只提取最便宜的 proxy statistics。

不要立即构建完整 micro-v2。

对 TP / FN / FP 比较：

* activity magnitude
* ACC variance
* GYRO variance
* short-term energy
* zero-cross / sign-change proxy
* peak count
* burst count
* active fraction
* local contrast

重点问题：

> short FN 是否在 5–10 s 上具有明显结构，但在 15 s 被平均掉？

---

# 13. Temporal scale 的成功标准

不是看某个 feature 的 p-value。

而是看：

* TP vs FP separation
* FN 是否向 TP 靠近
* fold direction 是否一致

尤其寻找：

> incumbent FN 在某个尺度上变得明显 meal-like，而现有 15 s representation 没捕获。

如果没有：

不启动 multi-scale micro。

---

# 14. Track B.2 — Repetition / Periodicity Audit

不训练深度模型。

对局部 candidate window 计算轻量 proxy：

* autocorrelation first peak
* autocorrelation peak strength
* dominant spectral frequency
* spectral entropy
* burst interval mean
* burst interval variance
* burst count
* repetition regularity
* peak-to-peak interval statistics

只选择低成本、可解释指标。

---

# 15. Repetition audit 的核心问题

真正进食往往可能包含：

`reach → mouth → return`

的重复动作。

因此检查：

> TP 是否比 FP 表现出更稳定的重复模式？

以及：

> FN 是否具有 TP-like periodicity，但 incumbent summary features 没利用？

如果答案明显为 yes：

这可能成为新的 cheap representation track。

---

# 16. Track B.3 — Temporal Morphology Audit

不要直接上 TCN。

先判断 TCN 是否值得。

抽取固定长度的低频 downsampled temporal profile，例如：

* ACC magnitude
* gravity-aligned gyro magnitude
* macro probability stream
* micro probability stream

将每个 candidate 对齐到统一时间轴。

只做：

* mean waveform
* quantile band
* autocorrelation
* low-dimensional PCA

不要训练 classifier。

目标：

判断：

> TP / FN / FP 的时间形态是否肉眼或 PCA 上明显不同，但现有统计 feature 无法表示。

如果差异不明显：

不要上 sequence model。

---

# 17. Track B.4 — Subject-relative Audit

研究：

* per-subject activity median
* MAD
* energy baseline
* gyro baseline
* candidate percentile
* candidate/local baseline ratio

但只做 descriptive analysis。

重点检查：

> weak folds 是否主要由 subject baseline shift 导致？

不要直接重启 Context-v2。

如果发现明显 subject-shift：

下一阶段才考虑：

* fold-contained subject-relative normalization
* robust normalization

---

# 18. Track B.5 — Hand / Scenario Audit

如果现有 metadata 支持，分析：

* dominant-hand
* non-dominant-hand

以及其他已知 scenario。

比较：

* ACC energy
* gyro energy
* repetition
* event duration
* candidate score
* FN concentration
* FP concentration

重点：

> 当前 weak performance 是否由某一手别 / scenario 系统性驱动？

不要因为 outer subgroup 表现差就立刻做 subgroup-specific model。

只用于提出 hypothesis。

---

# 19. Track B.6 — Fold Weakness Audit

历史 fold F1：

* fold0 ≈ 0.552
* fold1 ≈ 0.779
* fold2 ≈ 0.500
* fold3 ≈ 0.676
* fold4 ≈ 0.691

重点分析 fold0 / fold2。

但禁止：

* fold-specific threshold
* fold-specific model
* fold-specific features

只比较数据特性：

* subject count
* event duration
* hand distribution
* activity baseline
* candidate volume
* meal count
* short meal proportion
* FP density
* sensor coverage
* missing/gap statistics

目标：

> 找到弱 fold 是否共享一个可泛化的数据机制。

---

# 20. Track B.7 — FP Behavioral Structure

不要只在 feature table 聚类。

优先从时间行为上判断 FP 是否有少数 dominant modes。

可做轻量 clustering，但只用于描述。

尝试识别：

* short burst
* long continuous movement
* periodic repetitive movement
* fragmented high activity
* low-frequency long movement
* device handling / placement-like pattern

如果一类 FP 占据大量剩余 FP：

记录为 potential target。

---

# 21. 不要为了“解释所有 FP”过度 mining

只寻找：

> 占比较大、跨 fold、明显稳定的模式。

如果发现 10 个 FP 模式，每个只有 2–3 个：

停止。

不要继续做 taxonomy。

---

# 22. 数据分析必须采用固定 population

尽可能固定使用：

* matched incumbent TP
* final incumbent FP
* reachable FN
* C0 FN

避免频繁重新定义样本集合。

报告中明确：

* population size
* fold counts
* sampling unit

---

# 23. 禁止的低效分析

本轮禁止：

* 100+ feature brute-force ranking
* 大规模 Optuna
* Bayesian search
* AutoML
* random feature generation
* exhaustive window-size grid
* 大量 classifier benchmark
* dozens of clustering algorithms
* deep learning architecture sweep

目标是机制发现，不是搜索。

---

# 24. Tier 0 cheap proxy experiment

如果某个数据特征表现出明显机制，可允许一个非常便宜的 proxy。

例如：

* 单特征 threshold
* 2–4 feature LogisticRegression
* simple rank correlation

但：

**必须只在 inner development hierarchy 内做。**

目的不是报 F1。

只判断：

> 这个信号是否具有预测价值？

---

# 25. Cheap proxy 不允许晋级 outer

任何 Tier 0 proxy 即使结果很好：

不能跑 outer。

只能用来决定：

是否值得写正式 Tier 1 challenger。

---

# 26. 本轮最希望发现的三类结果

## Outcome A — Temporal scale signal

例如：

> FN 在 5–10 s 有明显 meal-like activity，但 15 s 不明显。

则建议下一阶段：

**multi-scale micro-v2**

---

## Outcome B — Periodicity / morphology signal

例如：

> TP/FN 存在稳定重复结构，而 FP 没有。

则建议：

先做 cheap temporal morphology features；

若仍不足，再考虑：

**lightweight temporal encoder / TCN**

---

## Outcome C — Subject shift signal

例如：

> weak folds 主要来自 amplitude/baseline shift。

则建议：

**fold-contained robust normalization**

而不是深度模型。

---

# 27. Adaptive fusion 的位置

如果 Track A 显示：

* LR/LGBM disagreement direction 在 TP/FN/FP 中明显不同；
* 至少多个 fold 同方向；

则可以把 adaptive fusion 保留为低成本 Tier 1 候选。

但本轮不自动训练。

如果 Track A 没有稳定结构：

fusion 路线立即结束。

不要继续。

---

# 28. 下一位 challenger 只能有一个主机制

数据 sprint 完成后：

最多选择一个主要 challenger。

例如：

* adaptive fusion
* short-scale micro features
* periodicity features
* subject normalization

不要同时：

`fusion + micro-v2 + Context-v3 + TCN`

否则无法知道提升来自哪里。

---

# 29. Challenger selection priority

建议按：

1. 明显跨 fold 数据机制
2. implementation cost 低
3. inference cost 低
4. leakage risk 低
5. 可解释
6. 与 incumbent 正交

优先。

不要单纯按 proxy 分数最高选。

---

# 30. 本轮 Codex 配额约束

将本轮看作“低预算 sprint”。

尽量满足：

* 不重复 full extraction
* 不重复 model training
* 不重复 outer
* 不重复 large test suite
* 不重复 documentation rebuild

如果某项分析预计需要大量全数据计算：

先说明原因。

能 cache 就 cache。

---

# 31. 建议代码结构

新代码优先放：

`scripts/research/`

或项目现有等价 research 目录。

不要污染：

`src/pipeline`

除非未来 challenger 真正需要 promotion。

本轮数据分析代码不应改 canonical inference。

---

# 32. 输出目录

统一：

`outputs/challenger_70/data_sprint/`

建议：

* `e04_1/`
* `temporal_scale/`
* `periodicity/`
* `temporal_morphology/`
* `subject_relative/`
* `hand_scenario/`
* `fold_weakness/`
* `fp_structure/`
* `summary.md`
* `manifest.json`

---

# 33. summary.md 必须回答

最终只回答下面这些问题：

1. LR/LGBM disagreement 是否值得进入 adaptive fusion？
2. 是否存在新的 temporal-scale signal？
3. 是否存在 repetition / periodicity signal？
4. 是否存在明显 temporal morphology signal？
5. 是否存在 subject-relative shift？
6. fold0/fold2 是否存在共同机制？
7. FP 是否有明显 dominant behavior mode？
8. 哪一种机制最值得成为下一位 challenger？
9. 哪些方向应停止？
10. 下一位 challenger 是否需要重新提取 representation？

---

# 34. 不要输出新的 headline F1

本轮是研究 sprint。

不要报告：

“新方法 F1 = X”

除非后续另开正式 Tier 1 experiment。

当前正式 headline 始终保持：

`0.6514285714 / 114 / 153 / 197`

---

# 35. 不更新比赛交付物

本轮不要修改：

* report
* submission
* visual
* dist
* release
* production README headline
* official model

只允许在 research README / experiment log 中写阶段状态。

---

# 36. 什么时候才允许重新跑完整训练

只有以下情况：

### Case 1

发现明确的新 representation hypothesis。

例如：

> 5–10 s periodicity 在 FN/TP 与 FP 间稳定分离。

才允许构建：

micro-v2。

### Case 2

发现 subject normalization hypothesis。

才允许重新训练对应 verifier。

### Case 3

adaptive fusion cheap audit 明显支持。

才允许进行 fold-contained meta-fusion。

否则：

不要训练。

---

# 37. 未来 Tier 1 实验的 gate

下一位正式 challenger 至少要求：

* inner aggregate F1 明显优于 control
* 建议提升至少约 0.01–0.015
* 多 fold 同方向
* event accounting 与 hypothesis 一致
* TP/FP 改善可解释
* no leakage

否则不进入 outer。

---

# 38. Full outer 预算原则

未来完整 outer：

**非常昂贵。**

默认一个研究阶段最多运行：

1 个主要 challenger outer。

不要连续 outer：

* A
* B
* C
* D
* E

然后挑最好的。

这会逐渐把 outer 变成开发集。

---

# 39. 研究结束条件

本轮结束时必须选择一个状态：

`NO_NEW_SIGNAL_FOUND`

`ADAPTIVE_FUSION_CANDIDATE`

`TEMPORAL_SCALE_CANDIDATE`

`PERIODICITY_CANDIDATE`

`SUBJECT_NORMALIZATION_CANDIDATE`

`TEMPORAL_ENCODER_JUSTIFIED`

`MULTIPLE_SIGNALS_FOUND_NEEDS_HUMAN_SELECTION`

不得自行开始下一位正式 challenger。

等待人工审核。

---

# 40. 最重要的原则

我们已经证明：

Context-v1 能产生巨大提升，是因为它加入了新的 information set。

现在不要继续：

> “再做一点 Context”
> “再调一点权重”
> “再调一点 threshold”

而是寻找：

> 当前模型真正没有看到的稳定信息。

优先寻找：

* temporal scale
* temporal morphology
* repetition
* subject-relative behavior

本轮成功的定义不是 F1 上涨。

而是：

> 找到一个足够强、足够稳定、足够低成本，值得花下一次正式 inner / outer 预算的新机制。

效率优先。

Cache first.

Data first.

Mechanism first.

Outer last.
