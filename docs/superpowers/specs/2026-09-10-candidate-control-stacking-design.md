# 候选控制、OOF Stacking 与模型晋级设计

日期：2026-09-10

## 1. 目标与边界

当前无泄漏五折开发证据为 F1 `0.478632`（TP/true/pred=`112/153/315`），短餐最终
召回 `20/39=0.512821`，union candidate recall `0.947712`，但原始 union candidates
达到 `3413`。本阶段首先降低候选噪声并改善验证器排序，目标是让严格 subject-disjoint
OOF 选择后的五折 aggregate F1 高于 `0.478632`，最终持续迭代到 `0.65+`。

本规范只覆盖目标域候选控制、模型聚合、模型固化和发布包同步。FD-I/FD-II 外部数据
表征预训练属于后续独立规范；本阶段 `external_fd_weight_grid` 仍必须为 `(0.0,)`。

## 2. 不可违反的评估约束

- 外层 validation 的 truths、labels、slice labels 和最终指标不得参与候选控制器、模型
  权重、阈值、正则化或每被试预算选择。
- 所有候选控制和 stacking 超参数只可使用对应 outer-train 上的 subject-disjoint OOF
  预测选择；同一被试不能同时进入拟合与评分集合。
- 保持冻结的 153-event eligible 口径和现有 macro/micro session-ID 对齐逻辑。
- 重复 outer-CV 结果属于开发证据，不得表述为最终 untouched 泛化分数。
- fold 0 只能用于运行时和几何诊断，不能据其结果手动修改参数。

## 3. 方案比较与选择

### 组内 0.652 结果的协议审计

组内仓库公开结果为 F1 `0.652`（TP/true/pred=`162/276/221`），其主要 FP 清理来自
双 LightGBM 窗分数、`min_duration=180s`、`fuse=600s` 和 13 维事件级 LightGBM。
这些机制可借鉴，但数值不可与本项目 `112/153/315` 直接比较：其窗分数先在全体41人上
生成 LOSO OOF，随后另做事件级5折，因此 outer-train 事件特征的窗模型可能训练过
outer-test 被试。对方 `64_leak_audit.py` 也将其标记为 L2 间接信息共享。这里仅移植
事件级树模型、时长和前后上下文信号；所有分数重新在本项目每个 outer fold 内交叉拟合。

### 方案 A：仅提高 micro threshold

实现简单且会减少候选，但当前四折已在训练内选择 `0.10`，单独收紧阈值容易损失短餐
召回，也不能改善 macro 与 micro 候选的相对排序。不采用为主方案，仅保留为控制器的
一个训练内候选参数。

### 方案 B：验证器之后硬截断

对每个被试只保留 top-K 可以直接控制输出数量，但无法减少进入验证器的噪声，且容易把
一天中多个真实进食事件互相挤掉。不单独采用；top-K 只作为最终解码策略中的安全上限。

### 方案 C：候选准入控制器 + OOF stacking（采用）

在原始 union 与最终事件验证之间增加轻量准入阶段。准入控制器利用已有 56 维多尺度
特征和来源/重叠信息，对高度重叠候选做确定性抑制，再以 outer-train OOF 分数执行
阈值和每被试预算。最终验证分数由 LogisticRegression 与小型 LightGBM 两种验证器
在 OOF 上进行离散权重融合。该方案同时解决候选数量和单一线性验证器容量不足，且能
复用现有特征，不增加新的原始数据缓存。

## 4. 候选准入控制器

新增 `src/pipeline/candidate_control.py`，只承担纯候选排序与选择，不读取文件或标签。
公共接口包括：

- `CandidateAdmissionConfig`：NMS IoU、准入概率阈值、每被试最大候选数。
- `suppress_overlapping_candidates(candidates, scores, iou_threshold)`：按
  `(-score, sid, start_ms, end_ms, source flags)` 的稳定顺序做同 session NMS。
- `admit_candidates(candidates, scores, groups, config)`：阈值过滤后按被试预算保留，
  输出保持规范化时间顺序。
- `select_candidate_admission(...)`：只消费训练 OOF 分数、训练候选标签/事件、训练
  truths 和 groups；在预注册有限网格上选择配置。

选择排序为：先满足 candidate recall `>=0.88`，再最大化训练 OOF 事件 F1，再最小化
admitted candidate/truth 比率，最后用配置字典序打破平局。如果没有配置达到 0.88，
选择 recall 最高者，再按 F1、较少候选和字典序排序。不得用 outer truth 数决定预算。

保留两个诊断计数：`raw_union_candidate_count` 和 `admitted_candidate_count`。后续候选
体积 gate 以 admitted candidates 为准，同时继续报告 raw union，防止通过改名隐藏爆炸。

## 5. OOF 模型聚合

现有 56 维特征保持固定，训练两个互补验证器：

- 现有 median-imputed L2 LogisticRegression；
- 受限深度、单线程、固定 seed 的 LightGBM candidate verifier。

两者必须使用完全相同的 subject-disjoint splits 产生一行一次的 OOF 概率。融合分数为
`w * logistic + (1-w) * lightgbm`，权重网格固定为 `(0.0, 0.25, 0.5, 0.75, 1.0)`。
每个权重下先在训练 OOF 上选择准入配置，再在 admitted candidates 上选择现有事件
阈值与 subject cap；最终以事件 F1、PPV、较少 admitted candidates、较简单权重顺序
选择唯一策略。所有模型随后在完整 outer-train 上重训，outer validation 每种模型只
评分一次。

禁止简单平均五个 outer-fold 模型来评估其训练过的被试。部署模型聚合只在全数据训练
阶段发生，不回写 outer-CV 分数。

## 6. 配置、结果与缓存身份

`RunConfig` 增加显式、可序列化的候选控制和 verifier ensemble 网格；默认关闭，保证
`micro_enabled=False` 以及当前 multiscale 路径结果不变。预测缓存 schema 再升级，身份
包含候选控制配置、LightGBM verifier 参数和融合权重网格。

`FoldResult` 增加选定融合权重、选定准入配置、raw/admitted 候选计数以及两路 OOF/外层
评分耗时。旧 JSON 只可向后读取，不可命中新 schema 的结果缓存。

## 7. 成功门槛与迭代纪律

一次实验只有在预注册五折完整运行成功后才可判定。相对当前 `0.478632`，任意严格
aggregate F1 提升都触发“模型晋级流程”；但成为 README 推荐默认还需同时满足：

- aggregate F1 高于 `0.478632`；最终项目目标仍为 `>=0.65`；
- aggregate PPV 不低于 `0.355556`；
- short-meal final recall 不低于 `20/39`；
- admitted candidates 不超过 `4 * 153 = 612`；
- 五折 wall clock（不含一次性特征提取）不超过 600 秒。

如果 F1 提升但其他默认门槛失败，仍固化为“改进但未推荐”的版本，README 明确失败项，
不得静默覆盖当前推荐配置。

## 8. 模型固化与 dist/ 原子晋级

每次合法 F1 提升必须在同一提交周期完成以下全部动作，否则不算成功更新：

1. 将五折训练模型、imputer、选定策略、特征 schema 和完整 RunConfig 写入
   `models/event_stack/<run_key>/`；使用原子临时目录后 rename，禁止半包。
2. 生成 `manifest.json`，包含 Git SHA、run/experiment key、训练数据指纹、模型文件
   SHA-256、Python/LightGBM/sklearn 版本、五折计数和开发证据限定。
3. 用全目标域重新训练部署 ensemble；其产物与 outer-CV 模型分开标识，不能用于回算
   五折指标。
4. 重建 `dist/event_stack/`，包含推理入口、所需源码、模型、策略与 manifest；运行无
   训练数据的 CPU smoke test，并验证同一固定 fixture 在仓库入口与 dist 入口输出一致。
5. 同步更新根 README、`dist/README.md` 和 `docs/三阶段重构设计.md` 的架构、指标、
   复现命令、限制与当前推荐状态。
6. 只保留当前最佳正式模型、当前 dist 副本和紧凑历史 JSON。失败实验模型、临时导出、
   smoke 缓存、`__pycache__` 与被新 run_key 取代的非最佳模型在路径核验后删除，并报告
   释放空间；原始数据、session 缓存和当前20个 production micro缓存不得删除。

`dist/` 更新必须来自注册的打包命令，不允许手工复制遗漏依赖。模型晋级脚本失败时，
保留旧 dist 完整可用，不能部分覆盖。

### 设备选择契约

部署入口必须接受 `--device auto|cpu|gpu|cuda`，其中 `gpu` 是 `cuda` 的别名。
`cpu` 强制所有组件走 CPU；`gpu/cuda` 在 CUDA 不可用或模型不含 CUDA-capable 组件时
给出明确错误，不能假装加速；`auto` 在包内存在 GPU 组件且 CUDA 可用时选择 CUDA，
否则回退 CPU，并在 stderr/manifest runtime report 中写明实际设备。当前 sklearn 和
LightGBM 树组件仍在 CPU；后续 Transformer 可在相同接口下使用 CUDA。

同一 bundle 和固定输入的 CPU/GPU 分数绝对误差必须 `<=1e-5`，最终事件几何必须完全
一致。若未来 GPU 浮点误差跨越阈值，打包测试必须失败，而不是放宽事件一致性要求。

## 9. 测试与审查

- 纯单元测试覆盖稳定 NMS、同分排序、每被试预算、无候选、单类候选和准入选择平局。
- 集成测试记录每个 OOF 行只被未见其 subject 的两个验证器各评分一次。
- 外层 truth/label 变更不得改变融合权重、准入配置、verifier C、事件阈值或 cap。
- legacy macro-only、现有 multiscale-disabled-controller 路径必须逐字段保持原结果。
- artifact round-trip、校验和、原子失败恢复、dist smoke 和仓库/dist 一致性必须自动化。
- dist 测试覆盖强制 CPU、无 CUDA 的 auto 回退、无 GPU 组件时强制 gpu 的明确错误；
  有真实 CUDA 环境时额外运行 CPU/GPU 分数容差与事件几何一致性测试。
- 每个实现任务由独立审查者检查；按用户要求，复杂 runner/模型/发布集成使用 Terra，
  边界单元测试、清理和文档任务使用 Luna，不使用其他子代理模型。

## 10. 外部数据后续门槛

只有候选控制 + stacking 完成并得到新的目标域基线后，才启动 FD-I/FD-II 独立设计。
外部阶段优先自监督 IMU 表征预训练，不直接混合事件标签；必须保留随机初始化对照、
`external_weight=0` 对照和目标域内层选择，并遵守 `CC BY-NC-ND 4.0` 等适用许可。
