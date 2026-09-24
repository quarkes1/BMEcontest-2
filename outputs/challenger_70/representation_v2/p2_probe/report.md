# RV2-P2-001 — 目标数据表示 Probe 收口报告

状态：`P2_NO_SUPPORTED_BRANCH`。本报告仅是 outer-train 内的 candidate-level 开发证据；**没有新的 event F1、outer F1 或正式发布变更**。P3 按任务书 gate 停止。完整逐分区分数、逐受试者指标、模型权重与计时见本目录；本文件不替代机器可读账本。

## 固定比较与完整性

- 按原定 5 个 outer-train wrapper ×4 个 inner subject folds 完成 20/20 个 T/V 分区。正式评估复用每个分区的候选账本，需 200 次 macro/micro 基础拟合；此前两次候选计时 pilot 另消耗 20 次基础拟合。候选生产器在 T 内 OOF、对 V 用 T refit；micro threshold 只由 T 真值选出。V 真值仅在候选/特征生成后用于 probe 标签和指标。
- Q0：同架构、局部 encoder 随机冻结，训练时序 head；Q1：配对初始化、全分支 scratch 训练。两臂均固定 2,000 optimizer steps、相同训练候选、V 候选与 0.5 probe 阈值。总计 40 次目标分支拟合、80,000 步；另有 40 步合成吞吐校准。未训练 Q2/Q3、未运行 source FD fit。
- 20 个分区的 train/score 受试者集均不相交；V 候选键在各 wrapper 内无重复，4,987 条 candidate exposure 的标签/分数一一对应。每个 wrapper 重复使用部分同一批受试者，**4,987 不是独立事件数**。
- 4,987 条 V candidate 中 758 条无有效 IMU token，按注册规则以各自 T 候选阳性比例给分；未按 GT 删除。训练侧 14,785 条 candidate 中 12,636 条有效并进入分支拟合。
- FD-I 的 source/target ACC/GYRO 单位及换算不能从现有元数据核实；Q2/Q3 标为 `FD_DATA_REVIEW_REQUIRED`，不把未跑的臂写成失败。

## 结果

| outer-train wrapper | Q0 AP | Q1 AP | Q1−Q0 | Q0 log loss | Q1 log loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.35311 | 0.31167 | −0.04144 | 1.25901 | 1.73350 |
| 1 | 0.22304 | 0.24161 | +0.01857 | 1.26533 | 1.98086 |
| 2 | 0.36200 | 0.31680 | −0.04520 | 1.23052 | 2.93648 |
| 3 | 0.26720 | 0.24728 | −0.01992 | 1.24376 | 2.51176 |
| 4 | 0.23698 | 0.20460 | −0.03238 | 1.41603 | 1.94411 |
| 宏平均 | **0.28847** | **0.26439** | **−0.02407** | **1.28293** | **2.22134** |

预注册 Q1 gate 需 AP 至少比 Q0 高 0.02、至少 4/5 wrapper 正向且宏平均 log loss 不高超过 0.02。实际三项均不满足（AP −0.02407，1/5 正向，log loss +0.93841）。候选固定 0.5 阈值 F1 的宏平均为 Q0 0.33668、Q1 0.32918；这是 **candidate probe F1，不是进食事件 F1**。146 个 AP 有定义的 subject-wrapper 配对中，Q1 较好 63、较差 78、持平 5；这不是 146 个独立受试者。

Q1 的训练尾段损失很低而 held-out log loss 更差，与过拟合/置信度过高相容；该解释尚不能定位到 encoder、样本构造或 calibration 的单一原因。本次固定配置只支持“Q1 未带来稳定迁移增益”，不支持“raw IMU 无信息”或“全部时序方法失败”。

## 资源、修复与终止

正式 Q0/Q1 分两次命令运行，首分区约 218 秒、余下运行约 3,309 秒，合计约 **3,527 秒墙钟**；累计目标 CUDA 训练约 1,561 秒。两次候选 pilot 另约 23+104 秒，40 步合成吞吐校准约 8 秒；本轮已知总体墙钟约 3,662 秒（不含 P1 单测）。40 个 checkpoint 合计约 5.7 MB，未生成持久 token cache。均远低于 24 GPU 小时、48 墙钟小时、12 GiB cache 和 2 GiB checkpoint 限额。

收口时发现 checkpoint provenance 中的 NumPy 字符串使 `weights_only=True` 载入失败。后处理只将这 40 份本轮工件的 provenance 转为 JSON 基本类型；逐个限制身份、安全加载并验证权重张量逐项不变。未重训、未重算分数。修复前后哈希及身份见 `checkpoint_repair.json`。P2 分数账本的执行代码为 `4b9d2cce1cdcb4b94b8f5aefffc4dfd5a3f3eda0`；后处理修复代码为 `fff450fba9c65901fa5d4e8b259b861689586a1c`，不能将其混写成同一次训练代码。

P0 的 release/dist 保护快照前后完全一致；`outer_calls=0`。由于无支持臂，**不进入 P3，不触发确认 seed，不运行 P4 outer，也不变更 dist/release**。后续若要再试，须另行预注册一个窄假设；FD 迁移首先需补齐可信的物理单位合同。
