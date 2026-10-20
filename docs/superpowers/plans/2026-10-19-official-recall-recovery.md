# 官方反馈后的召回恢复实施计划

> Execution: superpowers:executing-plans；原地 main 执行，采用测试先行、逐阶段证据与提交。

**Goal:** 在10月30日前寻找实际改善官方事件检测的方案，尽早提供最多两个独立、自包含送测EXE，不提前替换正式交付。

**Architecture:** 先分离评价范围、受试者/会话匹配和部署限制，再以训练域内部证据选择窄挑战方案。诊断不训练、不调整阈值；新的完整标签协议与历史153餐协议并列，禁止追改历史发布成绩。

**Tech Stack:** Python、pytest、现有NumPy/LightGBM/sklearn和官方Windows推理入口。

**Spec:** 本文“已批准需求”及用户2026-10-19送测讨论为本阶段依据。

## 已批准需求

- 官方测试反馈：TP42 / FP26 / FN70，GT112、预测68，F1=0.4666666667；由用户提供，未获得逐餐结果或测试原始数据。
- 可再次申请程序测试，反馈时间不保证；阶段送测仅交独立EXE，不附源码、报告或可视化。
- 正式材料10月30日前提交且仅一次；研究不能依赖官方及时反馈。
- 基准正式模型仍为160afaf81debf1ee，本地历史嵌套五折F1=0.6514285714。
- 保护现有模型、正式提交包、已批准报告及用户README；失败变体不进入正式交付。
- 不代发邮件或上传；构建和测试授权不等于对外提交授权。

## Global Constraints

- 只在任务所属 scripts/research/official_recovery、tests/research 和 outputs/official_recovery 中新增审计内容。
- 同一批封存预测首先拆开比较 session匹配/subject匹配、eligible标签/完整有效标签；完整有效标签不冒充官方隐藏分母。
- 使用官方ID规范：去除全部空白并大写；规范化冲突或主体跨折重叠必须失败，不静默合并。
- 完整标签保留没有候选、没有宏窗、跨文件和无记录覆盖的餐，分别标记；不能为提高指标删掉困难餐。
- 仅给真实经过外层留出的主体计算冻结预测诊断；其余有标签主体单独计数，不冒充已评估。
- 官方阈值采用整数几何严格IoU>0.25；同时回放旧>=规则，不能混用。
- 原始数据不复制；第一阶段仅读标签、索引、已有窗口索引及封存结果，不读取几十GB信号数组。
- 后续新候选的normalizer、模型、策略均须在对应训练主体内拟合；官方单行反馈不能直接选阈值。
- 审计结果不填写新正式outer F1、不修改release/dist；最多两个EXE需先通过各自实验与独立运行验证。

## Review Focus

- 同一GT在多个会话出现时只按明确身份去重；不把会话重复误作新餐。
- subject规范化后不允许跨折训练/验证身份混淆。
- 严格四分之一边界、并列IoU、一对一竞争及空预测必须可复现。
- 不能由日志中没有候选推断没有原始信号；未知覆盖保留未知。
- 既有TP/FP账本必须回放114/83/197；任何缺行、重复或源变化拒绝审计完成。

## Task 1: 冻结反馈与评价范围审计

**Files:** 新增scripts/research/official_recovery/scope_audit.py、tests/research/test_official_recovery_scope.py；输出outputs/official_recovery/scope_audit/。

**Interfaces:** `match_events(predictions, truths, strict=True)`返回一对一索引对；事件为(subject/session,start_ms,end_ms)。`audit_scope(root)`返回四种匹配/标签口径的计数、GT差异、逐人记录及保护哈希；`python -m scripts.research.official_recovery.scope_audit --output ...`只写新输出目录，拒绝覆盖。

- [ ] 先写严格边界、重复预测、跨主体、空集、ID冲突、主体跨折、输入重复缺行等失败测试，运行观察RED。
- [ ] 实现最小审计，复用原始元数据但不调用训练；记录所有实际输入SHA。
- [ ] 运行单元测试和真实审计，独立重算计数；确认模型/EXE/release SHA前后不变。
- [ ] README记录官方反馈与口径区别，审计report写清开发诊断属性；完成后提交。

## Task 2: 部署覆盖与预算归因

**Files:** 同目录新增coverage_audit.py及对应测试；输出outputs/official_recovery/coverage_audit/。

**Interfaces:** 以Task1的规范主体/标签和已有宏微窗口索引为输入；按真实记录覆盖计算重叠区间并集，不把索引时段跨度当连续有效时间。不训练、不挑策略。

- [ ] 测试重叠记录去重、断档、跨午夜、短文件、零窗口和多天同一主体。
- [ ] 统计完整标签的宏/微窗可达性、旧eligible排除原因、每主体有效记录天数与预算饱和；明确缓存与真实官方入口的差异。
- [ ] 根据证据选择下一条窄假设，另写固定实验配置；不从sealed-outer诊断直接报告改进成绩。

## 后续条件阶段

Task1/2完成后，优先在官方式完整标签的训练内OOF验证输入适配或时长相关预算；若不支持则转向评分稳定性/已有表示。不得自动认定cap或阈值为根因。每个候选保存TP/FP/FN及逐人收益，达到实际净收益才构建送测EXE。

阶段EXE位于独立versioned目录，模型与依赖内置，无参数官方CSV入口不变；已送测基准及dist/submission不覆盖。多版本分别运行，清楚命名，不把程序测试当正式材料提交。

## 执行记录

- 2026-10-19：用户已批准改善方向并要求开始执行，无需再次等待计划审批。
- Ruling: 沿用用户明确授权的main原地修改；不创建worktree。
- Ruling: 首轮先完成可复现口径诊断，不按预先猜测放宽cap或阈值。
- Ruling: PowerShell中维护本文件执行记录作为ledger，不重复创建辅助脚本/工作目录。
- Task 1: complete — 冻结预测完整标注口径133TP/64FP/121FN、254餐；权威输出scope_audit_confirmed，源SHA/保护校验通过。旧153餐成绩不变。
- Task 2: complete — 完成宏微窗口并集覆盖、多天预算和跨文件排除分析；这不是原始信号有效性审计。
- LABEL-COVERAGE-V1完成：12 fits，50.19秒；宏窗AP -.03204，微窗 -.00045，均1/3折提高，gate失败。不继续此窗口变体。
- Review: 修复重复session/规范化别名风险，真实索引重复0/别名0；增加2项RED→GREEN测试。旧实验原始证据不追写；未来运行补齐旧标签、抽样行和上游源哈希。
- Verification: 全套收集快照993 passed/6 skipped（不含之后7项新增测试）；当前23项召回恢复专项通过。
- Ruling: 下一实验保持窗口监督不变，只检验事件级完整餐次监督；不是失败窗口变体的升级。代价：仍受原候选可达性限制，不能据此宣称全流程已修复。
