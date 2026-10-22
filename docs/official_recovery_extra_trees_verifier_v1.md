# EXTRA-TREES-VERIFIER-V1 — 随机子空间事件决策

2026-10-22登记。目标为第二份具有重复开发证据支持的独立送检EXE；用户已授权自主推进，不再询问例行审批。正式release、第一试测及dist/submission保持不变。

## 判断与单一改动

连续监督/候选合同改变均未稳定改善；完整监督只带来TP+4/FP+8。当前116D含相关Context特征，候选正例数量远小于窗口行数，逐步boosting可能形成高方差边界。假设随机阈值/特征子空间的多树平均能改善跨主体事件决策；这是待检验假设，不声称已确认失败机制。

相比另起Transformer或再扩阈值面，本项成本小、沿用已验证候选表示；相比上轮宏窗HGB/LGBM平均，本轮窗口及候选完全不换，改变的是最终非线性verifier。与旧E06融合、A9事件权重也不同：不学新融合、不改权重。已查现有任务记录未发现同一ExtraTrees合同；既有失败仍保留。

- 唯一学习改动：原LGBM事件verifier替换为SimpleImputer(median, keep_empty_features=True) + sklearn ExtraTreesClassifier。固定512 trees、max_depth=8、min_samples_leaf=3、max_features=.5、class_weight='balanced'、bootstrap=False、n_jobs=8、random_state=config.seed+23。无early stopping，不搜参数/seed。
- LR仍C=.1，概率仍.25LR+.75非线性；原宏窗HGB63D、micro LGBM47D、旧窗口/候选监督、116D Context-v1、canonical一次采样、四折subject window OOF、micro训练侧选阈与A完全相同。不得混入本轮失败完整监督、全负例加权或宏窗集成。
- 每次fit只接收当前train OOF候选及其标签，imputer/class balance/树均只fit train。score只predict。非线性组件保存名暂沿用lgbm用于统一研究CSV，plan明确actual estimator='ExtraTreesClassifier'，不得误称LightGBM结果。若获准打包，新增精确模型类型清单再修改独立导出器，不直接套旧模型身份。
- 训练侧choose_complete_policy、网格、strict IoU>.25及完整GT一次计数不变。三组A逐事件精确回放89/62/71、160GT。X为本轮分数；非线性模型/得分改变，不声称冻结incumbent score。

## 验证与工程门

复用macro_lgbm_event.py显式hypothesis='extra_trees'，新增纯工厂/fit函数extra_trees_verifier.py。记录该fit函数身份于plan，不通过monkeypatch全局runner替换。默认fit函数/其余hypothesis恒等；无推理真源修改。

T24三组16train/8health；每组三policy split，每producer四window OOF。所有9个policy fit先完成并seal/commit，再统一health。gate固定F1差≥.01、至少2/3严格提高、TP不减；FP增加时deltaTP>deltaFP。通过才完整39主体/254餐五折，所有15policy fit先seal/commit再统一outer；同门但3/5严格提高。父级须匹配hypothesis及通过门，原始A五折138/80/116。同属反复使用的开发集，不称独立确认/官方F1。

两门通过才登记独立全目标refit/export/EXE及新版本模型类型接口；最终交付冻结最新代码跑全套回归和EXE-only中文路径/无开发PATH/真实文件CSV byte parity/空错输入保护，官方原示例缺失明确跳过。失败关闭这一固定树合同，不在同一health调深度、树数、叶子或融合权重，不导出失败模型。

## 预算、文件与实施

- 输出outputs/official_recovery/extra_trees_verifier_v1_t24及_five，只留小型分数/策略/审计，不持久化失败树/特征。拒绝覆盖，异常INCOMPLETE。T24 120窗口/24meta、五折200窗口/40meta；ExtraTrees为原两个meta模型中的一份，不增加meta次数。
- 每cohort≤2h，RSS≤12GiB，证据≤128MiB，余量≥10GiB；数值库8线程，多树8线程，不嵌套多producer并行。CPU足够，不为GPU占用改模型。
- 代码/spec/Data/cache完整SHA、eligibility加载前后及fit后、66份正式/第一试测保护核查，原A采样及micro threshold逐fit精确断言。新阶段不依赖失败窗口AP父级；沿用validated frozen A provenance。
- TDD：固定工厂参数、真实fit的概率有限/范围、组件融合恒等、空score、单类非法；测试默认fit调用与ExtraTrees仅在新hypothesis启用；train-only fit及score不参与imputation。相关回归后Luna medium一次只读审查，严重项一次RED→GREEN修复，小项记录延期。先提交实施，再训练。
- 本文兼实施ledger，避免为小研究阶段散落额外计划/临时脚本。代码/本文在policy到评分期间冻结，全部完成再更新README和总记录。最终新EXE前最新冻结全套仍必须；之前376c369的1108项不冒充本轮验收。

Ruling：在前项失败后自主启动新的单一verifier合同——符合用户限时自主探索授权；代价是继续使用重复开发集的选择偏差，只有配对五折支持且明确标注开发证据后才建议独立送检，隐藏测试改善不作保证。

## 实施ledger

- 新测试先因缺少工厂/fit模块失败，再实现通过3项；连同窗口/元模型/OOF/采样相关回归50 passed（7.71s），git diff --check通过。无当前全套声明。
- Luna medium一次只读审查无Critical/Important。Final: minor (deferred)：fit_score注入路径尚无直接集成测试；参数工厂与真实fit已测试，policy/health两处调用经静态核对，不额外扩大本轮测试/调参。
- Final: Ruling：审查未判断尚不存在的T24/五折收益与延期打包行为——本轮先完成研究门，不能把代码通过当效果通过；两门失败不打包，成功则另注册并验证。代价是研究通过仍不保证隐藏测试改善。
