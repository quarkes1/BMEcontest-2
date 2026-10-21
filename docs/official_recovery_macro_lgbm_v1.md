# MACRO-LGBM-V1 — 宏窗识别器的单因素替换

用户授权自主推进；边界、固定max、负样本清理与固定弱监督路线无可靠改善。下一项先检验宏窗分类器的表达/优化是否不足，不再在这些失败policy上调参。

## 第一门：廉价窗口探针

固定T24三个16train/8health；完整val联合窗口只在各train内做原每session负窗≤max(1,3×正窗)采样，seed20260901，旧标签/忽略区不变。固定同样63D（含现有时间先验）宏窗输入，唯一替换HGB为LGBM：沿用micro工厂参数300树、31叶、min_child100、lr .05、colsample .8、lambda5、n_jobs1，但class_weight=None以保持与宏窗基准相同的先验合同；seed20260910。不搜索、不追加其他树模型。

窗口评价固定所有health完整GT标签>=0窗口，full GT生成标签方式沿用LABEL-COVERAGE-V1；train完全不看health GT。比较每组与平均AP/BCE/AUC、逐人AP（单类人记录None）、抽样SHA与输入角色图。主门平均AP至少+.02且≥2/3严格提高。不生成事件F1；AP过门仅支持下一步严格嵌套事件验证，不意味着送测有改善。

## 通过才进入事件验证

T24 3×3 policy OOF，每producer独立四折窗口OOF，宏窗工厂A/HGB或L/LGBM，micro与116D Context-v1、旧eligible事件监督、LR/LGBM .25/.75不变。所有训练侧OOF只使用其fit主体；全候选/完整GT策略目标、原网格不变。先9份policy全部封存提交再health。基准必须精确复现89/62/71；T24 gate F1≥+.02、≥2/3提高、TP不减。通过才39主体254餐五折，以冻结A138/80/116严格配对，五折gate≥+.02、≥3/5提高、TP不减。验证实现需在实际event fit前登记精确接口与来源，并审查。

## 保护与预算

零外部依赖，CPU数值库8线程，LGBM本体n_jobs1，6次窗口fit；首门墙钟≤10min、RSS≤8GiB、证据≤16MiB、余量≥10GiB。不保存探针模型或特征缓存，只保存分数与身份；输入CSV/index、十份val、manifest、所有src/scripts与本文SHA前后冻结；正式及第一试测66份文件不变。新输出拒绝覆盖，异常INCOMPLETE。不开V7/outer；重复开发证据，官方63FN不能逐例归因。

## 实施与审查

1. `macro_lgbm_probe.py`与`tests/research/test_official_recovery_macro_lgbm.py`先RED→GREEN：工厂参数、同一采样/训练与评分隔离、完整AP门、未知/非法标签拒绝。
2. 一次Luna medium只读审查，严重问题一次修复；提交后6fit，源/保护验证、报告、提交。失败关闭该工厂合同；通过才按上述注册事件实现。

Pre-flight：load_pools的features已含时间先验，fit不可再次追加；训练用old、评价用new，labels>=0过滤只作用于评价定义，不得把health标签用于训练采样。样本抽样使用canonical sample_training_rows，而非另写等价rng函数。

实施：4项新测试RED→GREEN，含真实缺失值LGBM fit；相关45项通过。Luna medium一次审查无Critical/Important；Final: minor (deferred): 探针graph将训练主体直接记在score节点而非显式fit→score边，单独train/score及抽样行身份仍准确，但图命名不够直接。未运行前不推断AP或事件效果。

窗口结果：6 fit共17.51s，峰值1.05GiB；平均AP .334534358→.353412104（+.018877746，2/3提高），未达到+.02，原初筛保持失败。逐组A/L .325975/.322397、.408040/.413533、.269588/.324307；没有事件F1。

## Amendment M1 — 一次探索性事件验证（窗口评分后、事件训练前登记）

显式取代“仅窗口gate通过才允许事件验证”的工程预算限制，不修改原窗口结果，也不降低最终F1门。理由：AP是替代指标，+1.89个百分点与预设+2的差额不能等价为事件指标无效；固定新工厂未调参、两组方向一致，现有CPU可在数分钟完成严格配对事件验证。该决定受到已见T24窗口结果影响，必须明确标为探索性，不能假称预注册确认性实验。仅此固定工厂一次，不借此重启已失败路线。代价：增加一次重复T24开发选择与计算预算。

事件实施精确合同：`build_candidate_ledgers`增加可选零参数`macro_estimator_factory`，默认None保持原调用逐位一致；L在全部inner OOF与最终refit均提供factory(当前config.seed+2)，micro仍原config.seed+12。每训练分区先`prepare_train(...,'A')`采样一次，旧标签、候选全池不变。C=.1、融合.25固定，所有train验证图与原116D方法不变。A复用已冻结逐事件回放的原score与policy，不重新择优；L使用原完整GT choose_complete_policy。每个L policy graph显式连接3个split score节点。

同一入口支持T24/five；T24所有9个L分区fit并封存三份policy才health，五折所有15个L分区fit封存才outer。五折另须T24事件门真实通过、parent源/protected核验。原A源JSON/Data/cache SHA对照，加载及两阶段间eligibility t_acc/imu_valid指纹；输入代码/本文/窗口探针证据SHA封存。T24预计120窗口/24meta fit，五折200窗口/40meta fit，均≤2h、RSS≤12GiB、证据≤128MiB、余量≥10GiB；只保存分数/身份，不保存探针权重/特征缓存。所有结果明确重复开发，不修改正式release/第一试测。

实现`scripts/research/official_recovery/macro_lgbm_event.py`，增加factory透传/默认恒等、parent失败拒绝、gate完整人口测试；一位Luna medium只读审查后执行。通过事件两门才登记部署合同；失败关闭工厂，不按health修改树参数或AP门。

审查修复：窗口parent原仅检查结果数字/保护身份，不能证明来自本轮源；增加固定plan SHA `2b6c1d8fc33ac0462c1e6a516163af79c57289bb8a12a2f79d76d1639a457e3a`与代码24aa4667597483a9eea144bf14d5a2986a9c485a，逐项核对parent原source SHA。仅允许本文M1追加与已测试的crossfit factory注入两处预期源差异；新增事件代码由新plan完整冻结。`test_window_parent_requires_exact_reviewed_plan_and_live_source`观察拒绝缺失RED后修复GREEN。eligibility在数据加载前后及fit后核对，不只在fit前后取样。一次审查其余无严重问题；事件结果未产生。

M1事件结果（完成，失败）：0a4417a封存9份fit后统一评分。A89/62/71、F1=.572347267；L87/60/73、F1=.566775244（−.005572023，TP−2/FP−2）。逐组.594059→.616822、.555556→.574074、.568627→.500000，虽2/3提高但总体下降，不能按多数折单独晋级。120窗口/24meta fit，500.74秒，66保护与源/eligibility不变；56项相关测试通过，全套9d04ef3快照运行中。原AP门失败与M1事件失败均保留，不启动五折、无新EXE，关闭该独立替换工厂。
