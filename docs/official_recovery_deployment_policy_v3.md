# R3 — 完整GT最终决策与部署policy对齐

2026-10-22，新限时自主授权下登记。目标尽快产出第二份有明示风险的独立试测EXE；正式promotion不变。

## 与R2的区别

R2在当前完整GT策略A之上再偏召回，没有找回GT。第一EXE却仍为旧eligible policy聚合：final threshold=.6595272837359293、cap6、NMS.3、admission.2。当前A的T24阈值约.39–.44/cap8，五折median约.458。不能把两个policy合同混为一谈，也不能直接搬cap8阈值到cap6。

本项固定模型/分数/候选，固定第一EXE admission和融合.25。对照D为第一EXE冻结的final threshold；新C以完整GT/F1选最终阈值，新R以完整GT/F1.5选最终阈值，两档都在每fold train OOF上选，tie F1/PPV/更高threshold，不新增beta或网格。保留原A完整GT policy作为第三参考，但主配对为D→C或D→R。

源分数来自此前已经冻结、角色隔离的negative_clean_v1/_validation_v1四模型拟合，**不是**第一EXE全目标四模型的外层预测。outer micro threshold均.1，与第一EXE一致；部分inner producer曾train-only选到.5，必须记录，而非谎称每个producer配置完全一致。通过者只复制第一EXE四个精确权重字节，加新最终policy；不把诊断F1称最终EXE嵌套成绩。

## 冻结、试测资格与路径

复用relative_score_probe.py新增显式deploy_f1/deploy_f15，两档use_relative=False。读取首轮已验收bundle并核对consistent_refit_v1/summary.json全部SHA，原policy所有非final字段固定。source/protected66/图/完整GT分母不变；每个policy图显式原OOF父节点。policy seal/commit后评分。报告events.A为D reference、events.G为新policy，额外events.source_A为历史完整GT策略参考。

T24完整160GT，过门才39主体254GT五折。外部试测资格沿用R2：T24净TP≥3/五折≥5，新增FP≤3×新增TP+2，F1不低于D−.02，至少2/3或3/5折TP不减；不要求F1提高，若F1下降必须明示。若F1确有提高，只称重复开发政策诊断提高，非官方收益或正式promotion。父级须匹配deploy模式/ID并复算gate。不得用旧A和不同admission的收益冒充D→新policy。

每档最终threshold为这五个**train-only**新阈值median；admission一直cap6/.3/.2，故没有R2的跨admission聚合问题。不得再用outer回扫该median。两个门通过可直接独立试测copy，不重训相同权重。最终新bundle版本official_deployment_f1_policy_v3或official_deployment_f15_policy_v3；EXE目录dist/official_trials/deployment_f1_policy_v3或deployment_f15_policy_v3。原EXE/正式submission不变。

独立copy/builder/EXE-only验收沿用R2已写明的模型byte SHA、sourcebundle、API、中文路径/无开发PATH/neutralcwd/真实双文件CSV byte parity/空错输入保护、最新冻结全套要求。无邮件、无自动正式提交。程序最终官方F1未知，明确用户需送检。

## 实施与预算

1. 默认relative/R2行为不变；TDD新合同固定deployment admission+reference policy、首bundle身份和源SHA校验、模式父级拒混。新stage单次Luna medium只读review（此前R2严重测试缺口已修复）；严重项一次RED→GREEN、小项记录。
2. 先两档T24各自sealcommit→score，再通过档五折各自sealcommit→score；源/spec在两stage间不变，结果均完成才编辑本文。
3. 通过后新增独立export代码（不改旧builder约束），冻结全套与真实EXE验收后交付。未过门不继续更低阈值搜索。

每phase≤10min/RSS12GiB/证据16MiB/余量10GiB，零模型fit、无新权重/特征缓存；source报告包含完整微窗选择列表和D policy来源SHA。第一版数据训练/模型没有被视为“泄漏”，这里是决策目标不一致，不声称修复后官方必然达.7。

Ruling：对准实际已送EXE的政策参考，而不是继续以已经更换政策的A作为唯一送检基准——避免把算法创新门误当部署政策修复门；代价是该参考来自已使用全队数据的既有policy，只能作工程开发诊断，不称独立确认。

## 训练/评分前实施ledger

- 模式/首bundle加载器缺失RED→GREEN；实际首bundle临时副本SHA校验、两部署模式完整seal→score、D与source_A分离、模式父级复算、输入/保护/封存漂移拒绝已覆盖。45相关项通过（5.24s）。一处测试误数8文件+summary为10，纠正为9；未放宽实际SHA集合校验。
- Luna medium单次只读review指出Important：R3 summary沿用R2 evidence_kind，易误读。新增断言先失败，修正为FROZEN_MODEL_DEPLOYMENT_POLICY_DEVELOPMENT_TRIAL_ONLY，actual_exe_cv=False并明确旧参考policy来自先前全队聚合、非outer-isolated；随后45项通过。无Minor，不重复review。
- Final: Ruling：review未判断实际EXE效果/验收及最新全套状态——这些后续步骤分别验证，代价是研究代码审查不能证明程序提高。当前无新EXE/当前全套通过声明。
