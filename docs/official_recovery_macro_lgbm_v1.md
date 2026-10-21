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
