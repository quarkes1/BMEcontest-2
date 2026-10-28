# FUSION-POLICY-V5：最后窗口的训练侧融合校准

2026-10-26。用户再次要求尽快产出新EXE，最终提交预计28日、最晚29日。官方最佳仍为56TP/42FP/56FN、F1=.533333；CAP8-V4失败且关闭，不以放宽其门保证交付。

## 单一预注册家族

原四模型/116D/原始候选池不变，cap6、NMS.3、admission.2不变。不新增训练或外部数据。只用已有OOF中LR与LGBM分量，训练侧选择固定凸融合p=w*LR+(1-w)*LGBM，w∈{0,.25,.5,.75,1}，以及该w对应事件阈值。原版w=.25包含其中。不是此前max融合、学习型stacking或再次降低全局阈值。

每fold只在自己的训练侧OOF选择最优F1；tie依次PPV、最接近原w=.25、较小w。每w阈值沿用原choose_threshold固定规则。所有fold policy写完并SHA seal后才读取对应health/outer score。LR、LGBM分量分别验证有限0..1，原score必须等于.25LR+.75LGBM；score graph、主体角色及旧主档事件/计数精确重放。无测试集CSV、饭点/ID、GT匹配列表进入决策。

先T24三组160餐，过门才39主体五折254餐。门：aggregate F1至少+.005、至少2/3或3/5折不差；TP最多减少2、FP不得增加超过净新增TP的2倍（TP不增时FP不能增）。失败不修改网格或重扫outer，不将结果冒充官方或无偏CV。

最终部署：五fold各自选择的w取众数，tie优先离.25近、再较小w；最终threshold取五fold**同一个最终w**各自train-only阈值的median，不能混合不同w对应阈值。最终聚合不是被精确CV过的全目标EXE，actual_exe_cv=False，报告只称重复开发诊断。

## 交付

两门通过才复制官方最佳bundle；4joblib和feature_schema字节不变，policy仅blend_weight与threshold改变。版本official_fusion_policy_v5，models/event_stack/该版本/deployment；独立dist/official_trials/fusion_policy_v5/EatingDetection.exe。旧0.533333EXE/模型、正式submission/frontend不覆盖。送检只需EXE，无邮件/自动push。

精确用户dirty两路径与SHA保护沿用cap8；75保护、全部源分数/图/微窗来源/R3封存证据前后校验。新模块放scripts/research/official_recovery，测试放tests/research或release。构建复用已验证PyInstaller叶函数，单文件CPU，无Python安装；中文EXE-only/其他cwd/真实双文件CSV byte parity/空错输入保护，源码冻结后的完整pytest。

零fit，每probe≤10分钟/RSS12GiB/16MiB证据，编译≤30分钟、验收进程≤300秒、磁盘≥10GiB。不保留不需要的特征/训练缓存，不绕过删除策略拒绝。Luna medium一次只读审查，Important一次TDD修复，Minor登记。

## Ledger

- 收口：T24完整对照86/44/74、F1=.593103；新融合77/32/83、.572491，TP−9/FP−12，仅1/3组不差，失败，未进行五折或打包。2026-10-28停止所有实验。

- Ruling：CAP8失败后转向训练侧融合校准，不将cap8参数继续搜索——用户要求实际F1收益且最后机会，代价是增加一个有明确边界的家族；仍不能保证官方提高。
- Pre-flight：复用cap8身份/用户dirty guard及R3 source，不修改这些既有源文件以维持失败实验可追溯。融合probe→交付仅消费COMPLETE过门证据。
