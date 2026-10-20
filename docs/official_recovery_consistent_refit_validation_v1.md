# 一致性重训：同规模验证与独立送测

2026-10-20；衔接outputs/official_recovery/report.md。用户已授权自主研究，正式dist替换不是自动动作。

## 依据与目标

部署HGB正类基准.031810，五折.31–.35；最终meta用训练内窗口预测，而CV用OOF。
T24受控模拟中恢复抽样使F1 .362791→.427273（TP+8/FP−3，2/3折提高）；再恢复OOF使.427273→.548263（TP+24/FP+15，3/3折提高）。达到开展一次更广开发评估的工程门，但不是正式/官方提升，更未达到.70。

目标：验证同规模修复，尽快提供独立送测EXE。暂不增加网络、PPG、全局阈值放宽或跨文件事件拼接。

## 固定五折对照

1. 使用正式attestation的39个实际评估主体、原折归属。完整有效254餐为共同主分母，eligible153餐另列诊断；另外3个无同口径外层预测主体/17餐单列，不冒充已评价。
2. A/B都用对应outer-train的完整候选窗口，来源为五份val NPZ去重联合后按身份裁剪；不以抽样no_meal_train代替完整heldout候选范围。灰区/负样本可作为候选窗口，不等于窗口拟合样本。
3. A release-style：窗口fit全量label>=0，verifier特征来自同模型训练内预测。
4. B consistent-style：窗口fit该折既有sampled train角色，verifier特征来自train内部四折窗口OOF。
5. 两臂都固定旧窗口标签、旧eligible verifier标签、116D Context-v1及现有工厂。前两实验已分别隔离抽样/OOF，本轮只检验组合的同规模效应。
6. 每折policy/micro阈值/blend/C直接用该折封存train-only artifact，读前验证manifest SHA和主体身份。不使用全数据deployment policy评价单折。零阈值/cap/融合/seed搜索。
7. 最终macro seed原+2、micro+12、verifier+3/+23；B内部工厂与最终窗口seed对齐。imputer和任何模型不得fit heldout，保存完整祖先图。
8. 严格整数IoU>.25、一对一匹配；完整GT含无候选餐。全部五折完成才汇总，报告TP/FP/FN/F1、PPV/recall、逐折/逐人改善恶化、候选可达性、运行时间、训练类比例。

历史197条冻结预测的完整口径.589800仅为参考，**不是本次A**。重复outer分区仍为development evidence，非新独立测试；失败不得手工调参回跑同一outer。

## 门与预算

预注册B-A pooled F1>=.03，至少3/5折非劣且至少3折严格改善；明确TP/FP成本，不挑最好折。不达门则保存失败，不用官方反馈调参数。

CPU数值线程最多8，允许安全并行独立折；<=2小时、证据<=128MiB、磁盘剩余>=10GiB。不保存临时模型或特征缓存，不读原始IMU向量。实际输入（缓存/CSV/策略/session时间戳与有效性）前后SHA/数组指纹一致，正式模型/release/EXE保护不变。输出拒绝覆盖，含preflight异常均记录INCOMPLETE。

## 通过后的全数据重训

- 先冻结采样seed、OOF拓扑和policy聚合。五份val按窗口身份去重后**只抽样一次**，不拼接fold_train重复样本，也不叠加各折负采样。
- 每session最多3×正窗负样本、至少1个，灰区不fit窗口；记录正类比例、样本身份哈希。需测试与CV合同一致性，而非仅模型类型相同。
- verifier训练用subject互斥窗口OOF；最后宏微模型在同一抽样合同下refit。全数据训练F1不作泛化证据。
- 独立版本模型目录，不覆盖160afaf81debf1ee或release指针。policy只按预注册折间聚合，不再用全数据GT调阈值。

## 独立EXE与交付

验证支持后构建dist/official_trials/consistent_refit_v1/EatingDetection.exe，模型/依赖内置，官方CSV及无参数接口不变。阶段只交EXE，不重建正式源码/可视化提交包。

必须验证官方示例、真实数据Python/EXE一致性、无仓库/Python/PATH依赖运行、多文件同人、空结果/错误输入/输出格式/耗时/哈希。清理编译临时目录，只保留EXE和必要验证记录。不自动发邮件/上传。

dist/submission、dist/inference和前端正式模型不动；独立送测不等于正式材料提交或release晋级，正式替换仍保留关键节点确认。

## 执行记录

- 2026-10-20：用户批准推进；继续在已授权main原地工作，以本节作为执行ledger，不新建worktree或额外辅助脚本目录。
- Pre-flight：封存fold policy/config SHA → 固定融合与阈值；39主体索引 → 254/153统一分母；完整候选联合 → 每个fit前的train-only切片。七项关键测试已观察RED→GREEN。
- Ruling: 完整窗口联合仅作为评分候选宇宙，不作为监督训练角色；保留旧window_train对象，所有fit只接收train切片。代价：必须逐阶段检查身份，不能仅凭容器名称声称隔离。
- Ruling: 复用load_pools的身份适配入口，传入39主体而不是固定T24；它仅按给定身份裁剪缓存，未改变标签/特征。代价：旧函数命名不等于本轮人口，必须以冻结subject_fold为准。
- 预算：预计60次窗口fit、20次verifier fit，CPU8线程；逐折顺序运行避免重复内存，回归测试可并行。
- 首次执行在任何模型fit之前终止（0折完成）：旧loader保留`HNU 21013`和`hnu21033`原始拼写，冻结队列使用`HNU21013`/`HNU21033`，导致身份切片拒绝。输出保留为INCOMPLETE，不形成性能结果。
- Ruling: 训练角色在切片前使用已通过别名冲突检查的传感器索引映射规范ID；不修改窗口、标签、折归属或评估分母。新增真实数据适配回归测试RED→GREEN后，以`consistent_validation_v1_retry1`重启同一预注册实验。代价：若映射错误会影响隔离，故未知session仍硬失败，且继续保留fit祖先审计；不进行参数调整。
- 全分支只读审查未发现Critical/Important；运行结果和资源开销由实际执行验证，静态审查不替代实验。
- `retry1`被干净工作树门拦截（仍0次fit）：前一次失败证据尚未入库。先将两次INCOMPLETE证据固化，再以`retry2`启动；不覆盖失败记录，不改变实验设置。

## 同规模验证结论与后续实现

`retry2`完成，461.41秒、60次窗口fit/20次verifier fit；零策略搜索，输入指纹与57项正式文件保护通过。
完整254餐：A=90TP/34FP/164FN、F1=.476190；B=126TP/47FP/128FN、F1=.590164，差+.113973，4/5折改善；逐人29改善/6退步/4持平，达到独立送测门。
eligible153诊断：A=.519856，B=.687117；不能用该较高数字替代完整口径或称官方提升，亦未达到.70目标。

Ruling: 本轮B训练候选正例计数日志发生NumPy int8溢出，仅该诊断字段作废；labels未修改，模型fit、概率、事件与gate不使用该求和值，不重训/不改写原始证据。新增300正例计数RED→GREEN测试，后续改用count_nonzero。代价：本轮不能报告B训练候选正例的精确数量，但窗口类比例仍有效。

按已批准规范继续原地实现，以下是剩余任务的执行清单，不增加实验参数：

- [ ] `scripts/research/official_recovery/consistent_refit.py`与对应研究测试：验证五折gate和证据SHA；从唯一完整val联合构建全数据角色；复用sample_training_rows(seed=20260901, negative_ratio=3)，每流只采一次；验证无重复/灰区、原session合同、身份哈希；四折subject OOF→116D verifier；最终窗口refit，模型写独立`models/event_stack/official_consistent_refit_v1/deployment`。失败gate禁止fit，所有输入前后校验，训练内不报告F1。
- [ ] 训练预算：CPU8线程、<=2小时、证据<=128MiB、空间>=10GiB；只保存最终四个必要模型，不存中间权重/特征。policy使用既有canonical_deployment_policy等权mode/median（mode平票最小值），不看本轮成绩加权；模型seed与五折B保持一致。
- [ ] `scripts/research/official_recovery/trial_exe.py`与对应交付测试：固定只写`dist/official_trials/consistent_refit_v1`，禁止覆盖正式目录；嵌入独立bundle和非晋级trial身份，PyInstaller onefile；复用官方接口并保留源保护。单独构建日志和证据，清理本次临时编译树。
- [ ] 验证官方示例、真实数据Python/EXE一致、多文件同人、空/错误输入和隔离目录无开发PATH；对最终变更做一次全套回归及整体只读审查；更新README/报告，逐项固化并保持工作树干净。独立送测不自动发送、不晋级正式release。

Pre-flight接口：refit输出deployment manifest与训练证明→trial builder输入；gate证据summary/plan/五折report/score→refit预检。重点检查原始ID别名、多文件cap共享、固定策略不调参、错误时不发布半成品、源及模型身份。用户已授权连续执行，不另发计划审批。

- 全数据refit完成（99.23秒）：macro一次抽样26916窗/8897正例（.330547），micro61469窗/19252正例（.313199）；3323个OOF候选/152正例，116D。新独立四模型和全部SHA已保存，源/资格指纹与正式保护均通过；没有训练内F1。
- 固定聚合policy：micro=.1、blend=.25、NMS=.3、admission=.2、cap=6、event threshold=.6595272837359293、C=.1、无最终event cap。与原deployment策略相同，不将五折成绩当权重。
- 最终审查一个Important：原通用bundle writer会替换在预检后出现的独立目标。补Windows原生rename原子拒绝覆盖安装（含空目标目录），3项真实文件系统RED→GREEN测试；模型先写专属staging，再原子发布整个版本目录。此次训练已在修改前安全结束，未发生并发目标；不因安装层保护补丁重训，不改既有模型SHA。EXE同用此保护。
- Ruling: 原子no-replace发布限定Windows（本交付平台），其他系统明确失败，不将POSIX rename冒充同语义。代价：研究导出需Windows运行；通用正式artifact writer保持原行为。
