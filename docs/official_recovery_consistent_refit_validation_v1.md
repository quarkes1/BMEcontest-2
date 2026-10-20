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
