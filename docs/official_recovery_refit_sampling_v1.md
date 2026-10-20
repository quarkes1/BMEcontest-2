# REFIT-SAMPLING-V1：部署重训抽样差异的固定策略对照

2026-10-20。训练侧机制实验，不是正式成绩，也不直接修改部署模型。

## 已验证的实现差异

正式模型160afaf81debf1ee五个outer宏窗HGB保存的基准正类概率分别为.337115/.310366/.333601/.346865/.322457，而deployment为.031810。读取前验证各自manifest中的joblib SHA。
代码build_full_target_dataset将五份完整val联合同时作为window_train和candidate_train；fit_full_target_deployment直接以所有label>=0窗口训练宏窗，没有沿用旧train缓存的负样本抽样。HGB无class_weight。这不是CV主体泄漏，而是CV与最终重训的数据分布不同，可能改变分数和候选触发率。

另外，deployment的verifier使用当前窗口模型在自身训练数据上的预测构建特征，CV用窗口OOF特征。本实验先只隔离抽样；不同时修复OOF stacking，否则无法解释来源。

## 配对设计

既有T24三个16train/8health划分；完整候选池，两臂一致。A在当前16人的全部完整缓存窗口(label>=0)上训练窗口模型；B使用同16人的原CV抽样window_train/micro_window_train，样本、标签不重写。HGB、LGBM架构/参数/seed相同。

两臂verifier均使用训练内窗口预测形成的116D候选特征，旧eligible事件监督，LR/LGBM各.5；这样不把OOF修复或新完整事件监督混入抽样因素。窗口seed+2/+12、verifier seed+3/+23对齐规范最终fit工厂。

每折micro阈值、admission、event threshold固定为confirmed EVENT-COVERAGE-V1同一16train分区的old臂选择；不使用全数据正式deployment策略，因为它不能被称为当前health完全隔离。源账本只提取train-only微阈值及祖先身份，绝不使用源health评分选择参数。不能声称本研究A是正式EXE逐位复现：它是相同重训机制、不同训练人数与已注册.5融合的受控模拟。

全部策略预先固定，零参数搜索；归一化、imputer、窗口模型、verifier仅fit当前train。统一完整160餐/严格IoU>0.25评价health，不删除无候选或无覆盖餐。报告每折/逐人TP、FP、FN和宏窗正类比例、health窗口分数分布。

## 门与资源

预注册B-A pooled F1>=.01且至少2/3折提高才支持更广训练侧验证。即使A严重下降也不推断官方损失全部由它造成；需独立验证和真实官方接口才能交付。T24反复开发，非独立测试。

24个模型fit（12窗口、12verifier），CPU4线程，<=30分钟、输出<=64MiB、剩余磁盘>=10GiB；不保存模型，不读原始IMU、不改生产代码/release/dist。不要求GPU，因为所用模型为现有CPU树模型。
