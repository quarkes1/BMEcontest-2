# A3-S1 — 无量纲FD表示的目标域迁移对照

用户自主探索授权下执行；上游 `RV2-A3-S0-001` 已通过source gate。ID `RV2-A3-S1-001`。新注册不修改A2/R0结果；无outer、无dist、无promotion。

## 固定实验

唯一允许的source checkpoint SHA256：`c6276837b1366117075e7e8a7504e084926cbfc5d80b98b863e442255995efc4`。加载时同时校验schema、27/7来源参与者、源实验完整性检查、无target/outer访问、训练步数及plan/code身份；禁止用自洽但被替换的summary/权重绕过固定身份。

使用A2同一原inner T的24人、GroupKFold(3)全部三折；原inner V七人及outer排除。这些是反复使用的开发人口，不称独立测试集。

每折重新拟合macro/micro候选生产器（10次/折），只用当前train。读取train/health候选的A1时序token，配置gravity_align=False，再执行A3每token/triad无量纲处理。保持sample_mask/真实coverage；token_mask还须满足非退化条件。整体无效candidate以训练有效candidate的subject均衡先验回退。

三臂：M=5D coverage/duration；R=source checkpoint保存的随机初始encoder；F=最终FD encoder。R/F采用相同无量纲输入、4段mean/std+5D共133维，固定StandardScaler+L2 LogisticRegression(C=.1,max_iter2000,tol1e-6)，subject均衡权重、无class重采样；M同样读出配置。

encoder冻结、无target神经网络优化；source checkpoint核对SHA和身份后可跨target分区共享，因为从未看过target raw/label。每fold祖先图包含FD的27 train IDs；任何目标health/原V和7名FD validation不可出现在拟合祖先。target scaler/readout只fit该fold T；输入归一化无fitted statistics。

source head不用于目标评分。R/F只输出16D embedding；9次readout、30次基础模型fit、0次新神经网络fit，仅保存readout JSON/score ledger，不复制source权重。

## 预注册gate

使用all候选（含无效回退），每折指标再等权平均，全部满足：

1. F相对R macro AP增加>=.02、至少2/3折AP提高；
2. F macro raw BCE<=R+.02，且<=subject均衡先验baseline+.02；
3. F macro AP高于M；
4. provenance、finite、readout优化完整性和资源检查通过。

通过 `S1_SUPPORTS_WIDER_INNER_PROBE`；否则 `S1_NO_SUPPORTED_TRANSFER`，不改变C/池化/source seed/增强追分。保存all/valid/invalid、分受试者指标、raw logits和负结果。通过仅允许下一份更广泛内层/互补性注册，不直接outer；事件级TP/FP证据仍是0.70目标的必要条件。

## 资源与实现

上限1 GPUh（仅推理）/3 wallh，计入原总24 GPUh/48 wallh；新增token缓存0、不复制source权重。当前正式研究已知约1948训练秒/4491墙钟秒，历史单测成本另计，不称精确GPU kernel时间。

入口 `python -m scripts.research.representation_v2.transfer_probe`；输出 `outputs/challenger_70/representation_v2/amendment_a3_target/`。保存source/spec/code SHA、subject/祖先图、sourcecheckpoint SHA、预算、模型及候选身份。代码/注册训练前commit；完成后更新README、registry、报告并commit。
