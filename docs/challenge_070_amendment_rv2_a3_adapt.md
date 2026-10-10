# A3-S8 — 源/目标监督适配的事件级对照

ID `RV2-A3-S8-001`。用户授权自主选型、训练、验证；发布节点人工审批。
正式F1仍为.6514285714，目标.70+。模型复杂度没有全局硬上限。

## 假设与固定对照

S1的FD冻结133D表示可学习，但冻结表示的固定融合、无序拼接及小型时序读出没有稳定事件收益。
下一步允许encoder以目标事件监督适配，而非继续只改后处理。本轮不做架构/超参数搜索。
原inner T24的3个16train/8health wrapper，各wrapper再3折policy OOF；原V7、outer完全排除。
所有三fold policy先落盘冻结，再读取health。复用S5隔离引擎，但不复用已拟合的目标模型。

三臂：B=原116D LR/LGBM；R=冻结FD encoder的251D联合verifier；F=适配FD encoder的同宽251D。
R在本实验表示frozen reference，不是随机encoder；随机信息效应已在S1/S2/S5/S7报告，
本轮只检验在相同源初始化下目标适配的增量，不声称证明预训练优于scratch。

251D = 原116D + S1四段mean/std128D + 四段sample-time coverage4D + logduration1D
+ valid token fraction1D + any-valid1D。整体无效时前128D=NaN，质量项保留；train-only imputer。
全候选、不裁长事件，真实时间/gap不变；gravity_align=False、逐token/triad无量纲处理不变。
R/F同宽、相同固定StandardScaler/LR(C=.1)+原restricted LGBM、固定.5/.5 blend。
此联合表示不是S5的38D无序池化，也不是S7的64bin非线性head；主要归因仅限F对R。

## 每个训练分区独立适配

源checkpoint固定SHA
`c6276837b1366117075e7e8a7504e084926cbfc5d80b98b863e442255995efc4`。
FD bank只允许源27 train参与优化；7 source-val不用于训练/早停/参数选择。
原pickle整体包含34人的数组，反序列化后筛选27人，不称物理上从未读取source-val文件字节。
核对S0记录的源文件hash，不下载新数据。

初始化：当前train有效候选的冻结FD 133D，fit S1 StandardScaler/LR；拷贝为可微linear head。
scaler固定；encoder所有参数开放，以小学习率适配。最终事件verifier只用encoder embedding，
不用辅助head概率，避免再混合不同评分尺度。

固定1000 optimizer steps，无early stopping。target batch4，subject→candidate（不按class重采样）；
source batch64，subject→class→token，沿用source共享ACC/GYRO三维旋转。
loss = target普通BCE + .25×source手势BCE；source head冻结但梯度通过它传回encoder。
AdamW：encoder lr1e-5、target head lr1e-4、weight_decay .001，clip1；固定seed20261011。
target不新增增强，不改source/target采样比例。每100步保存target/source loss及梯度有限性。
使用最终步，不因训练loss持续下降自动延长；轨迹只作诊断，不据health挑checkpoint。

训练representation后在同一train拟合联合verifier属于监督特征学习；完整pipeline给policy-val/
health打分时，其encoder/head/scaler/verifier/producer必须全部排除该受试者。
严禁在整个wrapper train先适配一次再给其policy OOF共用。

## 事件gate与诊断

采用既有S2 gate：F pooled inner health F1比B增加>=.015，F>R，至少2/3折胜B，
TP不减少，且FP降低或新增TP多于新增FP；完整性和资源检查全部通过。
记录三臂TP/FP/FN、逐人F1、候选AP/BCE、所有预测与GT、source/target轨迹、预算。
重复开发人口上的inner F1不是正式outer F1。通过仅进入更广泛inner验证，不能直接发布。
NO OUTER / NO DIST / NO RELEASE / NO PROMOTION。发布成绩提升>.03时人工审批再更新dist。
历史IoU>=.25仅用于同协议对照，不暗改竞赛严格>的边界语义。

## 资源和实施台账

预注册120个producer、72个event verifier、12个初始化LR、12次encoder适配，共12000步。
CUDA可用则使用；CPU4线程。6wallh/3GPUh是本次停止预算，不是硬件/模型能力上限。
磁盘保留20GiB、初始空闲显存>=1280MiB、峰值<=80%初始空闲；输出<=64MiB，
不保留token缓存、不复制source checkpoint；适配后的encoder/readout小型JSON权重为可复核证据。
入口 `python -m scripts.research.representation_v2.adapt_probe`；输出`amendment_a3_adapt/`。
代码/规范先commit；结果后更新README/registry并commit。失败配置不在health上修改步数/seed追分。

Ruling：本文件兼作计划/台账，沿用研究目录，不另建重复文件夹；自主授权覆盖研究实现，
但不覆盖dist发布。代价：只测一个固定适配预算，失败不等于所有fine-tuning无效。

- [x] 1. 先RED→GREEN：可微133D池化与S1对齐、无效输入、source head冻结、原encoder不变、
  训练subject约束、251D质量/缺失、真实verifier及gate。
- [x] 2. S5 engine增加显式变体hook，默认S5行为不变；actual fit graph、baseline完整回放、
  全policy冻结屏障保留；独立Luna/medium审查，预注册commit。
- [ ] 3. 执行一轮strict nested事件对照；相关及全仓回归，封存正负结果、清理临时文件、commit。

Pre-flight：可微池化→原133D scaler一致；source梯度可流过冻结head但不更新head；
适配weights每partition重新从source初始化；F训练encoder不能给其他policy-val生成OOF。

实施记录：缺失adapt模块RED→核心4项GREEN；缺失engine contract行为RED→GREEN，
补充CUDA真实训练2步后共6项通过。Luna/medium独立只读复核无实质阻塞；
实际适配fit_subjects仅包含有有效运动输入的train人，源训练身份在report单独记录。
