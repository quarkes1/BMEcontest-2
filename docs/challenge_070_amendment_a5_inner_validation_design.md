# Amendment A5 — 已有收益的扩大 inner 验证设计

状态（2026-10-12）：书面设计与实施计划已获用户审阅通过；代码已实现并修复审查问题。
全仓回归中途停止，未取得完整通过记录；尚未生成manifest或开始训练。
按用户“先别启动，讨论Transformer”的要求暂停，不自动恢复，也不记为算法失败。
新架构讨论见[A6设计](challenge_070_amendment_a6_transformer_design.md)。
如后续获准恢复，仍须按回归→manifest→训练执行；本设计合同在manifest生成时逐字节锁定。
阶段 ID：`RV2-A5-S15-001`。

## 1. 目标与依据

目标仍为正式事件 F1 0.70+，本阶段只回答：冻结 FD 与 S11 的开发收益能否在不同
受试者组合下复现，以及两者是否具有错误互补性。不新增架构、不搜索适配参数、不训练融合器。

历史 T24 三折：B incumbent116D 为66TP/20FP，F1 .721311；R冻结FD251D为69TP/20FP，
F1 .741935；F为S11 batch32适配FD251D，66TP/16FP，F1 .737430。
R只有1/3折改善，F只有2/3折改善；两者低分位受试者均有风险。
S8/S11旧gate失败结论保持不变。本设计明确将历史对照信号用于新的探索，不事后宣称旧实验通过。

## 2. 数据边界与证据等级

- 从 `registered_e03_config(0, workers=4)` 的 outer-train 载入既有特征，按现有
  `inner_subject_partitions(data, n_splits=4)[0]` 取原T24及原V7，立即裁到T24。
- 严禁原V7及outer受试者参与raw加载、拟合、阈值选择或结果分析；source-val也不参与训练。
  既有特征表的物理反序列化边界须如实披露，不声称未读取过其他人的文件字节。
- 这是**扩展划分而不是扩展人群**：T24已多次用于开发，重新划分不是新独立验证集。
  不产生新的正式outer成绩，不宣称获得独立泛化确认。
- manifest必须明确列出T24/V7/outer/source-train/source-val身份及hash；任何身份重叠、
  T24不是24人或缺少训练窗的情况均停止，不静默调整人数。

## 3. 固定划分：两次三折，全部运行后统一分析

两次划分盐值固定为 `20261012`、`20261013`。每次按
`sha256(f'{salt}:{subject}'.encode('utf-8')).hexdigest()` 升序排序T24，
hash相同以subject字符串排序，按排序位置模3分配health组；每组8人，其余16人训练。
不按标签、餐数、已有成绩或窗口数量重新平衡。不尝试第三个盐值。

每次每人恰好一次health；两次共6个wrapper，但仍只有24个独立受试者。
划分manifest及全部配置在计算任何新health成绩前提交锁定。若两次划分等价或与历史三折
等价则停止并报告，不能自动换盐值。原producer OOF与policy OOF的GroupKFold逻辑不改。

每个16人wrapper内，三折policy OOF独立重建producer、表示及verifier，然后分别为
B/R/F选择既有准入网格和单一event阈值。六个wrapper的全部policy冻结并核验hash后，
才允许进行health最终refit与评分。health候选标签不能用于训练或policy选择。

## 4. 三臂算法合同

- B：现有研究对照116D，固定LR(C=.1)/restricted LGBM各.5融合；不是部署bundle的完整重放。
- R：S8冻结FD251D有序联合表示，固定原source encoder；不是随机对照。
- F：S11 batch32/norm1适配，1000 optimizer steps，encoder/head学习率、source_weight=.25、
  AdamW、seed20261011、source bank及增强均保持原合同，不据新health调整。
- source checkpoint SHA-256：
  `c6276837b1366117075e7e8a7504e084926cbfc5d80b98b863e442255995efc4`。
- 每个policy训练分区与final refit都重新从source初始化F；禁止跨分区复用适配checkpoint、
  scaler、imputer、verifier或已选择的policy。R只允许复用外部固定encoder。
- 同一split的三臂共享candidate ledger及producer计算，不改变candidate控制、缺失行为、
  匹配器IoU>=.25语义和eligible分母。本阶段不混入官方严格>边界迁移。

## 5. 预先固定的判读，而非复合全优门槛

主要指标：每次划分先对三个互斥health组聚合TP/FP/FN，计算一次pooled event F1；
以两次pooled F1的算术平均作为臂的开发指标。不能把重复出现的GT计作新增独立GT。
分别报告R-B、F-B以及F-R，原始两次结果不可被均值覆盖。

完整性/预算检查是硬要求，未完成六个wrapper则只能报告INCOMPLETE，不能选winner。
全部完成后，每个候选分别标注：

- 平均delta<=0：本轮未支持改善。
- 平均delta>0但任一次delta<=0：有平均收益但划分不稳定。
- 两次delta均>0、平均delta<.005：方向一致但实际收益小。
- 两次delta均>0、平均delta>=.005：支持进入下一份有限候选研究设计。

.005是本次开发阶段的工程分层，不是统计显著性，也不是发布门槛。
R/F分别判读，不要求F胜R才承认F相对B的收益；不要求TP、FP、AP、BCE同时占优。
逐人退化必须醒目标注，但不能事后变成隐藏gate。没有任何分层自动授权outer或发布。
若两臂都获支持，后续设计优先保留平均F1更高者；差值<=.001视为工程近似持平，
优先较廉价的R，同时保留F的互补诊断。此规则不替代后续设计审阅。

## 6. 必须输出的诊断

- 两次总体和六折的TP/FP/FN、PPV/recall/F1、短餐完整匹配计数及candidate AP/BCE。
- 每次逐人F1 median/IQR/P10、改善/下降/持平人数，以及两次逐人delta均值；
  不将48个subject-occurrence称为48名受试者，不把六折当独立重复样本做显著性声明。
- 每次按完整GT匹配结果，对每个GT记录三臂是否检出：新增/丢失TP、共同FN、R独有/F独有TP。
  FP按原始sid和区间保留清单及数量，不另引入任意FP配对阈值来声称“相同误报”。
- candidate级保存对齐score及分歧，用于后续提出假设；本阶段不搜索阈值、融合权重或规则。
  不报告GT-aware合并模型的F1，不把“任一臂命中”的计数当成可实现性能。

## 7. 实现边界与效率

复用 `scripts/research/representation_v2/adapt_probe.py` 的split级训练及池化，新增独立
`validation_probe.py` 组织划分/冻结/汇总，测试放在
`tests/research/test_representation_v2_validation_probe.py`。不大改原S5/S8/S11入口或旧gate。

现有split函数内硬编码的S2 baseline replay只适用于旧划分；新入口必须显式使用新的
split实现或显式注入replay策略，默认旧路径不变。禁止全局monkeypatch绕过保护。
新B与同split独立调用现有116D scorer对齐；另用历史小型fixture验证原replay保护仍有效。
每个学习节点输出实际fit_subjects及祖先图，检查policy-val/health/V7/outer均未参与其拟合。

预计24个split级拟合、240个producer fits、144个event verifier fits、24个初始化readout、
24000个神经optimizer steps；这些是预算，运行时另外核对实际计数。
CPU线程固定4，CUDA可用则使用；单卡依次运行，避免多进程复制token bank。
总wall预算8小时，adaptation+encoding同步计时预算4小时；输出上限128MiB，磁盘保留20GiB，
CUDA初始空闲>=1280MiB、峰值<=初始空闲80%。任何超限留下INCOMPLETE，不挑已完成折下结论。
资源检查至少在每split前后、适配每100步、health开始前进行；不得最后才发现超限。

## 8. 证据、测试与清理

输出限定 `outputs/challenger_70/representation_v2/amendment_a5_validation/`：
plan、manifest、policy冻结marker、各split scores/adaptation、六fold报告、summary、
complementarity、state、report。代码/spec/source/manifest/策略均hash锚定，禁止覆写attempt。
只保留必要复核证据，不落盘token bank，不复制source checkpoint。

实现计划须以测试覆盖：确定性划分/无标签依赖、subject隔离、全policy冻结屏障与hash篡改、
新旧replay路径隔离、两次结果不重复统计、空预测/缺失运动输入、实际拟合预算、
运行中超限/异常恢复标记。先RED→GREEN，相关测试通过后再启动训练。
完成后运行相关回归；共享核心代码发生改变时执行全仓回归。文档/registry与结果同步更新，
仅删除本阶段可重建缓存和临时文件，不删除旧正负证据或用户文件。

## 9. 发布边界与下一步

NO OUTER / NO DIST / NO RELEASE / NO PROMOTION。
正式发布仍为run `160afaf81debf1ee`，F1 .6514285714。
本轮不注册融合器或新时序网络；完成后根据收益及互补证据提出下一份窄设计。
若后续正式可比F1相对发布提升>.03，按用户要求先审批，再更新dist。

## 10. 审阅与实施状态

- [x] 核查S8/S11合同、现有引擎与数据边界，形成此书面设计。
- [x] 用户审阅书面设计。
- [x] 编写并审阅逐任务实施计划。
- [x] 实现固定划分、显式回调、六wrapper屏障及统计；50项相关测试通过。
- [x] Luna/medium只读审查；启动失败状态问题已补测试RED→GREEN修复。
- [ ] 独立代码审查及全仓回归，锁定manifest后执行。
- [ ] 完成六wrapper并封存报告、必要回归、缓存清理及git提交。
