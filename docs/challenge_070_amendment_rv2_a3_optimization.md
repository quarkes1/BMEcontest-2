# A3-S9 — 仅训练侧的适配优化审计

ID `RV2-A3-S9-001`。用户授权自主研究，正式F1目标.70+；dist发布仍需审批。
本轮是有边界的诊断，不搜索health/outer参数，不产生新事件F1。

## 问题与固定输入

S8固定步数适配未胜冻结FD，11/12个分区的末段target batch BCE高于首段。
随机batch均值不能区分采样噪声、负迁移和优化故障，先检查固定完整训练人口。
固定S8 fold0_policy0的11个train subjects，不根据health选分区。
只读取该partition元数据及其adaptation JSON（锚定sealed commit8206cc6），不读取S8 health文件或
其他policy结果。继承loader加载outer-train特征表后立即切到11人；原V7/其他人不参与任何
拟合、测量或raw token读取，outer validation不加载。
源checkpoint及源27 train bank保持S0身份；source-val仅原pickle反序列化后排除，不参与梯度。

训练候选须重新按同分区OOF生成，**不生成validation ledger**。8个既有候选producer OOF fit，
无最终producer refit、无新encoder/verifier训练、无optimizer更新。现有crossfit增加空score
分区的train-only能力；非空score的旧行为保持不变，测试其训练ledger一致性。
只加载允许train subjects的raw候选token，不创建持久token缓存。

## 测量与解释

1. 重算源encoder的133D训练特征，独立fit一次初始LR/scaler验证重放；须与saved参数一致。
2. 固定同一valid train集合及subject均衡权重，报告四种encoder/head组合的BCE/AP：
   initial/initial、final/initial、initial/final、final/final；不混入无效运动候选。
3. 报告每个人的固定train BCE、latent标准化漂移和encoder/head权重漂移。
4. 固定seed20261012、四组subject均衡抽样的4个target候选，分别测初始/最终模型的
   encoder target梯度及64个source token梯度的范数、余弦和.25加权源梯度范数比。
   source保持共享ACC/GYRO SO(3)旋转，不能反传更新权重；检查target梯度实际可达encoder。

不以单个batch余弦证明因果，不声称train BCE下降代表泛化。
若固定train损失恶化，后续可独立注册一个去除source辅助项的单因素对照；
若优化已有效而事件无益，则停止该适配配置，转向细粒度/多尺度时序证据。
即使诊断有利，也不把它记入正式performance或直接启动outer。

## 资源、产物与台账

入口 `python -m scripts.research.representation_v2.optimization_audit`。
输出 `outputs/challenger_70/representation_v2/amendment_a3_optimization/`：plan、train_metrics、
gradient_analysis、state、report。0 encoder steps、8 producer fits、0 event verifier fits，
另有1个固定initial readout重放fit，不搜索参数，不用于新验证成绩。
30分钟墙钟停止预算、4 CPU线程、CUDA可用则用；输出<=2MiB，磁盘保留20GiB。
源/输入文件hash及发布保护快照前后核验；异常记录incomplete，禁止覆盖attempt。
无新增依赖、无dist/release修改，代码/规范先commit，结果与README/registry后commit。

Ruling：自主授权覆盖有边界诊断及后续研究实现，不覆盖发布；沿用docs与research结构，
本文件兼作计划/台账，避免重复文档。代价：单一训练分区诊断不能推广为全人群结论。

- [x] 1. TDD：train-only OOF ledger等价与隔离；固定readout评分；target梯度可达encoder；
  梯度方向/零范数行为；独立审查后预注册commit。
- [x] 2. 执行审计，核验完整性；根据证据选择下一条单因素或新时序路线，不在health追参。
- [ ] 3. 回归、README/registry与证据封存、临时文件清理、commit。

实施：空score路径先RED（empty subject错误）再GREEN；缺失audit模块RED→三项核心GREEN。
139项表示研究相关测试通过，4条既有imputer提示。Luna/medium只读审查无实质阻塞。
审查指出readout replay fit应单列，已记录为1，不与0 event verifier fits混淆。

结果：66.34秒，`S9_TRAIN_ONLY_AUDIT_COMPLETE`。272 valid/41positive/72invalid，
初始与最终完整train BCE .221839→.261428，AP .764366→.748730，仅3/11人BCE改善。
target梯度有限非零；4个诊断target batch全negative，不用于断言positive/source因果。
发布保护/输入hash不变。决定预注册两个有界train-only优化单因素对照；不直接进outer。
全仓回归在下一阶段代码冻结后统一执行，本项台账保留pending，避免声称已完成。
