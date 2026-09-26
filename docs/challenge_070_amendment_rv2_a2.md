# RV2 Amendment A2 — 自主探索：冻结表示与低容量读出

用户于2026-09-26授权自主探索0.70路径。本轮沿用原预算和数据隔离，不继承R0的人工等待点；允许按本注册执行内层探索，不授权outer evaluation、dist变更或发布。目标是寻找有证据支持的路径，不承诺达到0.70。

## 假设与备选路线

R0中随机冻结局部encoder仍过拟合，说明可训练事件网络是必须控制的混杂因素。优先比较相同低容量读出下的随机表示与SSL表示。直接扩大scratch预算不优先；FD迁移继续等待单位核实；Transformer不作为失败后的自动替代。

本轮分类为既有研究pipeline内的有界实验扩展；用户自主探索授权取代每项设计反复确认，但注册、测试和证据gate不取消。配置在第一次拟合前提交git，不使用health曲线选择epoch、C值或增强参数。

## R1：固定三折探针

ID `RV2-A2-R1-001`。只使用outer wrapper0 / 原inner fold0的24名T受试者；GroupKFold(3)全部三折作为train/health。原inner V七人及outer受试者仍完全排除。第0折曾作为R0开发诊断，不将三折称为独立未触及测试。

每折重新生成health-train OOF candidates与health候选，生产器、阈值、归一化、scaler、encoder、readout均不fit health。每折10次macro/micro fit，合计30次；readout三臂每折各一次，合计9次；SSL每折一次，合计3次。无监督损失不读取candidate label，但采样人口来自训练内监督候选生产器，必须称 **candidate-conditioned SSL**，不是全自由生活无标签预训练。

### 固定三臂

- M：仅输入4段真实时间coverage和log1p(duration_seconds)，5D。
- R：随机冻结token encoder的16D表示，每candidate分4个顺序时间段，每段mean/std共32D，再加相同5D元信息，总133D。
- S：同一初始token encoder经过SSL后冻结；下游与R完全相同。

R/S不训练事件TCN。有效token按中心时间落入段，越界尾token夹到最后段；没有有效token的段表示为0。coverage按40ms sample cell与段交集积分。整体无效candidate以训练集subject均衡阳性先验回退，三臂完全相同；完整报告all/valid/invalid与常数基线。

读出固定StandardScaler + LogisticRegression(L2, C=0.1, solver=lbfgs, max_iter=2000, tol=1e-6)。训练每subject总权重相同、样本权重均值1，不做class重采样。Scaler同样使用这些权重，任何不收敛warning视为无效attempt，不扩大max_iter偷偷重跑。无超参数网格。

### SSL合同

沿用现有16D局部TCN；projector为Linear(16,64)-ReLU-Linear(64,64)，只训练SSL时使用。VICReg式损失：25*MSE(view1,view2) +25*两视图平均ReLU(1-std) +两视图off-diagonal covariance平方和/64；std使用unbiased variance加1e-4。采用无negative方法，避免相邻进食窗口被当负例。方法参考：https://arxiv.org/abs/2105.04906 ，具体IMU参数是本地注册假设，不是原文的验证结果。

- seed=20260926，AdamW lr=1e-3、weight_decay=1e-4、clip_grad_norm=1；batch128、固定5000 steps，无early stopping、无warm-start。
- bank仅来自当前train有效candidate token。按(sid,token_center_ms)去重；同key取按candidate_key字典序最小者的candidate-local预处理结果。每subject最多2048 tokens，按seed固定均匀无放回选取；先均匀sample subject，再均匀sample token。注明不同候选局部gravity/filter上下文可不同，去重规则不是物理原始采样去重的替代。
- 输入已gravity-align；增强只做共同yaw角[-10°,10°]（ACC/GYRO同R），六通道共同幅度[0.9,1.1]，之后用train normalization，归一化空间Gaussian sigma=.01，再共同mask一个连续12-sample区间；两view独立随机。无roll/pitch，不旋走gravity轴。
- 每100步保存loss/variance/covariance及梯度。训练结束在训练bank固定最多4096 tokens上记录encoder维度std、有效秩；mean std>=.01且至少8/16维std>=.001且有效秩>=2才算无明显collapse。该检查不保证表示有用。
- encoder和projector保存唯一final checkpoint及train normalization/provenance。绝不复用含health subject的encoder。

## Gate及后续

三折完整后报告macro AP/BCE/Brier、每受试者诊断、有效/无效比例、覆盖元信息基线、学习预算。没有任何事件F1字段。

Gate的AP/BCE统一使用每折all候选（含无效回退），再对三折等权平均，不挑valid子群。R1支持进入更广泛严格内层比较的必要条件：S相对R的macro AP>=+.02，至少2/3折AP提高；S的macro raw-logit BCE不高于R+.02且不高于subject均衡常数基线+.02；S的macro AP同时高于M；全部provenance、finite、collapse、优化及资源检查通过。否则记录 `R1_NO_SUPPORTED_SSL_BRANCH`，不反复修改本配置追三折分数。

通过也只表明值得进一步验证。下一步先注册全20分区及与incumbent互补性的严格OOF实验，仍不消费outer。失败则报告读出是否改善概率质量、随机表示是否优于元信息、SSL是否有增益，再决定下一条独立假设；不自动改参数重跑同一R1。

## 预算、实现及验证

R1额外预留3 GPUh / 6 wallh；计入原24 GPUh / 48 wallh上限，历史已知约1715训练秒/3930墙钟秒，未计量单测时间另注明。磁盘余额>=20GiB；token bank仅内存且每折释放，不持久化raw/normalized token。checkpoint预期<10MiB，总上限沿用2GiB；单GPU进程，CPU<=8线程。

实现位于 `scripts/research/representation_v2/low_capacity.py`、`ssl.py`、`explore.py`；测试位于 `tests/research/`。输出位于 `outputs/challenger_70/representation_v2/amendment_a2/`，JSON显式LF写入避免跨平台hash变化；记录source/spec SHA、subject集合、模型身份、运行时间、缓存和错误。

顺序：先测试时间顺序/掩码、subject权重、增强物理一致性、VICReg finite/collapse、bank去重与subject隔离；最小合成smoke；提交代码和注册；运行三折；记录所有负结果；保护快照对比；README/registry更新；清理本轮临时文件并commit。不在production目录改算法。
