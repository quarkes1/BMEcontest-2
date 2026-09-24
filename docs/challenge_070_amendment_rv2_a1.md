# RV2 Amendment A1 — 协议修复与训练侧表示健康检查

用户已于本轮批准按分析方向推进。目标仍为最终 event F1 0.70+，本 amendment 先恢复可信的输入/训练合同，再检查冻结预训练表示的可行性。原 RV2-P2-001 的文件和分数保持历史记录，不回填成修复后结果。

## 1. 已确认的问题与解释边界

旧 P2 的 Q1 macro AP 0.26439 < Q0 0.28847；有效输入子集仍负向。已发现：

- 200 ms 间隙合同被动态两倍 packet 周期切段替代；220 ms 实际间隙可被插值。
- candidate 前后 1s 信号参与双向滤波，违背 candidate-local 输入合同。
- 保存结果未包含每100步轨迹，不能执行末两个400步区间的收敛检查。
- sampler 实际为 class→subject→candidate，而非 subject-first。
- bin coverage 使用 token coverage 平均而非真实时间覆盖。

这些发现不抵消已观测的负结果，也不证明修复必然提升。旧结果应解释为旧实现未通过 gate，不能作为完整协议验收通过的证明。修复不重写发布性能。

## 2. 固定修复合同

1. packet 时间展开后额外按真实间隔 `>200 ms` 切段；不修改生产 release 的公共 timeline 函数。
2. candidate-local 抗混叠滤波只访问 `[start,end)` 内数据；短 span 仍 mask，禁止借候选外数据补足。
3. bin coverage 按 sample-mask 所代表的40ms观测区间与bin的交集计算。时间网格和64个bin保持不变。
4. sampler 明确定义为：均匀选 subject → 在该subject存在的类别中均匀选class → 均匀选candidate。单类subject保留。两类存在的subject在期望上类均衡；不强制每batch全局4:4，因为这与严格subject-first在单类subject存在时不能同时保证。记录实际class/subject计数。
5. 保存每100步loss/梯度/累计抽样，保存raw logits，安全checkpoint记录收敛结论。固定2,000 steps、batch8、AdamW和原架构保持不变。
6. 当steps>=800，末400步loss相对前400步下降>10%记为 `CONVERGENCE_UNRESOLVED`。不足800步为 `NOT_ASSESSED`，不得伪称收敛。

## 3. R0：只在原 inner T 内进行健康诊断

实验ID `RV2-A1-R0-001`。固定 outer wrapper0、原inner fold0 的 T；再按subject GroupKFold(3)取第0折为 train/health。原inner V及outer validation不进入任何健康拟合或评分。候选生产器和normalization均只fit health-train subjects。

在完全相同的candidate ledger上，对配对初始化的随机冻结encoder与scratch分支各跑一次2,000 steps。每100步观察health logits，仅解释训练进程，不选择best epoch、不改变步数、不调阈值。无效candidate按训练候选阳性率回退；报告有效/无效子群与恒定先验基线。

检查finite loss/gradient/logits、预测饱和、训练与health loss差距、subject覆盖、收敛标记。所有曲线含完整预算，不使用health曲线追选新超参数。数值错误、输入合同失败或祖先交集直接停止并修复；不得启动完整20分区比较掩盖问题。

## 4. 后续表示路线

R0结果用于形成下一份可复核的具体预训练注册，不将R0 health得分称为P2完整通过。优先目标数据train-partition-only自监督预训练与冻结encoder；FD路线继续以单位、坐标、intake中点语义核实为前置条件。预训练objective、augmentation、token population及steps必须在训练前固定并写入独立plan identity。未完成这些合同，不开始大规模SSL fits。

主线可继续在当前用户授权范围内准备与实现；若R0留下明确收敛/输入告警，则先提交完整诊断及下一步具体方案，不自动增加scratch训练预算。原P4/P5仍各需独立授权。

## 5. 资源与记录

继续原总限额：24 GPUh、48 wallh、新cache12GiB、checkpoint2GiB，单GPU训练进程。历史已知约3,662秒墙钟及1,563秒训练GPU计入存量；历史单测开销未逐次计时，不能声称总成本精确到秒。R0预留2 GPUh/3 wallh，token仅内存、checkpoint仅final，所有失败记录。

文件写入 `outputs/challenger_70/representation_v2/amendment_a1/`。每个新attempt有独立source SHA、plan SHA、subject集合和budget。发布与dist保护快照相同；registry仅追加audit/新attempt；README说明新旧结论边界。
