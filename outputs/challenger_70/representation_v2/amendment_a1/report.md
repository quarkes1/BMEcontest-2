# RV2-A1-R0-001 — 训练侧健康检查

状态：`R0_COMPLETE_REVIEW_REQUIRED`。本轮完成于固定2,000步，不选择最佳epoch。
这是原inner T内部的单次健康诊断，不是完整P2比较，不产生outer/event F1，不进入P3或发布。

## 1. 已执行内容与隔离

- 执行注册：[Amendment A1](../../../../docs/challenge_070_amendment_rv2_a1.md)。代码版本 `6e5647ee36b8fb92f018032c6f0fc490f67cee40`。
- outer wrapper0的原inner fold0：24名T受试者，再固定GroupKFold(3)第0折，分为16名训练、8名health；原inner V的7人全部排除。
- 候选生产器重新在health-train内生成OOF候选并refit后预测health。共10次macro/micro基础模型fit；normalization也仅使用有效训练bag。
- 训练候选504个，其中403个有效；health候选276个，其中233个有效。health正例32个，其中有效28个。候选并非独立GT，不能按这些计数推断事件召回。
- 配对初始化、subject-first抽样；Q0冻结随机局部encoder，但事件TCN及head仍训练。Q1局部encoder及事件网络一起训练。
- 各2,000步、batch8，每100步保存训练轨迹与health raw logits；每臂16,000次candidate抽样。无效health候选始终回退到训练阳性率0.125。
- 修复后的固定200ms切段、candidate-local滤波、真实时间coverage、抽样和日志合同用于本轮；旧P2文件不回填、不重跑。

## 2. 固定终点结果

以下仅比较同一组233个有效health候选，28正例、205负例。

| 指标 | Q0：随机冻结局部encoder | Q1：scratch |
| --- | ---: | ---: |
| Candidate AP | 0.346286 | 0.402415 |
| ROC AUC | 0.809408 | 0.787805 |
| Raw-logit BCE（不裁剪） | 1.188773 | 2.320750 |
| 概率裁剪后log loss | 1.181632 | 1.967433 |
| Brier | 0.123760 | 0.120748 |
| 正例预测概率<0.01 | 19/28 | 11/28 |
| 负例预测概率>0.99 | 5/205 | 12/205 |
| Logit范围 | -114.89至29.29 | -274.94至35.57 |
| 前400步区间loss：1201–1600 | 0.068658 | 0.054456 |
| 末400步loss：1601–2000 | 0.030465 | 0.030611 |
| 两区间训练loss相对下降 | 55.63% | 43.79% |
| 预注册收敛标记 | CONVERGENCE_UNRESOLVED | CONVERGENCE_UNRESOLVED |

同一233个有效候选上，训练先验常数基线BCE为0.367375、Brier为0.105754；两臂的最终BCE和Brier均更差。
所有276个health候选上，常数基线BCE为0.359144。43个无效候选（4正、39负）上，两臂与先验基线完全相同，BCE为0.314546。
完整all/valid/invalid、分受试者有效候选指标及源文件SHA见 `subgroup_diagnostics.json`，它是由封存logits派生的补充诊断，没有重新拟合。
完整有效/全体指标与每100步轨迹分别在 `q*_observations.json`、`q*_training.json` 中。

## 3. 可以得出的结论

1. **不能把收敛告警解释为“只需多训练”。** 训练loss仍快速下降，而health loss在后期较高；两者结合支持过拟合/过度自信的诊断，不支持自动追加scratch步数。
2. **冻结局部encoder不等于冻结整个模型。** Q0的事件TCN和head仍能拟合有限候选。仅把局部encoder换成SSL权重，不能保证解决下游网络的过拟合。
3. **Q1不是毫无可用排序信号。** 此单个health split的AP比Q0高约0.0561，但AUC低、raw BCE更差，指标并不一致。不能宣称修复已带来稳定提升，更不能外推为事件F1提升。
4. **必须保留raw logits。** Q1裁剪概率log loss为1.9674，而raw-logit BCE为2.3207；只报裁剪值会低估极端错误的代价。
5. **不能把损失差全部归因于校准。** 类别抽样与自然先验不同也可能影响概率；本轮没有将其与网络容量、弱候选标签分别消融。后续需要明确区分ranking、概率校准与最终互补价值。
6. **单候选重复训练强度差异很大。** 两臂每个训练候选被抽到7–522次；subject均衡不等于candidate均衡。不能据此违反原subject-first合同，但新读出层的样本权重必须显式规定，避免把少数重复候选当成新增独立证据。
7. **跨人改善尚不一致。** 有效子群中7名受试者同时有两类，Q1 AP相对Q0为4人提高、3人降低；剩余1人为单类、AP记为null。只有一个health split，不能宣称跨受试者稳定提升。

## 4. 下一步具体建议（待审阅，不是新实验授权）

优先将下一轮问题收窄为：**在相同、受约束的低容量读出层下，目标数据SSL表示是否优于随机表示？**

- 先注册低容量readout合同，再用完全相同的readout比较随机冻结encoder与SSL冻结encoder；不能一臂用新读出、另一臂用旧事件TCN，然后将差异归给SSL。
- 保留时间顺序的确定性分段池化接正则化线性分类器，是值得审阅的首选。其维数、正则化、样本权重、缺失处理须在下一次拟合前固定；本轮不搜索这些参数。
- SSL仍优先目标训练分区内、无负样本的augmentation-consistency路线。objective、物理一致的ACC/GYRO增强、token population、固定预算和collapse诊断必须单独预注册；每个held-out subject不能参与其encoder或normalization的拟合。
- 本次health曲线不用于回选“最佳700步”等新预算，不将health AP阈值包装成完整P2 gate。下一轮若更换下游readout，应明确是新假设而非原P2的无缝续跑。
- 先用严格内层对照验证表示与读出；只有通过已审阅gate后，才测它对incumbent的互补性。FD单位未核实，不启用FD迁移；不因scratch失败直接上Transformer。

这比扩大当前事件TCN或追加epochs更直接地针对本轮发现，同时保留目标数据预训练路线。它仍是假设，不保证提升。

## 5. 成本、产物与停止点

本轮4,000次optimizer updates，训练及周期health观测用时151.94秒，端到端267.66秒；训练计时包含观测开销，是GPU训练占用的保守代理，而非kernel时间。
两个final checkpoint合计298,042字节；没有中间epoch checkpoint、没有持久化token cache。保留JSON轨迹、population、provenance和final checkpoint用于审计，不删除必要证据。

原已知历史开销约3,662秒墙钟、1,563秒训练计时；加入R0后约3,930秒、1,715秒。历史单测开销未逐次计量，不声称总成本精确到秒。未接近24 GPUh / 48 wallh / 12GiB cache / 2GiB checkpoint上限。

运行终点已记录发布保护快照未变化，outer calls=0。正式发布F1仍为0.6514285714。
依据A1第4节，两臂存在明确告警，停在诊断审阅点；不自动扩展scratch预算、不启动全量SSL，不修改dist/release。

## 6. 收口验证

- `python -m pytest -q -p no:cacheprovider tests/research`：71 passed（本次重新验证）。
- plan/spec SHA、subject分区、每100步观测的20个终点、每臂16,000抽样、新checkpoint模型配置及provenance、final logits重算指标全部核对通过。
- 本次重新计算保护快照，与plan中的 `bf74ba7a008ed8813cd480822ffdb0cdab87318d19a539c180e18b706d485d27` 一致。
- 独立terra/medium只读审阅确认停止结论可信；指出缺少无效子群/同分母基线，已以 `subgroup_diagnostics.json` 补齐。无重训、无旧结果改写。
- 执行记录：协议修复与R0入口在 `6e5647e`；此次只补完整证据、报告及README/registry。已批准计划中的告警审阅点优先于继续训练，未自行扩大研究范围。
