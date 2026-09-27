# A3-S1 — Frozen FD target transfer probe

状态：`S1_SUPPORTS_WIDER_INNER_PROBE`。预注册全部门槛通过；没有新的事件F1、outer评估、发布或dist变更。正式F1仍为0.6514285714。

## 固定对照结果

原inner T的24人内三折；每折16人训练、8人health。候选生产器、scaler、readout严格分区拟合，原inner V与outer未参与。编码器为同一源实验的随机初始权重R及最终FD监督预训练权重F，全部冻结。输入均为无量纲5秒token，4段mean/std池化后统一133D正则化线性读出。

| 折 | R AP | F AP | R raw BCE | F raw BCE |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.316069 | 0.438294 | 0.315846 | 0.271747 |
| 1 | 0.348501 | 0.594704 | 0.322304 | 0.260990 |
| 2 | 0.476425 | 0.588513 | 0.301082 | 0.275954 |
| 等权平均 | 0.380332 | 0.540503 | 0.313077 | 0.269564 |

F−R AP为+0.160171，3/3折改善；F同时优于metadata M的AP 0.200601以及先验BCE 0.373467。所有分区的finite/provenance/optimizer/resource检查通过。未调参数、选择最佳折或重训source。

794个health候选中662个有有效运动输入，132个（16.6%）回退到各训练分区有效候选的受试者均衡先验。gate包括全部候选，不删除困难输入。逐候选raw logits、有效性、标签及readout参数均在fold报告中。

23名AP有定义的受试者中，F相对R：15名改善、7名下降、1名持平，AP变化中位数+0.043492；1名单类受试者AP无定义。24名受试者中17名BCE改善。受试者级结果并非全部一致，不能声称普遍提升。以上由fold报告中的subject_metrics直接汇总，无额外拟合或阈值选择。

## 结论与限制

本轮配对结果支持：外部进食手势监督学到的表示，比相同网络的随机表示更适用于当前候选辨别。此前A2的candidate-conditioned VICReg失败，不代表所有表示迁移无效。A2与S1输入归一化/encoder初始状态不同，不能把两次实验差异归因于单一因素。

这是反复使用的开发人口上的candidate AP，不是独立泛化证明，更不是0.70事件F1。尚未证明与现有116D verifier互补：新表示可能只重复已有信息。也没有证明候选排序的改善能在去重、阈值和事件匹配后转化为TP增加/FP减少。

无量纲处理绕过了公共正增益和固定偏置，但不是物理单位校准；轴向增益、传感器/佩戴差异及幅值信息损失仍存在。原FD raw-unit Q2/Q3依然未执行。

## 下一步方向（不是本轮性能结论）

优先预注册更广泛的inner-only互补性对照：现有116D基线、基线加随机表示、基线加FD表示。三个分支保持候选集合、容量/正则配置及inner选择预算一致；所有训练分区各自生成所需OOF表示和评分，禁止用in-sample训练分数选择融合或threshold。先验证FD相对基线与随机扩维都有收益，再看固定协议下事件TP/FP/FN和多数分区方向。

只有在更广泛inner事件评估中达到预注册的实际改善gate，才值得申请一次完整outer评估。暂不扩网、不转Transformer、不依据本次health错误制定救援规则。若不互补，应停止简单拼接路线，而不是反复调权重消费开发集。

## 资源与复现

代码commit `9932ed7`；入口 `python -m scripts.research.representation_v2.transfer_probe`。source SHA256 `c6276837b1366117075e7e8a7504e084926cbfc5d80b98b863e442255995efc4`，plan记录完整code/spec/source身份。严格loader验证source schema、27/7人隔离、全部gate检查及无target访问。

30次候选基础模型fit、9次线性readout、0次新增神经优化。墙钟299.73秒、encoder推理9.33秒；数值不是精确GPU kernel计时。结果JSON约0.415MiB，无持久token缓存、无checkpoint复制。正式模型及dist完整保护快照前后相同。95项研究测试通过；pytest报告旧缓存目录不可写警告，不影响测试结果。
