# RV2-A3-S0-001 — FD源数据手势学习

`S0_SUPPORTS_TARGET_PROBE`。本轮未读取目标IMU，没有outer、事件F1或发布变化。

FD-I右腕：27人训练、7人验证。共11,121个经过固定cap的有效token（8,704 negative、2,417 eating）；训练8,847、验证2,274。此次无退化token被剔除；5名参与者右腕无positive，不强造阳性。

验证AP **0.8440475**、AUC **0.9509311**、raw BCE **0.2570439**、Brier **0.0775583**。验证阳性率0.2119613；AP高于其+.10，BCE低于预注册sampler先验基线0.6182658，因此通过source gate。

先验分别为：训练抽样器期望0.4259259、cap后bank 0.2187182、cap前全部eligible训练token 0.0131713。这些不是同一人口；最后一种先验在抽样验证人口上的BCE为0.92818，不能用于宣称真实自由生活校准性能。cap后bank先验BCE为0.5166783。

此结果仅证明：每token/triad去均值与RMS后，源手势仍可学习且能泛化到未见源受试者。它不证明目标域迁移有效，不解决量程、轴增益、饱和或左右腕映射，不等于meal事件识别成功。

固定5000步，无early stopping；训练38.60秒，含指纹/读取/预处理/保护检查的墙钟96.35秒；峰值CUDA分配约38.9MiB，无持久token缓存。实际抽样640,000 token，positive272,918、negative367,082。只保留final权重和轻量证据。

代码 `1cc8706`；source文件SHA、split、spec SHA在plan，轨迹、validation logits、population和checkpoint均保存。研究测试87 passed；前置审核纠正cap后bank先验不能叫自然人口先验，并补回归测试。保护快照未变。

下一步是独立登记的目标域随机/FD冻结配对probe，不能用source AP替代目标候选或事件成绩。
