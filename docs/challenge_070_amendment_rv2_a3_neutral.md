# A3-S4 — Invalid-motion neutral fallback

ID `RV2-A3-S4-001`. 用户批准继续执行S3之后的窄对照。目标只检验：没有有效运动输入时，直接保留基线评分是否优于混入训练先验。不是已经确定的bug修复，不承诺提升。

## 锁定的一项变化

来源固定为S2 commit `46d7c18ab11da2da025934a73f3c273e68c2aabb` 的严格分区账本。每个文件必须与该commit对应blob字节相同，保存SHA。候选、标签、validity、模型、source、分数及其祖先全部复用，不读取原始IMU、不做模型fit。

- B：原116D LR/LGBM基线分数保持原样。
- R/F运动有效：逐位保留S2已有随机/FD融合分数（75%B+25%motion）。
- R/F运动无效：直接使用B分数。
- 不改有效分支权重，不学习回退常数、置信度门限或校准器。

三个wrapper均沿用原16人policy-train/8人health的T24开发人口。对每个wrapper，先校验三份policy OOF账本和合并policy祖先图，再用原S2完整admission grid及相同tie order分别选择B/R/F策略。B策略必须与S2逐项相同。不是新拟合，也不是把旧policy阈值直接用于新评分。

**阶段屏障：全部三个wrapper策略落盘后，才允许读取任何health score/report文件，连health文件hash读取也放在屏障之后。** 不以health结果回头修改策略。health的train/score/excluded必须和对应policy拓扑一致，score及policy的全部拟合祖先隔离；FD source祖先仅原27人。新中性变换是无fit节点，显式记录在新policy图中。

随后每个health仅应用已冻结策略。回放S2 B/R/F旧预测/事件计数验证输入与decoder兼容；新B结果也必须完全一致。三臂全候选AP/BCE、event TP/FP/FN/F1、每人结果全部保留，含无效候选与不可达GT。S2旧结果保留为配对比较，不覆写。

## Gate

保持S2工程门槛：F pooled health F1比B至少高.015，优于R；至少2/3 health折改善；TP(F)>=TP(B)，且FP下降或新增TP多于新增FP；所有完整性/资源检查通过。另检查F高于S2原F结果，否则不支持该变化。通过只记 `S4_SUPPORTS_WIDER_INNER_EVENT_VALIDATION`，否则 `S4_NO_SUPPORTED_NEUTRAL_FALLBACK`。不按结果改变门槛、不搜索回退常数。

这些是被重复使用的开发health，不是独立测试、更不是正式outer。更广泛inner gate和outer人工审批仍保留，绝不自动promotion/dist更新。

## 预算、产物与实施

0模型fit、0optimizer steps、0GPU、最多30min CPU wall，4线程；新增输出<=10MiB，不复制候选原始长数组/模型/缓存，只保存源文件引用和hash、新分数、policy图、预测/指标。来源18个文件：9 policy scores、3 policies、3 health scores、3 health reports。源及保护模型/dist哈希不变。入口 `python -m scripts.research.representation_v2.neutral_probe`；输出 `outputs/challenger_70/representation_v2/amendment_a3_neutral/`。不覆盖既有attempt，异常保存incomplete状态。

执行顺序：

1. TDD增加中性回退/有效行原样保持/输入不可变、阶段屏障、错误mask拒绝、健康拓扑拒绝、gate实际改善条件测试；不改生产模块。
2. 单一research orchestration复用S2 policy/metric函数与S3验证器；terra/medium只读审核，重点检查屏障与旧模型分数冻结。
3. 研究及candidate/event回归通过、协议代码commit后运行一次。
4. 回放策略/预测/计数、检查源hash，更新README/registry/report并commit；旧数据不清除，无临时训练缓存。

本轮隔离于生产代码，复跑全部research及相关policy tests；上一阶段646 passed全仓库结果仅标记历史，不冒充本轮全仓库运行。自主探索授权覆盖实施，不重复等待实现许可；失败仅结束此固定回退方案，不自动扩大搜索。

运行前记录：11项新增测试RED→GREEN，研究/候选/event回归175 passed，2条已有缺失列imputer测试提示。terra/medium审核提出的两项保护已补齐：B候选/事件/每人指标逐项精确回放；每次health读取前，重新校验磁盘上ALL3 policy和freeze marker完整hash，并以反序列化策略执行。新测试覆盖指标漂移、策略文件篡改及无候选GT计数。后续运行结果只写独立报告，不回填为运行前证据。
