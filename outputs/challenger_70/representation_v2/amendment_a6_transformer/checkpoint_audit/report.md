# A6 Frozen checkpoint attribution audit

DIAGNOSTIC ONLY / USES GT. No training, threshold search, event decoding, new F1 or release change.

All rows use the same 223 V7 candidates/base scores/tokens. Own train-only normalization per checkpoint.

| Checkpoint | True motion mean | True motion median | Negative motion mean | Fused AP | Fused BCE |
| --- | ---: | ---: | ---: | ---: | ---: |
| refit_c | 0.266038 | 0.071516 | 0.023177 | 0.737430 | 0.195455 |
| refit_t | 0.260101 | 0.005379 | 0.013839 | 0.765548 | 0.189996 |
| wrapper0_c | 0.677675 | 0.782491 | 0.085016 | 0.755808 | 0.193531 |
| wrapper0_t | 0.661365 | 0.841573 | 0.061565 | 0.812321 | 0.176464 |
| wrapper1_c | 0.487710 | 0.469501 | 0.060903 | 0.739837 | 0.200567 |
| wrapper1_t | 0.501522 | 0.543791 | 0.056527 | 0.724694 | 0.199370 |
| wrapper2_c | 0.736332 | 0.902507 | 0.115828 | 0.761427 | 0.196171 |
| wrapper2_t | 0.606965 | 0.843899 | 0.051406 | 0.765190 | 0.191758 |

Comparing checkpoints controls evaluation population and input, but not training seed, size or composition.
T24 reference uses different subjects, candidates and fold-specific fitted models; it is descriptive, not paired.
V7 must not select checkpoints, fusion weights, thresholds or rescue rules. Future changes require train-only OOF validation.

## 归因结论

本次用户批准的诊断固定了 V7 的 223 个候选、基础分数、token 与有效性 mask，仅更换已冻结的 checkpoint，并保留各自训练侧归一化。八次模型推理均未训练、未选择策略、未解码事件；没有新的 event F1。

- 32 个标签为真的候选中，27 个有有效时序输入；负候选中 143 个有效。上表的 motion 统计只针对有效输入，不把无效回退概率当作时序判断。
- 三个旧 T24 checkpoint 面对同一批 V7 真候选，TCN 平均概率为 .488–.736，Transformer 为 .502–.661；最终 refit 为 .266/.260。最终 refit 的真候选平均分在七名受试者上都低于三个旧 checkpoint，但幅度不一；逐人结果保存在 summary.json。
- 最终 refit 的负候选平均分也更低，因此变化包含整体更保守/更低的概率尺度，而非只对正例作出错误判断。不能仅凭正例均值就证明分类能力崩溃。
- 最终 refit 重算的融合分数与封存证据的最大绝对误差为 TCN `4.47e-8`、Transformer `6.05e-9`。本次未发现缓存重建或推理重构造成的分数失配。
- 只读预检中，最终与旧 checkpoint 的归一化中心差异最大约 .146 个旧标准差、尺度比约 .971–1.091。没有明显的数值异常，但这不等于证明归一化无任何作用。

### 两个已知回退真例

以下仅解释此前封存的 V7 事件损失，不对任何新 checkpoint 解码或报告事件成绩。候选行号为原 v7_scores.json 的零基下标。

| 候选 / 受试者 | 基础分数 | 时序模型 | 最终 refit motion | wrapper0 / wrapper1 / wrapper2 motion |
| --- | ---: | --- | ---: | --- |
| 44 / HNU21025 | .657737 | TCN | .175449 | .782491 / .350404 / .856544 |
| 44 / HNU21025 | .657737 | Transformer | .011314 | .985196 / .161491 / .995021 |
| 154 / HNU21036 | .878293 | TCN | .080825 | .851097 / .440877 / .489163 |
| 154 / HNU21036 | .878293 | Transformer | .000115 | .000017 / .004721 / .991929 |

原封存策略下：44 的融合分数跌破最终阈值；154 在 TCN 下通过 NMS、但受试者内排名从第5变为第7，被 cap=6 淘汰。154 的 Transformer 分数虽低，原封存 Transformer 仍保留了该真例；不要把低 motion 概率本身当作最终漏检。

### 可以与不可以得出的结论

**支持：**“V7 的人完全不可识别”不足以解释失败；同一批输入上，旧模型明显没有出现最终 refit 这么强的整体压分。最终拟合产物的稳定性与概率迁移是下一步优先验证的假设。固定 .25 融合及硬 cap 会把这种分数变化放大为事件损失。

**不支持：**直接认定新随机种子就是唯一根因，或认定训练24人必然劣于16人。旧模型与最终模型同时存在训练人数/组成、候选训练集、初始化/采样种子、训练侧矩统计的区别。固定1000步也不意味着每个训练样本有相同暴露次数；末批训练 loss 不能用来判定泛化。

Transformer 最终融合 AP=.765548，高于此前基础候选 AP=.7365；因此仍有排序信号，不能把它描述成完全塌缩。任何旧 checkpoint 的较高 AP 也不构成选择该模型、更新 dist 或宣称 F1 改善的理由。

## 下一阶段建议（未启动）

先在严格 T24 train-only / nested OOF 内验证 refit 稳定性：预注册固定种子重复与三种子均值对照，保留原单模型、原架构/数据/每模型1000步/固定融合，不挑最优种子。比较时序概率稳定性、候选 AP/BCE 和最终事件 TP/FP；只有训练侧事件收益与多数折方向支持，才考虑下一项校准/融合改动。

不能直接平均三个旧 wrapper 作为 T24 OOF 改进结果：其他 wrapper 可能训练过当前被评估人。合法集成必须在每个 inner-train 内独立训练，所有成员共同排除相应 held-out 人。V7 已用于诊断，不能再次作为不受污染的验证门；任何后续验证方案需另行明确其证据边界。

本次未启动上述训练，也未调用 outer、修改 dist/release。源输入、模型、缓存和正式发布快照在结束时均通过哈希复核；没有新增 token 缓存，必要诊断证据保留于本目录。

## 实现与验证记录

- 代码提交 `a5829c0` 在真实诊断推理前完成；六项新增测试先观察缺失实现导致的 RED，再观察 GREEN（6 passed）。表示研究回归 298 passed、20 项已知/预期 warnings；全仓回归仍在运行，尚不声称通过。
- Luna / medium 独立只读审查：未发现 Critical/Important 缺陷。Minor 为未使用 import；测试覆盖缺口为未用合成全流程覆盖 run 的全部门禁；本次真实完整运行、封存分数复现及结束哈希复核提供了实际路径证据，但不等于覆盖所有错误分支。审查未判断泛化/因果结论。
- 真实诊断 COMPLETE，8 次已冻结模型推理，training_fits=0、outer_calls=0；模型推理与评分汇总阶段约4.55秒，不包含入口导入/预检耗时。未新增 token 缓存。
- 源输入/cache/model/code SHA、正式发布保护快照、scores.json SHA 均经独立只读复核通过。所有新输出只在 checkpoint_audit 子目录，不覆盖 T1/V7 封存结果。
