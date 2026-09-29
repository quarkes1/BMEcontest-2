# A3-S5 — Frozen representation joint verifier

ID `RV2-A3-S5-001`. S4未支持中性回退；停止调整固定概率混合。新假设：让受约束的verifier联合解释原116D与冻结FD运动表示，可比独立评分混合更有效。所有结果仅是重复使用T24的开发证据，不是正式outer。

## 唯一注册配置

- B：S2相同116D、LR C=.1与原浅层LGBM，固定各50%。
- R/F：116D拼接38D，合计154D；相同verifier超参数/类别权重/种子。R使用S0保存的随机encoder，F使用S0的FD监督encoder，均冻结，不微调。
- 38D顺序：全candidate有效token的16D均值、16D总体标准差(ddof=0)、4段实际sample-time coverage、有效token比例、any-valid标志。无有效token的32D为NaN，quality仍保留；只能由训练分区拟合imputer，不做强制先验/基线概率回退。不追加时长（116D已有），不搜索维数、pooling或权重。
- 单token的std为0；无token的比例为0。输入沿用S1 unitfree、raw frame、5s/25Hz与有效性规则。仅有效token进入encoder；全部candidate保留。NaN只允许缺失embedding与已有base缺失值，inf一律拒绝。
- 不使用source分类head，以避免其初始随机权重未保存造成不匹配的控制。

## 隔离与冻结

严格复用S2的T24/三折16train+8health；每个T16内部三折重建candidate producers并训练联合verifier，生成policy OOF。每个训练分区所有目标域learned ancestors只用本分区；FD仅原27个source-train，source-val不参与encoder fit。source checkpoint hash沿用S0锁定值。

全部三个wrapper的policy先落盘并hash冻结，然后才生成任何health评分。每次health前重新加载并验证完整冻结清单。候选准入grid、event threshold selector、tie order与S2不变。最后各T16重拟合一次、应用冻结policy于H8。全程不根据health改变配置。

B必须重放S2候选身份、标签、groups、概率（atol=1e-12,rtol=0）和policy、预测与事件计数；不一致停止，不把不可比结果解释为算法提升。旧health文件只在全policy冻结后读取。祖先图来自真实producer fit，不能仅列声明人口。

## Gate与资源

沿用S2：F pooled F1比B至少+.015、F优于R、至少2/3折改善、TP不下降，且FP减少或新增TP多于新增FP；完整性/资源检查全通过。通过仅支持更广inner验证，不自动outer。失败不局部修改后重跑。报告全部TP/FP/FN、candidate AP/BCE、每人结果、折间方向。后续outer工程gate仍需约.78的广泛inner结果与人工审批。

120 producer fits、72 verifier fits、0神经optimizer steps；CPU4线程、CUDA仅冻结编码。单次wall<=3h、编码<=1h、输出<=64MiB、磁盘保留20GiB、GPU peak<=启动空闲80%；不保存token cache或新模型。累计预算仍服从原24 GPU-hour/48 wall-hour，不重置。保护release/dist/hash不变。

## 实施记录与审查

1. TDD：38D顺序/无效token排除/NaN/单token/空输入，154D严格宽度与训练期imputer、冻结编码、真实祖先、三policy屏障和gate。
2. 新research入口 `scripts.research.representation_v2.joint_probe`，不改旧S2/S4协议；先协议代码commit，再运行一次。
3. terra/medium独立只读审核：特别检查目标域拟合隔离、实际feature宽度、无效样本、冻结清单、B重放与资源上限。相关research及candidate/event回归；最近全套646 passed/5 skipped，完整重跑约21min，本次不声称重新跑过全套。
4. 输出 `outputs/challenger_70/representation_v2/amendment_a3_joint/`，更新README/registry/report、commit，清理仅本轮临时物。

Ruling：采用全candidate mean/std32D而非S1有序128D，以限制新增维数；这同时改变pooling与融合方式，因此只能称联合方案检验，不能把增益单独归因于融合。代价是可能损失时间顺序信息。按用户整洁要求，本文件兼作实施ledger，不另建重复计划文件。

实施ledger：38D合同/冻结编码/154D真实LR+LGBM、严格嵌套入口已实现。空token coverage测试先失败后修复；B候选与逐人指标回放两个测试先失败后修复。相关research+candidate/event共183 passed、4条预期全缺失列imputer warning（无测试失败）。terra/medium独立只读审核无重要阻塞。

Final minor (deferred)：每个health report可附marker SHA；目前完整冻结文件/清单、每次重新验证的代码与最终summary已保存，暂不增加重复字段。不是隔离缺陷。

状态：代码/协议待提交后运行一次；尚无S5成绩。
