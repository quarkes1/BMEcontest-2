# A8 — 全天定位可行性探针（D0）

2026-10-14，按用户自主授权执行；设计、计划和执行账本合并于本文，避免重复文档。目标仍为正式事件F1≥.70，但D0不产生新的F1，不改变dist。实施方法：主代理直接执行，测试先行；使用executing-plans。

## 根据与取舍

A7固定PPG代理P1/P2均失败，停止在其权重与窗口上搜索。A6的候选复核已经尝试TCN、Transformer和种子集成，但其设计中T2全天定位尚未执行。选择先检验source-only FD动作证据的定位覆盖；相比立刻重训更大的Transformer，这一步不新增训练或raw读取。另一个备选是左右腕/饮水源域训练，暂不同时改动，避免归因混乱。

**限制：现有token仅227个候选所在会话，不是全部696个T24会话。** 必须报告所有97个eligible GT中的缓存覆盖与未覆盖，不把缺缓存当作阴性，也不把局部结果称全天性能。用于下一步的线索只是source-only动作头是否存在新增proposal信息，不能证明新的目标模型有效或无效。

## 固定计算合同

- 原A3 source-only FD encoder/head冻结，无目标拟合、归一化、PCA、早停或阈值选择。
- 输入仅已封存T1 token；校验seed-ensemble登记的完整header/NPZ SHA、source checkpoint SHA、source训练主体、时间轴metadata、sid归属。禁止读取V7/outer、raw数组、新建token缓存或修改旧缓存。
- 每有效5s token计算sigmoid(z·w+b)。固定阈值.5；同sid且同acquisition span，正token间隔≤120s合并；无效token、span变化、非5s连续时间立即切断。每run至少2个正token。候选边界为首末正token支持范围（center±2500ms），裁剪会话边界。不额外padding，不按GT修边界。run内允许低分有效token，但不能跨缺口。
- 候选与资源/输入身份先封存并提交，之后才能读取T24三份health truths与原候选；所有匹配≥.25沿用当前canonical开发口径，明确不是官方独立测试。
- 报告227/696会话、有效时间、候选数量/时长、各wrapper原候选/新候选/联合可达GT、新增GT与重复、未匹配candidate数量（不是最终FP）。不能调用verifier，不生成融合策略，不自动计算诊断event F1。
- D0只支撑后续假设；不能以GT选某组候选送进下一训练。下一目标监督dense head须另行固定结构及严格祖先OOF，复核采用预先指定对照；不直接将source探针阈值移入dist。
- 墙钟≤10min、进程RSS≤8GiB、证据≤16MiB、磁盘剩余≥20GiB；CPU数值线程2。原始读取0字节、optimizer steps0、outer calls0。

## 文件与实施计划

代码`scripts/research/representation_v2/localization/probe.py`与`__init__.py`；测试`tests/research/test_dense_localization.py`；输出`outputs/challenger_70/representation_v2/amendment_a8_dense_localization/`，不覆盖旧attempt。

Task1（纯计算合同）：

- [ ] 测试RED：两正token合并、单点丢弃、无效token切断、span切断、超120s切断、边界裁剪、阈值包含等号；真实SessionTokens与EventRef，不mock输出。
- [ ] 实现`proposals(tokens, scores) -> list[EventRef]`，拒绝概率非有限/越界/错位；GREEN并提交。

Task2（证据封存与只读报告）：

- [ ] `register`锁定代码/本文、source checkpoint与既有cache inventory、三份health truth证据、完整protected_snapshot；清洁工作树且新输出目录。
- [ ] `extract`校验注册与提交；逐会话source head推理并写候选，校验预算、生成seal与EXTRACTED状态，提交后方可report。
- [ ] `report`先拒绝未提交seal及篡改，按fold一对一/可达匹配分别计数；报告原候选与新候选的GT union、未覆盖GT及新增候选未匹配数，不写F1；状态REPORTED。
- [ ] 跑新测试和PPG专项回归，核对保护树及无新raw缓存，更新README并提交。

Review Focus：缓存只覆盖候选会话的选择偏差；无效token不能成为合并桥；正确区分多candidate可达和一对一匹配；未封存结果禁止打开truth；source概率未作跨域校准，失败不等于目标监督定位无效。

Ruling：最新自主授权覆盖设计/计划的审批，不再次询问；诊断仍不能作为发布收益。复用旧source cache而非重编码，本轮只检验有限缓存范围。
