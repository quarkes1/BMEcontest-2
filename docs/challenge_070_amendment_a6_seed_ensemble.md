# A6 — 固定三种子拟合稳定性对照

2026-10-13，用户已批准短设计的实施与实验。目标：检验固定三种子平均是否缓解单次拟合不稳定，同时保留真实事件F1收益；不把诊断结果当成种子唯一因果证据。

## 冻结合同

- 只用原T1 manifest 的T24、三个wrapper与各自三个policy split。V7与outer不参与本轮评分、训练或选择；共享特征表物理整表加载后过滤，raw eligibility处理前限定T24。禁止重开V7 gate、更新dist或自动发布。
- split_id=`4*wrapper+inner`，health的inner=3。原种子=`20261012+split_id`；新增种子为原种子+100000、+200000。同分区TCN/Transformer配对采样，不挑最优种子。三者在**同一分区**内概率等权平均，禁止跨wrapper直接平均。
- 原source-only FD encoder、候选生成、116D基础复核器、输入预处理/上下文、结构、FP32、每模型1000步、batch16、AdamW .001/.01、clipping1及.75 base+.25 motion均保持原样。单模型原臂与原pooling臂只复用已提交证据，不重训。
- 每个分区重建训练OOF候选与held-out候选，重建基础分数；held-out候选顺序/标签/主体与原T1证据完全一致，基础概率及原checkpoint时序分数在atol1e-6/rtol1e-5内复现才训练新增种子。原checkpoint训练主体与当前训练主体必须完全一致。训练归一化重新计算并核对原checkpoint，不能复用其他分区统计量。
- 从原T1已验证token缓存只读构建序列，校验source encoder、原raw checksum、元数据与NPZ SHA；缺失或不匹配即停止，不偷偷重新编码，不新增token缓存。持久派生缓存总限额沿用2GiB，新增证据/模型限256MiB，磁盘保留20GiB。
- policy阶段完成全部9个split后，分别对ensemble c/t使用原choose_policy/grid；b/p复用原对应policy。所有wrapper policy提交冻结后，才运行三个health split。每个health模型的所有训练成员排除该health人。各单种子分数只作稳定性诊断，不单独择优解码。
- “不改threshold/cap”指选择算法与搜索空间不变，不是把原单模型的数值策略强加给不同分数尺度的集成。新策略仅由训练侧OOF选择并冻结；本轮估计同流程下的集成收益，不是固定旧策略的纯分数反事实。
- 实际新增拟合总数：120 producer、24 base、48 neural、48000 optimizer steps。CPU线程4，优先CUDA；神经阶段累计2h、CPU1h、序列读取/组织1h、实验墙钟4h，每头20min。总耗时不包含实施和全仓回归。不得通过改batch/步数/种子绕过失败。
- CLI分register/policy/health三阶段；登记代码、输入、spec及保护快照并提交后训练，policy与health间必须提交。已有attempt不覆盖；异常标记INCOMPLETE，不自动重试或更换配置。

## 评价及决策

报告incumbent、原单TCN/Transformer和三种子TCN/Transformer的aggregate event F1、TP/FP/FN、PPV/recall、短餐、可达GT、逐人median/IQR/P10及改善/恶化人数。比较单模型与ensemble各wrapper方向；同时报告三种子时序概率std、有效真/负候选分布与AP/BCE，不能把AP当事件收益。

TCN为预注册主检验；工程进入下一阶段条件：ensemble相对incumbent至少+.01 F1、相对原单TCN不下降、至少2/3 wrapper相对原单TCN非下降。Transformer为次要证据，不能替代失败主检验。达门槛只支持申请下一阶段，不自动调用outer。微小改善按开发信号解释，不宣称科学突破。

T24已经反复开发使用；本轮不是独立测试。三种子平均改变的是估计稳定性与总计算预算，不能单独证明V7失败源于某个坏种子，也不能证明训练人数变化无影响。失败后记录结果，不在同一轮改校准、fusion、cap或threshold。

## 文件与执行记录

代码集中在`scripts/research/representation_v2/transformer/seed_ensemble.py`，测试在`tests/research/test_representation_v2_transformer_seed_ensemble.py`，输出在`outputs/challenger_70/representation_v2/amendment_a6_transformer/seed_ensemble/`。无根目录脚本、无新外部依赖；更新README，保留必要模型与分数证据，及时提交。

实施顺序：测试RED → 实现/GREEN → 独立审查/全仓回归 → 登记提交 → policy训练与冻结提交 → health训练/报告 → 验证保护快照、预算、清理临时文件并提交。具体运行状态以输出state.json和本节后续记录为准。

实施记录：首8项测试RED→GREEN；Luna/medium审查发现ignored token缓存需登记锚定、policy OOF覆盖需显式校验、Transformer逐折方向需汇总。三项回归测试先观察RED后修复。修复前全仓回归主动中止，不记为通过；真实训练尚未开始。登记将原缓存每份header及NPZ SHA与完整文件清单一并固化，阶段前后检查修改、增加和删除。
