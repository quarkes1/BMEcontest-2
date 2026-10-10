# A3-S12 — 条件后备：冻结手势证据的时序形态

状态：S11已完整失败，进入实现；**尚未运行S12**。只在S11完整运行且事件gate失败后启动；
S11 incomplete时先处理工程原因，不将未完成冒充算法失败。S11通过则暂缓本路线，
优先执行该配置的更广inner验证。不能按S11某一折漏检病例改本schema。

## 目标与研究判断

目标仍为正式事件F1≥.70；超过当前发布.6514285714达>.03时才提请发布审批。
本实验不得直接改变dist/release，也不得把重复T24 health当独立测试。

既有证据：S0在FD手势标签上学得到信号；S1表示有候选排序增量，但S2标量融合、
S5/S8隐向量拼接和S7小型顺序头均未满足稳定事件收益要求。S10只改善训练目标，
不能证明目标事件泛化。上述结果没有证明“手势概率的出现模式”不可利用。

选项比较：

1. **先做冻结手势证据形态（本设计）**：直接复用源分类头，检验隐向量均值/方差
   是否丢掉稀疏高分手势的密度、成串与间隔信息；无新增神经网络训练。
2. 扩大token TCN感受野/容量：现有33采样点局部感受野约1.32秒，可能不足以捕捉
   完整5秒动作，但这是容量假设，不是已证明瓶颈；需新源训练与对照。
3. 更强事件TCN/Transformer或多尺度raw分支：有长程能力，但目标受试者少，
   需严格partition-contained训练与预算诊断；不以换模型掩盖输入或优化问题。

先选1是为了更换**表示方式**，不是因为禁止更复杂模型。若失败，关闭固定schema，
下一设计优先比较更大感受野/多尺度表示，不在重复health上手调本schema阈值。

用户自主研究授权覆盖该有界研究。沿用brainstorming的方案比较，但不再逐项请求
设计批准；发布审批和outer评估纪律不变。本文兼作规范、执行计划与后续台账。

## 固定数据及推断合同

ID `RV2-A3-S12-001`。使用S0同一封存encoder及source head，不适配、不校准source
概率、不改变unit-free预处理。checkpoint SHA必须仍为
`c6276837b1366117075e7e8a7504e084926cbfc5d80b98b863e442255995efc4`。
source head输出只是源域手势证据，**不是**校准后的目标进食概率或真实进食次数。

每个候选使用已有5秒/25Hz raw token、sample_mask、token_mask及真实center_ms。
使用相同unitfree_bags；仅有效token送encoder/head，稳定sigmoid转概率。
冻结权重eval/inference_mode，不引入源验证或目标health拟合。
无效token打断连续run和lag配对；绝不跨candidate/session/subject连接。
不使用GT选择某个token、峰、阈值或统计量。

## 预注册24D形态schema（顺序固定）

1. 有效token概率：mean、population std、max、q25、q50、q75、q90、q95（8D）。
   分位数固定numpy linear插值，std单token为0。
2. 固定阈值.2/.5/.8下的above fraction（3D，比较使用>=）。
3. 同三个阈值下，各输出run count与longest run / valid token count（6D）。
   run count直接计数，不称作真实进食次数。run相邻必须都是有效token，且中心差
   >0且≤5000ms；无效token、超过5秒缺口或未达阈值均打断。零run的两值均为0。
4. lag1与lag2相关系数（2D）。只允许完整连续有效链；lag2的中间token也必须有效。
   中心逐步差均>0且≤5000ms；至少2个有效pair且双方方差非零才计算，否则NaN。
5. 连续有效lag1对的mean absolute probability difference（1D）；无pair时NaN。
6. token有效比例、全部真实token的sample观测比例、log1p(valid token count)、
   log1p(candidate duration seconds)（4D）。前两项独立暴露覆盖，不把缺失填成negative。

全部无效时前20D为NaN，后4D按定义保留（有效比例0、valid count0）；
没有真实token时sample观测比例0。duration必须正，输入必须按真实时间递增；
重复/逆序时间或非有限有效概率fail closed。输出宽24、不含inf。
不足5秒的尾token不增加run长度秒数；本schema只报告token计数，不伪称真实持续时间。

## 严格nested事件对照

复用既有T24三wrapper与每wrapper三policy OOF + final health refit：

- B：原116D verifier，完整重放封存S2候选、分数、policy、事件计数。
- R：S8同一冻结FD251D四段mean/std表示（不是random）。
- F：原116D + 固定24D形态 = 140D，独立相同LR/LGBM/.5 blend verifier。

三臂候选、OOF producer、阈值/admission grid和decoder相同，不松开candidate control。
120 producer fits、72 event verifier fits、0新encoder/readout神经训练步。
每partition verifier只fit train，score subjects及其raw token不参与拟合；
source外部27人祖先与目标id命名隔离。所有wrapper policy选择完成并hash冻结后才health。
不能在health挑F/R、调形态阈值、换pooling或反复尝试组合。

固定事件gate：F pooled≥B+.015、F>R、≥2/3fold胜B、TP不减、FP减少或新增TP多于新增FP，
同时完整性/资源通过。通过仅允许更广inner验证，不自动outer。失败完整封存，
没有第三个临时臂或阈值网格。逐人median/IQR/p10、improved/worsened必须报告；
它们是诊断而非临时新增promotion规则。

## 工程、验证与资源

研究代码放`scripts/research/representation_v2/gesture_probe.py`，不在根目录放脚本。
优先复用joint_probe引擎、ordered_joint_features的合同风格；不得修改发布src。
若通用引擎需要注入feature宽度/score callback，保留原默认逐位兼容并以回归证明。
研究输出`outputs/challenger_70/representation_v2/amendment_a3_gesture_morphology/`。

保存plan、schema、source/hash身份、完整producer/verifier祖先图、分区scores、
policy冻结marker、每fold/逐人事件、summary/state、report；结果写registry时
outer_f1/event_f1=null，inner health字段另列。禁止覆盖attempt和保留token缓存。

停止预算4wallh/2workh、CPU4、CUDA可用则用、输出≤64MiB、磁盘剩余≥20GiB、
CUDA peak≤初始空闲80%。这只是单实验停止条件，不是用户硬件/模型复杂度上限。
异常记录incomplete，不据部分折下科学结论。

执行计划：

- [x] 1. 确认S11正式完成且gate失败，记录S12启动依据；不读取GT病例设计规则。
- [x] 2. TDD：固定24D顺序/常量/单token/全无效；阈值等号、gap/invalid断run、
  lag2完整链、乱序拒绝、无GT输入；source head冻结及CPU/CUDA allclose。
- [x] 3. 实现研究入口，独立Luna medium审查；相关测试与diff检查后commit预注册实现。
- [ ] 4. 单次严格nested运行；报告事件/逐人/覆盖/资源，不据中途health改配置。
- [ ] 5. 全仓回归、README/registry更新；清理本轮产生的临时/可重建缓存，commit。
- [ ] 6. 通过→更广inner注册；失败→关闭本schema并自主设计更强感受野/多尺度路线。

预检：24D+116D=140D；R保持251D；所有概率阈值固定，不依赖event policy。
本计划不存在source/target新训练或early stopping依赖。尚无S12实验结果。

设计审查：Luna/medium只读审核未发现重大问题。实施必须专测三臂不同feature width、
R/F两臂source27祖先以及全NaN形态列的train-only imputation，不能为拼接方便统一填0。

启动依据：S11封存3053f66，12分区完整，F .737430未胜R .741935；不以部分门槛通过放行。
实现直接复用prepare callback，无需改动joint引擎；原有S5/S8/S11路径不变。
除B精确重放外，R也逐分区绑定S8封存8206cc6账本与评分（rtol1e-6/atol1e-7固定数值容差），
防止参照臂实现漂移。S11 summary/state须与封存commit逐字节一致且完整失败方可启动。

实施验证：模块/冻结参照replay缺失RED→GREEN；17项S12测试通过，含真实tiny
crossfit+三臂verifier、score GT改变不影响评分、CPU/CUDA一致性及source权重/梯度不变。
tiny fixture显式改为10秒decoder几何（测试数据只有10秒窗），实际实验仍用原登记配置。
相关表示研究162 passed、20条预期/既有imputer提示；全仓回归将以本次代码重新执行，
不冒用S9/S10的704项证明。Luna/medium整体只读审查无阻塞。
Minor（暂缓）：S11 state.outer_calls未单独重复检查；已对state逐字节commit绑定，
且summary outer_calls检查为0，不影响本固定输入有效性。
入口 `python -m scripts.research.representation_v2.gesture_probe`。
