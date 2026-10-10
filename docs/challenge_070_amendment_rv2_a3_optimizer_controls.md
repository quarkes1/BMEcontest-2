# A3-S10/S11 — 有界优化对照与条件事件验证

目标仍为正式event F1 .70+；用户授权自主选择研究方案，发布需人工审批。
S9发现固定train BCE .221839→.261428，target梯度不断图，但未证明source负迁移。
不先改变source loss或扩大架构，以两个独立单因素检验优化行为。

## S10：train-only pilot，ID RV2-A3-S10-001

固定S9同一个S8 fold0_policy0训练分区（11人），8个producer OOF fit重建同一train候选。
初始化encoder、scaler、head必须重放S8/S9固定初值；仅valid运动候选进入辅助训练。
FD27 train bank、架构、source_weight=.25、学习率、AdamW、1000steps、seed20261011不变。

两臂均执行完整预算，不据中途loss延长或early stop：

| arm | target batch | global clipping | 其他 |
|---|---:|---:|---|
| unclipped | 4 | 无（仍检查有限梯度） | 与S8一致 |
| larger_batch | 32 | norm1 | 与S8一致 |

扩大batch仍固定1000 optimizer steps，因此target例数从4000变32000；
不把它归因为“纯降噪、相同数据暴露”，必须同时报告例数/时间/显存。
两臂各fit一次相同初始LR（共2次readout fit），不训练新的event verifier。
保存两份小型JSON权重/轨迹，token只在内存；不读取health、原V7 raw、outer或任何新事件指标。
loader先加载outer-train特征表然后立即切11人，不将物理加载特征表说成从未读过其他人字节。

固定train gate：subject-weighted BCE较initial降低>=.01，至少6/11人BCE不增加，
固定完整FD27 train bank的subject→class均衡BCE不比initial增加>.03，有限性/资源全部通过。
不以train gate证明泛化。source bank测量不使用source-val或目标health。
选择规则提前锁死：unclipped若通过就选它，不管larger_batch是否更好；否则选通过的larger_batch；
两者都不通过则**不运行S11**，关闭此优化修复分支，转向新的多尺度时序表示。
禁止第三个optimizer配置、组合两项改动、看health重选。

输出 `amendment_a3_optimizer_controls/`：plan、source_bank_identity、两臂adaptation、
train_comparison、selection、state、report。预算1wallh，输出<=16MiB，磁盘保留20GiB；
CUDA峰值<=80%初始空闲，CPU4线程。不覆盖attempt，异常留incomplete。

## S11：仅当S10通过，ID RV2-A3-S11-001

使用S10已冻结selection的**唯一配置**，不是把两个配置都送health挑winner。
沿用S8严格nested引擎：T24三wrapper，各自3个policy分区+final health refit，
120个producer、72个event verifier、12个initial readout、12000 optimizer steps。
B原116D，R冻结FD251D，F选中的优化配置FD251D；R不是random。
每个partition独立初始化训练，所有policy先冻结后才health，完整回放sealed S2的B。
S10训练人只属于开发T24；selection是开发选择，不声称T24 health是全新独立测试。
原V7及outer仍不参与任何拟合/选择/raw token；发布保持不变。

event gate沿用S8：F pooled比B至少+.015、胜R、至少2/3fold胜B，TP不减，
FP下降或新增TP多于新增FP，完整性/资源检查全通过。不能因为F比S8好了就放宽gate。
通过仅支持更广inner验证，不自动outer或promote；失败封存并关闭本优化配置。
记录TP/FP/FN、逐人结果、candidate AP/BCE、全部轨迹/weights及实际计算成本。
6wallh/3workh停止预算、输出<=64MiB、20GiB磁盘、CUDA峰值<=80%初始空闲。

## 实施计划/台账

入口 `python -m scripts.research.representation_v2.optimization_probe --stage pilot|event`。
S11输出 `amendment_a3_optimization_event/`，不可覆写S8/S9/S10。
source checkpoint不复制；NO DIST / NO RELEASE / NO PROMOTION。
正式可比F1较发布版提高>.03时请求人工审批；inner/train/诊断均不触发dist更新。

- [x] 1. TDD：裁剪开关的实际梯度行为、batch/预算边界、预注册选择规则；默认S8不变。
- [x] 2. 独立审查、代码/规范commit，执行S10，封存选择及证据。
- [ ] 3. 依选择执行或跳过S11；记录失败而不在health追加配置。
- [ ] 4. 全仓回归、README/registry、清理新增临时文件、commit。

Ruling：自主授权覆盖本有界两配置研究；本文件兼作规范/计划/台账。代价：训练gate可能
筛掉真正泛化有益的配置，但优先避免在尚未优化好的模型上消耗完整health/outer评估。

验证：新增功能均观察RED→GREEN；表示研究测试145 passed、4条既有imputer提示。
包括真实裁剪/无裁剪梯度、batch32例数、source subject/class均衡、固定选择优先级、
source遗忘与逐人门槛，以及incomplete/比较文件hash篡改拒绝。全仓回归仍待运行。
Luna/medium独立只读审查无阻塞。Minor记录：显存仅pilot总峰值，不做两臂独立归因；
source指标按每人实际存在的类别均衡，不额外假定每人都有两类；沿用固定FD27 bank。

S10：代码973698a，343.63秒，2000步完整执行。初始BCE .221839；无裁剪 .213449
（改善.008390，未过.01门槛）；batch32 .202209（改善.019629，9/11人不劣，
source BCE下降.004561）。固定规则选择larger_batch。无health/outer评分。
证据与selection hash先commit，随后S11仅验证batch32/norm1；全仓回归运行中。
