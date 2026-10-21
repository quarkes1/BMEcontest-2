# MAX-FUSION-V1 — 冻结分量的确定性融合对照

用户授权自主推进。不同于旧E06学习型adaptive meta-fusion，本轮只有一个固定公式：p=max(pLR,pLGBM)，无新模型、无系数拟合、无新特征。检验平均融合是否压制单路证据；候选、两路原分量与上下游模型全部冻结。失败关闭此公式，不追加min/geometric/logit等变体。

## 合同与门

控制A为NEGATIVE-CLEAN-V1冻结A（89/62/71，完整160餐）；M使用同一原候选，max公式。先用三份train-only policy OOF重新选择M的事件threshold；admission保持对应冻结A的NMS、admission threshold、cap不变，不扩控制搜索。threshold目标完整train GT F1，tie PPV再较高threshold。只枚举独立通过固定admission的候选分数及0、1、nextafter(1,2)。三组参数全部seal提交后才对health运行M。

T24 gate：F1至少+.02、至少2/3组严格提高、TP不降。通过才对最近冻结A五折（138/80/116，254餐）实施同样训练侧OOF选择；五折gate至少+.02、至少3/5严格提高、TP不降。失败不部署。两道门都通过才登记精确全数据refit/推理配置与独立EXE版本；不自动改正式dist/submission。

每份score读取须核对train/score/forbidden主体、完整祖先图、字段/owner/唯一候选与0..1有限分量；原score须逐行等于.25LR+.75LGBM（数值容差1e-12），原A事件/指标须精确回放。数据CSV/cache身份与上一冻结plan核对；跨阶段保护及scope来源按geometry补强合同核对。所有src/scripts、本文与源JSON/数据SHA冻结，66保护文件不变。零训练/零组件重评分；公式重算必须如实计为融合变化，不称最终score冻结。

输出独立outputs/official_recovery/max_fusion_v1_t24或_five；拒绝覆盖、失败INCOMPLETE、每阶段≤10min、证据≤16MiB、剩余≥10GiB。结果是重复开发证据，不称官方提升。官方63FN仍无逐例信息。

## 实施

1. TDD纯max公式、非有限/越界/错baseline拒绝；固定admission下严格完整GT选threshold，角色/恒等回放。
2. 复用冻结score读取与geometry的来源/保护校验，独立policy→seal→commit→score；一位Luna medium只读审查、严重问题一次RED→GREEN修复。
3. 通过才五折，否则记录失败，更新README/报告、提交，不制造EXE。

Pre-flight：score生产器只提供已封存LR/LGBM与几何，融合不拟合额外模型，故不引入跨层stacking的OOF祖先问题。threshold属于训练侧全局policy选择，不能按health GT校正。现有_admission排序会随新score变化，这是被检验公式的确定性后果，保留原控制参数不等于保留同一入选集合。

Ruling：旧任务书禁止重启adaptive融合搜索；本次不启动该学习型方案，而对修复训练后的冻结分量实施一个固定无参数公式。若无效关闭，不扩为手工outer融合搜索。代价：不能代表所有融合架构的结论。

实施：4项新测试与geometry合计13项通过，含不同预测数的fusion gate；相关41项通过。Luna medium一次审查无Critical/Important。Final: minor (deferred): 保存的fusion policy祖先图沿用已冻结A policy，未再显式连接三个分量score文件节点；每个输入文件的完整祖先与精确角色均独立核对，且SHA冻结，当前未发现泄漏，但图形审计不够直接。
