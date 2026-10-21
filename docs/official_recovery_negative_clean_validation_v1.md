# NEGATIVE-CLEAN-FIVEFOLD-V1 — W方案同规模验证

用户授权自主推进；上一步W按预注册门通过（+.020986066，2/3折提高，TP不变），不是官方成绩。本文在本阶段任何拟合前冻结，合并spec/plan/ledger，main原地执行。

## 固定实验合同

- 原scope审计39主体、原五折、完整254餐；当前重复外层属于development evidence，不声称新独立泛化验证。
- 仅A/W：A为上一轮修复后一次抽样＋窗口subject-OOF meta；W只在当前训练分区遮蔽强重叠冲突负窗。原正例/灰区不改、不制造正例、不改meta标签。保留所有评分窗/候选及所有GT。
- 同一个outer-train先以sorted主体、seed20260901打乱后array_split三等份，生成三折policy OOF；每个producer在自己的policy-train独立四折窗口OOF、meta训练及窗口最终refit。完整GT阈值选择只看policy-score OOF，禁止outer-val归一化/采样/早停/调参。C=.1、blend=.25，原micro阈值网格、admission网格与事件阈值选择器，工厂及seed沿用NEGATIVE-CLEAN-V1。
- 五个outer的全部policy生成并封存提交之后，才能做任何outer评分；不看部分outer结果改代码或规则。最终每折outer-train四折窗口OOF→meta→最终窗口refit→outer-val。
- 每臂同完整分母，严格整数IoU>.25一对一；缺候选主体仍计GT。报告五折/逐人/候选可达性、TP/FP/FN、F1/PPV/recall、mask和采样身份、fit祖先。
- 唯一gate：W pooled F1较A至少+.02、至少3/5折严格提高、TP总数不得减少。失败关闭本轮mask路线，转全天弱监督定位，不根据outer改mask、阈值、融合权重。
- 预计400次窗口fit、80次meta fit；CPU8线程，<=2小时（含policy提交等待），证据<=128MiB、空闲>=10GiB；无持久中间模型/特征缓存。输入/代码/分区/资格及66份正式与第一试测文件前后SHA。失败INCOMPLETE、已有输出拒绝覆盖。

## 若通过才交付

新版本`official_negative_clean_v1`，目录`models/event_stack/official_negative_clean_v1/deployment`与`dist/official_trials/negative_clean_v1/EatingDetection.exe`。不覆盖正式或第一试测版。使用同39主体完整val联合；按W先遮蔽再一次采样，旧eligible meta标签、四折OOF，最终窗口refit。部署策略按五折W的train-only策略等权mode/median，micro阈值同法聚合；无训练内F1、无官方反馈调参。

PyInstaller自包含CPU EXE沿用官方无参数接口；构建须clean tree、源/模型哈希、无开发PATH自检、中文路径EXE-only真实双文件+空文件Python CSV字节一致、错误输入不覆盖、完整回归。当前原官方例子已缺失，保留skip缺口。只生成本地独立试测EXE，不自动发送、不晋级正式release/前端。

## 实现任务与Review Focus

1. 扩展现有paired helper仅接受显式A/W子集，不额外训练E；无效/重复/缺A拒绝。TDD覆盖。
2. 新`negative_clean_validation.py`复用全候选loader、分区图、完整评价和policy；TDD覆盖主体三折唯一覆盖、五折gate/TP退化/分母/失败拒绝覆盖。
3. 先policy→提交seal→outer评价；通过后再新版本重训/构建，失败不得导出。
4. 复核隔离、全GT、抽样一次、遮蔽不制造正例、来源保护、导出不覆盖；整体代码审查与全套回归、文档、commit。

## 账本

- 2026-10-21：W刚好通过前三组门；保留第三组退步。Ruling: 不把新A与旧.590164跨合同比较——本阶段两臂都重新在outer-train完整GT选择策略；代价是不能单独把所有绝对差异归于W。
- Ruling: policy划分用主体均衡随机而非按GT分层——避免借outer标签定分区；潜在代价是小分区餐次不均衡，固定后不重分。
- TDD：4项新增已观察RED→GREEN；连同上一阶段31项通过。只读全量预检确认39人/254完整GT/153旧eligible，macro285310窗、micro621199窗；所有窗口来自五个互斥val联合，不拼train缓存。
