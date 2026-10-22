# COMPLETE-SUPERVISION-V1 — 完整餐次监督的端到端闭环

2026-10-22登记。用户授权自主推进，不需再询问例行审批。目标是独立送检版有可靠事件F1改善；正式release、第一试测、dist/submission不覆盖。

## 为什么单独检验

已发现旧监督仅使用单文件完全包含餐次，完整39主体为254餐、原eligible153餐。修复训练抽样与OOF已获得官方反馈支持，但当前基准训练仍保留旧窗口/候选标签。此前完整窗口重标在窗口AP探针失败，完整事件标签单独替换也未过门；两者都没有在修复抽样/OOF与全候选池后完成完整监督闭环。

M1已给出AP提高而事件下降的反例，因此原AP失败不足以替代这个新整体合同的事件验证。保留原失败记录，不重新判通过。这次同时统一窗口目标、micro训练侧选阈目标及verifier目标，明确检验整体合同/交互，不能归因某一个分量。完整餐段仍是弱动作监督，可能反而降低识别能力；不保证成功。

## 固定模型与训练

- 保持原HGB宏窗63D、LGBM微窗47D、116D Context-v1、LR C=.1、融合.25/.75、原seed与原全部网格。不开新网络、不加权全负例、不使用失败LGBM宏窗/集成、不搜弱正权重。
- 每个当前fit分区仅用其完整GT，按sid所属subject对全候选窗口重新标注：macro overlap>50%为1，micro overlap≥50%为1；完全无重叠且距所有GT≥300s为0，其他−1。沿用既有corrected_labels；跨文件GT按完整subject时间区间判断，不因文件截断丢弃，不把同GT按sid重复计评价分母。
- 每个当前fit分区仅采样一次，沿用canonical每sid负窗≤max(1,3×正窗)、seed20260901。每个window OOF再在其fit subjects行拟合，不读取其score labels作选择。旧/新标签迁移与实际采样身份SHA均保存。微窗positive_middle_fraction必须None，拒绝旧eligible截取机制混入。
- macro/micro各独立四折subject OOF；在当前train的micro OOF上，以完整train GT选原micro threshold grid：budget=3×max完整GT数；可行项按recall、较少候选、较高threshold选；全不可行按F1/PPV/较高threshold。使用strict IoU>.25、subject完整GT一对一规则，不重复计跨sid GT。这个选择器只能看到train，不见health。
- meta训练标签对每个train OOF候选独立判断与完整train GT是否strict IoU>.25；同GT多个候选可各为正，最终评价仍一对一。score候选的GT标签不得用于fit或policy。完整目标只作用当前train；score窗口特征/候选生成不能读取score GT。
- 最终admission/event policy仍原choose_complete_policy、原网格、完整train GT目标；不修改可行性fallback，不趁本轮扩大阈值面。候选数/geometry与旧基准可能变化，不能声称冻结候选或分数。

## 验证与交付门

T24原三组16train/8health、完整160GT。A复用negative_clean_v1冻结A89/62/71、F1=.572347267并逐事件回放；C为本完整监督合同。每组train内三policy OOF，每producer内部独立四window OOF；全部9份fit/三policy seal并提交后才统一health。V7/outer排除训练及选择。

T24 gate：F1≥+.01、≥2/3严格提高、TP不减；FP增加时新增TP>新增FP。通过才原39主体/五折/254餐，A冻结138/80/116、F1=.584745763；全部15policy fit及五policy seal后才outer，同门但≥3/5提高。父级必须是本hypothesis的成功T24、源/protected未漂移。重复开发证据，不称新独立outer或官方F1。两门通过才登记全目标模型/策略/独立EXE精确版本；EXE-only验收及冻结最新回归后交付。

## 预算与组织

复用macro_lgbm_event.py显式hypothesis=full_labels；纯准备/选择器放complete_supervision.py，可选micro selector注入crossfit ledger，None保持旧默认。无新依赖。预计T24 120窗口/24meta、五折200窗口/40meta fit；每cohort≤2h、RSS≤12GiB、证据≤128MiB、余量≥10GiB。CPU数值库8线程。新输出complete_supervision_v1_t24/_five；拒绝覆盖、异常INCOMPLETE。保存分数/实际图/标签采样身份/逐折逐人TPFPFN与资源，不保存失败模型/特征缓存。

完整Data/cache/代码/spec SHA、资格加载前后及fit后、66保护前后核查。A来源Data/cache核对既有冻结plan；不把已失败AP或M1作为成功父级。源与代码在policy/health间不更改，结果全完再更新本文/README/报告。临时scratch只清本轮可安全清理部分，既有被策略拒绝删除目录不绕过。

## 实施与审查

1. TDD：跨sid同subject完整标签，macro严格/微窗含等号、5min buffer、train/score隔离、canonical采样SHA；micro选阈完整GT只计一次、严格IoU边界；meta完整标签；None选择器默认恒等、score GT改变不影响新候选/特征。
2. 严格图：custom micro selector的threshold祖先标明当前train subjects与其micro OOF，不沿用仅旧eligible owners的窄标记；policy显式连接三个producer score节点。review一次Luna medium，只读、严重问题一次RED→GREEN修复。
3. 提交后policy→seal/commit→health；过门才five。相关回归与新模型需要时最新冻结全套；不反复为无runtime变化的每个探索阶段并行整套回归造成混合快照，最终交付仍要求冻结代码全套通过。

Ruling：这是完整监督合同的联合检验，而非三个独立效果估计；若失败关闭当前硬标签合同，不在相同health改软标签权重。代价：即便成功也需后续消融才能归因各分量，但不影响按整条管线验证实际F1。

### 训练前实施记录

- 新选择器/准备函数及可选注入先见缺失接口的失败测试，再实现通过；相关回归72 passed（10.69s）。默认选择器与原候选/特征恒等、完整训练标签/采样身份、严格IoU及跨sid完整GT一次计数已覆盖。
- 真实数据只读预检（三组各16个训练主体）：macro采样行数/正例分别18385/6651、18226/6841、19117/6738；micro分别43438/14862、43035/15310。没有训练或访问health GT作选择。
- Luna medium只读最终审查：无Critical/Important。Minor deferred：尚无直接断言custom selector图的parent列表、且完整pipeline的score GT变更不变性仅由准备函数变更测试和callback隔离测试间接覆盖；本轮不追加调参或重新审查。
- 最近冻结整套回归1108 passed / 7 skipped属于376c369，不作为本次新增代码的整套验收；本轮上述72项为当前相关回归。最终若交付新EXE，须冻结最新代码另跑全套。

### 结果：未通过（2026-10-22）

8532b5c封存全部9个训练内分区后统一评分。A89TP/62FP/71FN，F1=.572347267；C93TP/70FP/67FN，F1=.575851393，差+.003504126、TP+4/FP+8，仅1/3折提高。逐折C为.610169/.547368/.563636，A为.594059/.555556/.568627。120窗口/24meta fit，含阶段间隔382.95秒；输入、资格及66保护文件一致。

未达到+.01、至少2/3折改善及新增TP大于新增FP的门；关闭本硬标签合同，不做五折、不导出、不尝试在相同health调软标签。完整标签确实恢复了正窗，但没有形成可靠事件收益，不能将它当成第二份改进程序。
