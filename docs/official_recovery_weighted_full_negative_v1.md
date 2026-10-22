# WEIGHTED-FULL-NEGATIVES-V1 — 不改变有效先验的宏窗数据保留

## 假设与范围

最初全量负窗未加权使宏窗正类比例从CV约.33降到.0318；恢复抽样已得到官方支持，但当前每个无餐session只保留1个负窗，丢弃大量背景动作。本轮保留全部旧标签宏窗，通过训练样本权重维持原每session正/负有效质量，检验更多负例覆盖能否减少假相似动作，而不再次压低正类先验。

只改变宏窗训练数据保留及其必要权重，沿用原HGB，不加入失败LGBM/集成、完整GT重标或新特征。micro仍canonical抽样、原LGBM与阈值；116D Context-v1、旧eligible verifier监督、LR C=.1、融合.25/.75不变。基准仍冻结A，完整160餐89/62/71与五折254餐138/80/116。

## 权重精确定义

对每个train sid，旧标签正窗P个、负窗N个：k=min(N,max(1,3P))。每个正窗weight=1；每个负窗weight=k/N（N=0无负行），所以该session有效负质量精确等于canonical抽样负窗数k。ignored=-1不fit，数组占位weight=1只为统一正权重接口。全train/每个subject-contained fold均保留同一有效正负质量；不额外归一化权重、不加class_weight。N、P只依赖此sid旧训练标签，不使用其他subject统计量，权重不能用score GT。

所有macro OOF与最终refit都用全部train标签>=0行及同一权重规则；`model__sample_weight`必须抵达真实HGB。micro不接权重。原imputer现在在全部macro训练行上拟合；HGB的bin构建与raw min_samples_leaf统计也相应改变，不能把本实验描述成“只改变损失且其他数学完全不动”。GroupKFold仍按其实际宏窗行数分配主体，可能与抽样基准分区不同；保存实际图，不能声称每个macro内折训练主体完全相同。候选窗口池、监督、外层/政策分区均不变。

## 评价与预算

不以AP代替目标，直接T24严格嵌套事件对照：3组train内三policy OOF，每producer独立四折宏/微OOF，全部9份fit/三policy seal+commit后统一health；通过才39主体原五折15policy fit全部seal后outer。参数选择算法/原admission网格完整GT目标不变。V7/outer不用于T24。

两门均：F1≥+.01、T24≥2/3/五折≥3/5严格提高、TP不减，FP增加时新增TP>新增FP。冻结A逐事件回放；微窗抽样及micro threshold每fit精确等于A。每session有效质量、权重min/max、输入/实际fit行与图、逐折/逐人TP/FP/FN、资源完整保存。五折过门才登记新版本全目标重训/EXE；不保证官方改善，不覆盖正式或第一试测。

每cohort≤2h，RSS≤12GiB，证据≤128MiB，余量≥10GiB；CPU库8线程。预计T24 120窗口/24meta、五折200窗口/40meta fit，但每宏窗fit实际读取更多行；不增加特征缓存或保存失败权重。完整Data/cache/代码/spec指纹、资格加载前后和fit后、66保护前后不变。新输出weighted_full_negative_v1_t24/_five拒绝覆盖、异常INCOMPLETE。

## 实施

复用macro_lgbm_event.py的显式hypothesis=full_weight；不依赖失败窗口AP来授权，不误用M1/ensemble门。新增weighted_full_negatives.py纯权重函数与准备角色；crossfit ledger增加可选对齐macro样本权重，只给macro pipeline/crossfit传入，None保持既有数学。父级T24→five按本hypothesis成功状态/ID/源/protected核对；宏窗数据源与冻结A既有Data/cache身份核对。

TDD：按session有效正/负质量等于canonical sample，零正/零负/ignored、未知标签、顺序非破坏；真实HGB权重传递与micro不受影响，None默认恒等；新hypothesis父级隔离与不覆盖。一次Luna medium只读审查，严重项一次RED→GREEN修复，提交后训练。失败关闭本权重合同，不按health改ratio/clip/权重放大。

Pre-flight：权重数组与data.window_train原序严格对齐，subject切片后再按labels>=0切片；每个OOF fit只收到其train行权重。per-sid规则不跨subject拟合统计，因此当前被评分subject的标签不影响其他fit subject的权重。

实施：纯权重与macro传递三项先RED→GREEN；真实Pipeline-fit spy验证subject/OOF切片的逐行权重等于原行身份且micro kwargs为空。真实常量HGB验证100行原正类比例.02，在权重下回到canonical质量2/(2+6)=.25；这是现有HGB sample_weight能力的额外验证，不冒称新功能RED。新的full_weight工厂/父级namespace/gate测试RED→GREEN。一次Luna medium审查无Critical/Important或Minor。代码冻结后运行事件实验与最新全套，在全套结束前不再扩展研究代码，以避免旧一次回归的混合快照问题。

结果：7e7ba06封存后完整160餐A89/62/71、F1=.572347267；W81/48/79、F1=.560553633（−.011793634、TP−8/FP−14、1/3提高），失败。120窗口/24meta fit、1058.17秒，源/资格/66保护不变，无五折/模型/EXE。只读匹配迁移诊断：失去13个原匹配GT（7旧eligible、6旧遗漏），新增5个（3旧eligible、2旧遗漏）；不是官方63FN归因。

全套：冻结376c369的代码全程不变，1108 passed/7 skipped/39 warnings，1326.80秒；7 skip为5符号链接权限、2原官方样例缺失。专项66项通过。没有新增特征缓存或失败权重；关闭本权重合同，不在health调整ratio。
