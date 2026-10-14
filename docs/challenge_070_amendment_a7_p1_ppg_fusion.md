# A7 P1 固定PPG候选融合实验

2026-10-14。用户已授权自主实现、实验和有实际提升后的交付，不再逐计划审批。基于P0提出的训练假设，注册后执行；不把P0描述差异算作提升。

执行版本：第一次注册后独立代码审查发现短餐matcher导入路径错误和写入完整缓存后的预算检查缺失，未进行任何缓存提取或拟合即终止，保留小型预检证据。修复、测试后在`amendment_a7_ppg_fusion/attempt_02/`重新注册。短餐沿用既有duration<10min，先对全部GT进行一对一匹配再统计短餐，禁止对子集重新匹配改变归因。

attempt_02的完整缓存696会话/97599窗口已成功封存，原2034候选的Q/P与P0逐值一致；随后拟合前配置比较因tuple/list JSON表示差异终止，无模型拟合或新评分。修复为规范JSON比较（全部参数值完全相同），在attempt_03注册；校验原始SHA、解析代码和完整缓存seal后复用该无标签缓存，不重扫波形。缓存来源和前次真实I/O独立记录。假设、模型和gate未改变。

## 问题与固定控制

P0三个health分区的during−pre频率真负差值方向一致，但相同base分数箱样本不足。P1回答固定PPG代理能否在严格跨人训练中改善最终事件，而非重新搜索特征。

- B：重建原116D Context-v1 verifier，要求与每个原T1验证候选/base分数复现一致。
- Q：相同LR(C=.1)/LightGBM及固定.5 blend，输入116D+固定Q9。
- P：相同LR/LightGBM，输入116D+Q9+固定P6。
- Q/P均只在至少一维P有限的训练候选上训练，预测仅对同条件候选替换B分数，其他候选逐值回退B。相同可用条件使Q/P比较包含同等选择影响。每partition可用正候选<10或负候选<30时两组都回退B，不改变规则。
- 不搜索融合权重、模型参数或特征子集；两模型均沿用既有estimator工厂及seed+20/+21。缺失由各partition模型内既有imputer处理，保留质量覆盖；不从验证主体拟合填补或标准化。
- producer、候选阈值和policy网格沿用A6冻结config。政策只从每wrapper的三个policy OOF预测选择；三个wrapper政策全部提交封存后才读取health结果。

## 数据和隔离

复用A6 manifest的24主体/696会话及12个划分，每个partition都重新调用build_candidate_ledgers(train,score)。禁止将旧12份验证OOF拼起来训练。上游producer和verifier仅在partition train主体学习，验证主体不参与其上游训练。

PPG逐会话完全确定，无学习，允许跨partition复用。另注册全部696会话原始SHA和PPG代码schema。P0的165个会话已保留全部窗口（由(valid_packets−tail_packets)/50与保留数严格匹配），可复用；其余531会话重新流式提取完整窗口。原输入SHA必须与注册值一致。未知sid/非T24直接拒绝，不伪装缺失；合法会话没有有效窗口才返回缺失。整个696会话的PPG统计先无标签封存提交，再开始模型训练。

T24之外不读取raw、GT或评分。本轮没有outer/V7入口。若P1支持提升，下一独立注册阶段才验证V7；最终dist晋级仍须完整正式评估，不能按T24 F1直接替换。

## gate与必报诊断

固定主假设P，Q为采集/运动控制。只有P−B>=.015且P−Q>=.005、至少2/3 health分区不差于B、TP>=B−1，且FP下降或新增TP数量大于新增FP数量，才认为值得下一验证。任意gate失败保留结果，不补做融合网格或health驱动rescue规则。

报告B/Q/P pooled TP/FP/FN/F1、逐wrapper方向、逐人指标及短事件、可用/回退比例、训练正负候选数、拟合数、墙钟/峰值内存和缓存字节。B旧T1总体复现目标.7213114754（66TP/20FP/31FN）；复现失败即INCOMPLETE，不解释算法收益。

## 实施与预算

代码集中`scripts/research/representation_v2/ppg_fusion/`：contract（只读清单和注册），cache（完整波形统计和聚合），probe（嵌套拟合/政策封存/health）。不修改P0注册代码、A6或release代码。

先写测试再实现：未知主体/缺缓存拒绝、P0完整窗口复用判断、无效P回退逐值一致、训练组别隔离、门槛双组同步回退、B复现差异拒绝、未封存policy拒绝health、NaN/混合schema身份检查。

CLI register→cache→提交→policy→提交→health。raw注册额外哈希I/O单独记账；cache四进程每数值库1线程，<=30min、<=16GiB。拟合三wrapper并行，每进程数值库4线程、LightGBM沿用n_jobs=1；<=6h墙钟、<=32GiB总RSS。新增持久证据<=128MiB、至少保留20GiB磁盘。不增加神经模型/GPU工作来加速本轮文本解析和小型verifier。

原始波形不持久化，producer中间结果只在运行内存持有，不保存训练模型或大窗口特征副本；政策、必要评分及身份哈希保存。失败INCOMPLETE不覆盖；模型有实际晋级价值才进入后续部署refit。固定类别/参数对照完成后更新README和报告并提交。
