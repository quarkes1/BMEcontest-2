# DENSE-MIL-V1 — 全天目标域弱监督漏餐补充

用户已授权自主推进，不再重复请求计划审批。NEGATIVE-CLEAN五折失败后关闭mask参数面；本文件合并设计、writing-plans实施步骤与账本，在任何真实训练前冻结。main原地实施，不修改正式包或第一送测版。

## 为什么换方向

现有错误诊断有大量可达候选深负及无候选餐；继续调cap或标签mask不能解决。旧Transformer/TCN仅复核候选，旧source-only定位只覆盖227个候选session。本轮直接使用五个互斥val中T24的**全部有效micro窗口**，不依赖旧候选、不新增FD/PPG假设。目标是补充召回，不称已有0.70证据。

## 冻结合同

### 数据与弱标签

- 仍为T1三个16train/8health、每个train三折policy OOF，V7/outer完全不参与拟合或选择。完整160餐、严格整数IoU>.25一对一，缺窗/缺候选GT也保留。
- 输入47维既有gravity-aligned micro特征，15s窗/7.5s步长。读取完整T24 val唯一联合；拒绝重复窗/未知主体/非固定窗宽；按sid和时间排序，相邻start差非7500ms立即断span，不跨session或缺口卷积。无时间先验、无旧候选标志或GT字段作为输入，不读取raw ACC/PPG，不使用仅227session的旧token。
- 每个当前fit分区独立拟合normalizer：每主体按真实时间排序后，在**有效观测行序列**上等距最多4096行（每行对应固定7.5s观测步长，缺口不补样、不重复边缘行），训练行拼接后逐维nanmedian填补、std下限1e-4，标准化后clip±8；全缺维填0/scale1。绝不使用score/health行统计。
- 每个完整GT是一个positive bag：取当前fit主体、GT内中心点的所有可见micro窗口，跨session片段可以属于同一bag，但encoder从不跨gap。至少2行才参与loss；缺失/不足单独记录，不删评价GT。**只约束bag最高20% logits均值为positive，不把每行标1。**
- negative bag中心在连续span内每300s采一个，长度按当前fit完整GT时长列表确定性循环匹配；请求区间必须完全在span内且距当前主体任何GT至少300s，至少2个中心。负bag同样取最高20% logits均值为negative。无合格正/负bag即INCOMPLETE，不补健康主体数据。采样按类别各8个bag、先均匀选有该类bag的主体再选bag。

### 模型与计算

- 固定47→64 Linear/LayerNorm/GELU；5层残差Conv1d(64,64,k3,dilation1/2/4/8/16)，逐token LayerNorm/GELU/dropout .1；输出单logit。每层对无效padding归零，非BatchNorm，无跨样本统计。感受野63行，单侧31行。
- 从全span切256行body，左右各31真实上下文；仅预测body。edge只在真实span边界补零，训练positive bag边界不产生encoder padding。chunking须与全span eval输出allclose，不允许GT边界成为输入线索。
- 每fit从零训练1000 optimizer steps、batch16 bags（8正8负）、AdamW lr3e-4/wd.01、grad clip1；FP32、dropout .1，无early stopping。种子20261021+100×wrapper+inner（最终refit inner=3）。CPU线程8，CUDA优先；train forward最多32chunks/microbatch、inference128。固定预算而非看health延长训练。
- 初轮9个policy fit＋3个health fit，共12模型；每次模型独立，normalizer/weights不复用。CUDA峰值allocated<=6GiB、RSS<=8GiB、总wall<=2h、空闲>=10GiB、证据<=128MiB；只保存概率/身份/小型诊断，不存特征或中间权重。GPU OOM或预算越界写INCOMPLETE，不静默改架构/batch重试。

### 定位与主要比较

- 固定dense proposal：frame p>=.5的hits，连续span内hit间隔<=120s合成run，至少2hits；首窗start到末窗end，两端各扩60s且clip真实span。event score为run hits概率均值，不看GT。固定新候选NMS IoU=.3，按score降序/稳定身份打破平局。
- 主要arm为**已有A最终事件＋dense rescue**，不替换/移除已有事件。新增候选与任一已有同主体事件IoU>.1则拒绝，再按score/subject cap接受。threshold固定候选(.5,.65,.8,.9,.95,.99)，新增subject cap(0,1,2,3)；cap0明确等于baseline。只在当前train的dense OOF上选阈值/cap，F1/PPV/少新增/高阈值/小cap依次破平局。
- A精确复用NEGATIVE-CLEAN-V1已冻结的A baseline：health最终预测，policy侧三份A OOF scores用同wrapper已冻结A policy解码；不得重新计算A scores或训练新A。启动前复算三个health结果为89TP/62FP/71FN、F1=.5723472669。其模型在自己的fit主体内训练；A policy使用T16内OOF选定合法，但不用于新encoder监督。
- 各wrapper的新encoder在每个inner-fit独立训练，再预测inner-score，禁止先训T16 encoder用于inner OOF。三个wrapper的所有rescue policy封存提交后，才做T16→health refit。baseline图与新encoder/imputer/policy图共同证明health排除。
- 三组完整后唯一gate：较冻结A pooled F1>=+.02，至少2/3组严格提高，TP总数不减少。standalone dense仅诊断，不作为另一个可供挑选winner的arm。失败不改本轮objective/池化/decoder。
- 通过才另行登记39主体254餐五折验证；最终通过才新版本全目标refit、独立EXE、EXE-only/Python一致和完整回归，不自动晋级正式release或发送邮件。官方.521277版继续保留。

## 接口、实施任务、Review Focus

1. `scripts/research/official_recovery/dense_mil.py`：Span/Bag、完整micro缓存读取、gap安全chunk、train-only normalizer、weak bags、TCN、固定step训练/预测、proposal/rescue。测试`tests/research/test_official_recovery_dense_mil.py`先RED后GREEN：物理缺口、跨文件bag不跨卷积、normalizer排除heldout、top-k非整段标签、负区间buffer、chunk等价、rescue不改baseline/稳定tie。
2. `dense_mil_probe.py`：clean tree/全输入及66份保护SHA；精确回放A；按T1嵌套执行policy→seal→commit→health，预算/祖先/缺GT/失败拒绝覆盖，完整gate与逐人诊断。输出`outputs/official_recovery/dense_mil_v1`，不增加临时脚本目录。
3. 真实policy与health结束前禁止改算法/网格/源文件；通过走新五折，失败关闭本轮参数面。一次Luna medium只读整体审查，严重问题RED→GREEN；最后完整回归使用通过即清理的tmp_path策略控制scratch，文档及时提交。

重点：全会话范围而非旧候选缓存；GT仅决定训练bag和train policy；normalizer/模型不得见自身score主体；跨session的bag仅做输出pooling；卷积真实halo不得按GT截断；缺失GT始终保留分母；A冻结与新branch选择不接触health；不能把追加事件等同TP必不减。

## 账本

- 2026-10-21：按已批准的条件分支自主启动，不因失败W降低门槛。CUDA已实测PyTorch2.11.0+cu128 / RTX5060 Laptop 8GiB可用；选择微窗统计TCN而非重新抽取raw encoder tokens，减少IO与打包负担。
- Ruling: 首轮采用保守追加分支而非全替换baseline——避免丢失已确认的检出，代价是不能修复与baseline事件明显重叠的边界错误；该限制固定，不根据health放宽。
- Ruling: 使用bag top20%而非餐内逐片硬标签——吻合进食/停顿交替，代价是meal边界只能由固定proposal聚合近似，未必达到IoU要求。
- 实现预检：冻结A逐事件精确回放，完整160餐结果89/62/71；全输入391444窗口、702个sid、1440物理连续span，与旧227候选session缓存不同。27项相关专项通过（本阶段14项），含CPU/CUDA合成fit与p allclose、chunk/fullspan等价、normalizer按物理时间采样顺序不变；尚未启动真实target fit。
- Ruling: compact A scores没有完整stream细节——仅在每份文件candidate geometry唯一且health逐事件精确回放时，使用原冻结score和geometry复原admission；代价是若将来出现同几何多stream重复行必须拒绝，不猜测tie行为。
- Final review：Luna/medium无其他严重问题。Final: Ruling: 审查将“时间等距”歧义列Important；本方选择明确为按有效观测时长/行均匀，而不按跨缺口的calendar时间取最近窗——后者可能重复缺口边缘，给缺失时间加权。所有真实target训练前修订合同并补gap长度不变性测试；算法不变，该性质既已满足，不冒称RED→GREEN修复。代价是较长有效span在同一主体统计中权重更高，这是归一化目标而非对未观测时段估计。
- Final: minor (deferred): baseline loader未自动检查既有COMPLETE state/完整policy seal；当前真实训练前另外手工核对，A仍逐事件精确回放，全部输入SHA冻结。后续泛化复用入口应加强来源门，不为该小项改动当前代码。
