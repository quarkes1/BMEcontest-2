# A7 P2 — 冻结IMU先验的低维PPG残差

2026-10-14。用户授权自主推进。P1已明确失败，本次是新的模型形式；不重新搜索P1融合权重或health阈值。所有T24结果仍是反复使用的开发证据，不称独立验证。

## 根据与假设

P1可用正候选50个中40个评分被新131维模型压低；P丢4餐/救3餐，新增8FP。未使用PPG候选的分数虽然不变，重选全局阈值仍改变其去留。本次问：保留强IMU概率作为固定prior，仅拟合低维残差，能否控制FP成本并保留TP。

## 固定模型

- B直接使用冻结T1候选/概率/政策；不重新训练或挑选B。
- Q：logit(p_B)+β·[1,normalize(Q9)]。
- P：logit(p_B)+β·[1,normalize(Q9,P6)]。
- 保持同一any finite P的可用mask；不可用分数逐值回退B。模型预测仍使用同一原B政策，取消新增policy网格。全输入不可用时，事件与B完全一致；混合输入仍可能通过候选竞争影响未修正候选，须报告GT迁移，不作局部事件绝对不变承诺。
- β从零初始化，最小化主体均衡权重的Bernoulli NLL＋16/2·||β||²，包含intercept惩罚。权重=N/(G·N_group)，不平衡类别、不优化λ、不训练base slope。
- 每模型在其训练主体可用候选上拟合median填补与std缩放（floor1e−6），z裁剪±3；score侧只transform。缺失映射到训练median。base仅为logit计算clip至[1e−6,1−1e−6]，回退直接复制原值。
- scipy L-BFGS-B，最多500迭代，ftol1e−12/gtol1e−7；梯度显式验证。可用正候选<10或负候选<30，两分支同步回退B。无神经拟合或feature搜索。

## 严格训练拓扑

每个原12partition的train主体再分3个GroupKFold。每个内部OOF子折重新执行全部macro/micro producer、阈值选择及116D verifier，仅用inner-train主体，然后在inner-val产生candidate/base probability/label/Q/P。这3份真正全祖先OOF构成残差训练集。

最终score使用原T1对应partition的冻结候选和base分数；其原先训练祖先正是partition train。禁止用训练内base预测，禁止复用其他partition可能见过当前验证主体的base模型。最终政策原B全部复制且提交冻结后才运行health。

仅T24及已封存完整696会话PPG缓存，不新增raw读取、不读取V7/outer。来源：A7 P1 attempt_03 registry/windows/cache seal；校验所有上游文件哈希，P1注册输入与代码保持不变。

## gate、资源、收口

沿用P1主gate：P−B>=.015、P−Q>=.005、≥2/3分区不差于B、TP>=B−1，且FP下降或新增TP>新增FP。不因某一个residual系数方向好看而修改门槛。失败后收束固定PPG代理路线；成功才单独注册下一验证。

预计36个子折：360 producer拟合、72 base拟合、最多24 residual优化；无需最终重训base。三个wrapper并行，每数值库4线程，LightGBM沿用既有单线程。总拟合墙钟<=6h/RSS<=32GiB，输出<=64MiB，磁盘余量>=20GiB。

代码`scripts/research/representation_v2/ppg_residual/`；单测loss梯度、logit prior零修正、全缺失精确回退、OOF祖先隔离、median仅train拟合、clip边界、B政策不变、固定gate。register→policy→提交→health，注册只校验和复用已有完整缓存。报告B/Q/P事件指标、GT迁移、逐人/短餐、可用比例、系数/支持数/拟合数/资源；更新README并提交。不根据开发F1直接改dist。

执行前审查补充：policy最终资源记录与三份固定政策一起封存并提交，health先校验该预算哈希；truth与一对一匹配GT得失写入报告。初始注册未启动拟合，以INCOMPLETE保留；修正执行位于attempt_02，科学参数没有变化。
