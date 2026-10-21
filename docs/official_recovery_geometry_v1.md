# EVENT-GEOMETRY-V1 — 冻结分数的全局边界校准

用户授权自主推进。两轮mask/MIL未形成可靠改善，继续保留官方.521277版，不打包失败模型。这里检验固定窗口/密度decoder是否存在可校正的时间偏差；不是重试S6的macro/micro相交裁剪，也不改变任何分数、NMS、cap或threshold。

## 训练前合同

- 每个最终接受事件仅做同一全局仿射变换：duration倍率∈{1/2,3/4,1,5/4,3/2}，中心偏移∈{−60s,0,+60s}，共15组合。不得加参数、分人规则或候选源条件。
- 整数毫秒：new_length=(num×length+den//2)//den，至少1ms且不超过sid权威session长度；new_start=(start+end−new_length)//2+shift，平移至session内以保留new_length，new_end=new_start+new_length。原事件须在权威bounds内，不默默修正输入。恒等配置逐事件精确回放；顺序/事件数不变，不再NMS/去重或重评分。官方接口本身不自动合并或去重，重复事件按原评估计FP。
- 参数只用当前train的冻结OOF接受事件与完整train GT选。目标F1最高，tie优先更接近恒等倍率、较小绝对偏移、稳定数值顺序。固定参数作用于score事件；GT不参与逐事件变换。
- T24：使用NEGATIVE-CLEAN-V1冻结A（89TP/62FP/71FN，160完整GT），其3×3 policy-score OOF按原冻结A policy解码；health逐事件精确回放。先3组参数全部seal/提交，后统一health评分。gate≥+.02、≥2/3组提高、TP不减少。
- 仅T24过门才做39主体254餐原五折：控制为NEGATIVE-CLEAN-FIVEFOLD-V1冻结A（138/80/116，F1=.584745763），不是原153餐release headline，也不把它与早期.590164跨合同混比。每折参数只用其outer-train三份A policy-score OOF，按已封存A policy解码；控制outer事件完全回放。五折参数全部seal/提交，再统一score。gate≥+.02、≥3/5折提高、TP不减。
- 评价严格整数IoU>.25一对一，缺候选GT仍在分母；报告TP/FP/FN、逐折/逐人、参数、原/新事件与输入SHA、祖先图。反复开发证据，不声称全新测试。零模型fit，不计算新score，不重训verifier，不访问raw传感器或新特征缓存。
- 数据来源/旧score JSON、CSV/index、所有src/scripts与本文SHA、66份正式与第一试测文件保护。新输出拒绝覆盖，异常INCOMPLETE；每阶段≤10min、≤16MiB证据、空闲≥10GiB。失败关闭本轮几何参数面，不能根据outer扩网格。

## 通过才交付

两道门都通过才登记全目标策略聚合/新版本交付。部署仍使用一次抽样＋subject OOF的四个基准模型，几何参数按五折train-only选择等权中位数（只在原离散集合取确定的中位数）；基准policy按五折A封存策略mode/median，无官方成绩调参。是否可以复用第一试测权重需核对micro阈值、factory/seed/OOF合同；不匹配则按同合同独立重训。

新EXE仅写新的dist/official_trials目录，不覆盖第一版或dist/submission。先登记精确版本/接口/模型与代码身份，再实现推理选项和打包。新输出必须与研究变换逐事件一致，通过EXE-only真实双文件、中文路径、无开发PATH、空/错输入与完整回归。没有新的官方分数前，不能保证隐藏测试改善。

## 实现任务与审查重点

1. `scripts/research/official_recovery/geometry.py`提供纯整数非破坏性transform/15选1，测试`tests/research/test_official_recovery_geometry.py`先RED→GREEN：恒等、整数边界、clip平移、未知sid/非法策略、事件数与输入不变、GT仅参与train selection。
2. `geometry_probe.py`复用frozen baseline回放、角色图与完整GT；T24 policy→seal→score，过门才five-policy→seal→score。所有score保持原存储值，不重算LR/LGBM或混合。
3. 一次Luna medium只读审查、严重问题RED→GREEN、完整证据后才判断门；更新README/报告、提交。当前并行回归是301ff95代码快照，新增geometry测试独立补跑，交付runtime修改后仍需最新全套。

## 账本

- 2026-10-21：已阅读S6旧审计，它只把dual-source区间与micro支持相交，在旧协议没有增加TP。当前是修复训练后的完整餐次、小网格全局宽度/中心校准，两者不等价。
- Ruling: 保持session内、全局同一变换——避免跨文件或按GT修边界带来的新自由度；代价是无法修复真正跨session的长餐。
- Ruling: 五折以最近冻结A为严格配对控制——保持原分数/事件完全固定；它不是第一试测EXE的逐位CV复现，不能据此报告官方增益或宣称已超过所有历史开发结果。
