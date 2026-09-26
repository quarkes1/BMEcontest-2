# RV2-A2-R1-001：低容量读出与candidate-conditioned SSL

状态：`R1_NO_SUPPORTED_SSL_BRANCH`。三折全部完成，0次outer，发布F1仍0.6514285714。

| 三折等权平均、all候选 | Candidate AP | Raw-logit BCE |
| --- | ---: | ---: |
| subject均衡常数先验 | 0.122538 | 0.373467 |
| M：仅coverage/duration | 0.200601 | 0.351474 |
| R：随机冻结表示+固定线性读出 | 0.397788 | 0.322740 |
| S：SSL冻结表示+相同读出 | 0.295870 | 0.354496 |

SSL相对随机表示AP为-0.101918，**0/3折提高**。每折R→S分别为0.29909→0.27770、0.43765→0.27475、0.45662→0.33516。不能进入更广泛SSL比较或以此启动事件stack。

## 解释

- 有序运动表示在这个小型probe中优于单纯coverage/duration；随机投影已经携带一些可用运动信息。
- 固定正则化读出组合的BCE明显低于R0的可训练事件网络，但两次试验还改变了seed、池化和权重，不是纯粹单因素“去掉TCN”消融；不能据此断言具体哪项变化造成改善。
- 当前候选条件下VICReg没有超越随机表示。三个训练bank各自封闭；不能用“loss下降了”或“没有collapse”代替held-out辨别能力。
- 这不是所有SSL的否定结论：训练人口、5s视图与增强可能更强调通用动作不变性，而非进食辨别。此解释仅是假设，本轮未进行归因消融。
- 下一条独立探索优先利用外部手势标签，而不是扩大当前无监督网络或更换Transformer。未知单位仍是直接raw-unit迁移的阻塞；无量纲输入可另立协议，但不是单位已验证。

## 隔离、资源与复现

24名原inner T受试者的GroupKFold(3)，每折16人fit/8人health；原inner V与outer始终排除。readout/sampler/scaler/SSL及候选生产器祖先均在 `fold*_provenance.json`，每折 `checks` 记录provenance/finite/optimizer/resource检查结果。第0折曾用于R0，全部指标仅是development probe。

每折SSL5000步，合计15000 optimizer updates；3次SSL、9次readout、30次macro/micro fits。SSL及训练诊断194.47秒，端到端465.13秒。新增产物约0.8MiB；没有持久token缓存；仅3个final checkpoint。

代码 `d089391ca67340b0f48ce298518fc22cc49637e3`。plan/spec SHA、bank keys hash、checkpoint SHA、subject split、每100步完整轨迹与health logits均保留。保护快照与正式发布一致。没有选择best epoch、调C值、重跑seed或掩盖负结果。

新模块及研究回归测试80 passed；其中合成smoke和测试成本不包含在上述正式运行秒数内。前置terra/medium审核发现的祖先检查与gate完整性缺口已在训练前补齐并测试。

外部数据只读复核见 `external_data_review.md`。数学意义的无量纲迁移作为后续独立A3，不回填为本轮或旧FD Q2/Q3成绩。
