# GYRO-POOL-MACRO-V1 — 宏窗增加长时旋转信息

2026-10-22登记，用户授权限时自主探索。目标为有证据支持的第二份独立送检EXE；正式版、第一试测、dist/submission不变。

## 证据、范围与固定表示

源代码features/macro.py只用session.acc生成62D，再加时间先验63D。imu_features.py的47D微窗中末12D为gravity-aligned gyro三轴各mean/std/RMS/power。现有macro HGB看不到较长时间的这些gyro特征，事件verifier只间接看到micro probability形态。固定增加长时旋转信息，比继续替换同一表示上的分类器更值得检验；这是算法假设，不是GT证明。

拒绝两个备选：重扫raw提取新频谱（耗时且新缓存大）；在失败随机树上继续调参或集成（仍无新信息）。本轮仅用现有、无标签的47D微窗输入，新增24D，不重训神经网络、不用外部数据、不混入失败完整重标。

每个宏窗[s,e)只取相同sid、start>=s且end<=e的完整微窗。固定取列35:47，按微窗等权计算12D mean及12D population std(ddof=0)，顺序[mean12,std12]。至少2个有效微窗，否则24D全NaN；输入微窗统计要求有限，禁止跨sid补齐或读取GT、概率、subject标签。完全不依赖微窗training labels，ignored微窗只要输入有效也可汇总。NaN只由当前fit imputer学习填补。

输入宏窗原62D先用原_time_prior增加为63D，再追加24D，固定87D；原63D严格逐值保留。prefix sums在每sid内按start排序、end单调且无重复，范围包含两端等号，O(rows log rows+47×rows)；浮点方差负舍入截0。无需持久化增强特征，在内存构建后按原角色slice。score subject自己的固定特征聚合是推理输入，不是训练或adaptation，不参与任何fit统计。

## 学习与验证不变项

- HGB原150trees/15leaf/depth4/minleaf100及seed不变；micro原LGBM47D及标签/训练sample/threshold完全同A；meta116D、原LR/LGBM、C=.1、.25/.75融合、policy网格不变。唯一改动宏窗输入63→87D，宏窗概率及其候选可能改变，不声称冻结分数/候选。
- macro/micro canonical每sid负窗≤max(1,3npos)、seed20260901采样一次，原标签。所有拟合/选择仅train；四折window subject OOF。加入可选macro input adapter到crossfit helper，None精确沿用原_with_time_prior。新adapter只接受87D并恒等返回，防止重复增加时间先验。
- macro_lgbm_event.py新增hypothesis='gyro_pool'，纯汇总函数gyro_pool_macro.py；先全候选数据确定性增强，再由现有prepare_train/train/score切分和OOF拟合，不增加标签依赖。实际sample身份/micro threshold逐fit与A精确核对。
- T24原三组16train/8health，9个policy fit全部seal/commit再统一health；冻结A89/62/71、完整160GT。gate固定F1差≥.01、≥2/3严格提高、TP不减，FP增加时deltaTP>deltaFP。通过才39主体254GT五折，15个policy fit先seal/commit，A138/80/116，同门但3/5严格提高。父级须匹配新hypothesis通过。反复开发证据，不称新独立外层或官方分数。
- 失败关闭此24D宏窗表示，不在相同health选择pool列、分位数、窗口半径、模型或阈值。两门通过才另登记全目标refit及新增87D runtime schema/导出版本，至少真实session的研究/推理汇总逐值等价、最新冻结全套和EXE-only接口验收后交付。研究阶段不改变canonical推理或正式bundle身份。

## 预算与实施ledger

输出outputs/official_recovery/gyro_pool_macro_v1_t24及_five；T24 120窗口/24meta、五折200窗口/40meta。CPU8线程；单cohort≤2h/RSS12GiB/新证据128MiB/空闲≥10GiB。只保存原分数/策略/图/覆盖分布/源及资格指纹，不保存失败增强缓存/模型。原66保护文件SHA核对，拒绝覆盖，异常INCOMPLETE；全部代码/spec在policy到评分期间冻结。

TDD步骤：手算两列mean/std、sid隔离/完全包含/边界、缺失微窗NaN、行顺序独立、重复/非法width拒绝；原62+时间63逐值保留、标签不变、修改score sid不影响train sid特征；crossfit默认恒等以及新adapter真实fit/score路径。相关测试后Luna medium一次只读review，严重项一次RED→GREEN，小项延期；提交再训练。单个本文兼设计/实施ledger，避免散落临时计划。

Ruling：从监督/分类器变更转向固定增量表示——此前失败不足以证明IMU信息无效，只说明既有表示上这些合同未改善；代价是若87D成功还需额外runtime/schema验收才能成为EXE，不能用研究脚本直接替代生产。
