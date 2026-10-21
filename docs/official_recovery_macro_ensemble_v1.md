# MACRO-SOFT-ENSEMBLE-V1 — 宏窗固定等权集成

## 依据与边界

独立宏窗LGBM替换的窗口AP+.01888，但M1事件总体−.00557：前两组提高，第三组下降较大。不能选择丢弃第三组，也不能把单模型送测。当前检验保留HGB、以固定等权概率平均减少模型替换方差。不是verifier max、不是旧三种子Transformer集成；只有一个预先确定的组合，不根据health调权重或挑成员。

## 固定合同

- 宏窗factory是sklearn VotingClassifier：原HGB(seed)与MACRO-LGBM-V1 factory(seed)，voting=soft、等权、n_jobs=1。每个macro OOF及最终refit均独立fit这两个成员，在同一63D和同一训练抽样上输出概率均值。先均值再生成候选，不能跨wrapper平均或以另一模型全train结果替代OOF。
- micro/116D Context-v1/旧eligible监督、LR C=.1、.25/.75融合、候选与完整GT policy算法/原网格不变。采样及micro阈值每fit精确等于冻结A；macro工厂是唯一模型变化。A仍为89/62/71（T24）与138/80/116（五折），严格配对完整160/254餐。
- 不先重复AP筛选：M1已证明替代指标可能与事件结果反向。本轮直接用严格嵌套事件目标。三个train×三policy×独立四windowOOF，9份fit/三份policy全部seal+commit后统一health；通过才五折15份fit/五policy全部seal+commit再outer。参数只用各自train OOF与train完整GT选择，V7/outer不参与T24。
- 两道工程门均固定：aggregate F1至少+.01；T24至少2/3、五折至少3/5严格提高；TP不减少；FP增加时新增TP须大于新增FP。此门在任何ensemble事件评分前登记；原窗口、M1及其他已失败实验不重判。降低本轮预算门至1个百分点是为了允许实用而非宣称突破的改善；多数折与TP/FP约束保留。通过只说明重复开发中有送检价值，官方改善仍待反馈。
- 父级窗口来源仍固定plan SHA/code SHA及只允许文档/crossfit注入两处已声明源差异；另外哈希完整M1失败结果，禁止事后改写。parent T24→five必须核对本实验门与其source/protected，不能误用失败M1门。代码/数据/缓存/资格与66保护前后核查；任何漂移INCOMPLETE。
- 训练仅CPU，数值库8线程、成员n_jobs1；预计T24 180基础窗口fit/24meta，五折300窗口/40meta。每cohort≤2h、RSS≤12GiB、证据≤128MiB、余量≥10GiB。保存分数与图，不保存探针模型/新特征缓存。

## 实施与门后交付

复用macro_lgbm_event.py，增加显式hypothesis=lgbm|ensemble，不更改旧默认与旧M1门。ensemble独立输出macro_soft_ensemble_v1_t24/_five；门与parent注入须按hypothesis隔离。TDD核对两个成员概率精确均值、默认旧工厂/旧gate不变、新gate实际幅度/多数折/TPFP条件、父级失败拒绝。Luna medium一次只读审查后运行。

两门通过才另登记独立全目标ensemble refit与EXE版本；第一版/正式release/dist/submission不覆盖，前端不自动切换。EXE-only接口、真实数据Python/EXE字节一致、空错输入、自检、中文路径无开发环境与最新回归后才交付。不能保证隐藏测试一定改善，也不把T24或重复五折称官方F1。

Pre-flight：VotingClassifier成员各自封装imputer，不共享fit状态；clone后的成员seed与train主体不变。宏窗predict_proba接口和classes_兼容canonical Predictor，因此若通过无需改预测数学，仅需模型、策略与构建身份新版本。

实施：4项测试RED→GREEN（含真实Voting fit），随后一次Luna medium审查指出M1失败结果仅被当前哈希不能证明父级身份；新增第五项失败测试后修复，固定M1 plan SHA `fe1eb7de1de58c99cf6c610f8ef62c970944666946e575af7bfcb66e1571f8e2`、COMPLETE/failed/87-60-73、保护身份与原source。仅允许原M1文档新增结果和本入口扩展ensemble两个已声明的源差异；源指纹仍由本轮plan冻结。父级五折入口额外明确验证hypothesis/ID/成功门，避免误用M1。无其他严重问题或Minor。
