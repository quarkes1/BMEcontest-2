# A9 — 主体×事件均衡复核损失

2026-10-14；用户自主授权，主代理直接执行，不请求重复审批。本文件合并设计、实施计划与执行账本。目标：正式事件F1≥.70；本轮仅T24开发探针，不直接更新dist。

## 假设与单一改动

A7低维PPG残差未增益，A8固定源域proposal只救1个可达GT；旧候选已可达93/97GT，最终检出66。选择检验candidate分类损失与subject/event评价的权重错配，暂不增加表示或定位模型。此选择来自汇总机制，不按具体失误GT制作规则。

B复现原T1 116D LR(C=.1)+LGBM概率均值与原候选，分数核对后直接复制旧证据。W使用完全相同producer、候选、116D、imputer/scaler、LR/LGBM参数、seed与.5 blend；仅向两分类器fit传入同一权重，预处理仍按原训练数据拟合。

正候选按当前canonical规则best IoU≥.25归属一个训练GT（等IoU按truth原顺序）；每subject内，每个有正候选的GT获相同总正权重，每个GT的重复候选平分。负候选在每subject内平分总负权重；没有正/负样本的主体不虚构该类别。最后分别把正类权重归一化至N_pos、负类至N_neg；全局mean=1。保留既有estimator class_weight，不额外做类别平衡，避免重复缩放改变C的有效尺度。只使用train truths，不用health truth/hand metadata/外部raw。

权重确定性、无可选λ/clip/超参数。按原12partition严格重建producer OOF ledger与verifier；labels必须和权重归属一致，train/score/excluded及全祖先隔离校验。每partition重算B复现旧候选/标签/groups/base分数（atol1e−6/rtol1e−5）。新W政策沿用原choose_policy/grid，仅在九个policy分区OOF选取；三份政策封存提交后才能health，不看结果后冻结最优旧阈值。

## Gate与预算

主gate：W−B≥.01 aggregate F1，≥2/3 wrapper非下降、TP≥B−1，且FP减少或新增TP>新增FP。固定gate失败即关闭本权重配置，不在health上改归一化、C、阈值或做截断搜索。

仅T24原manifest，V7/outer不参与。预计120 producer拟合、24 B verifier拟合、24 W verifier拟合；三个wrapper CPU并行、数值库4线程。无GPU训练、raw读取或新增缓存。policy+health墙钟≤1h、RSS≤16GiB、新证据≤64MiB、磁盘剩余≥20GiB。

报告同口径B/W F1、TP/FP/FN、逐wrapper与逐人、短餐、GT迁移、训练权重范围/有效样本量、资源与保护快照。通过只允许另行注册验证，不把T24 F1称正式提升。

## 实施计划（executing-plans / TDD）

代码`scripts/research/representation_v2/event_balance/{model,probe}.py`；测试`tests/research/test_event_balance.py`；输出`outputs/challenger_70/representation_v2/amendment_a9_event_balance/`。

- [ ] Task1 RED→GREEN：权重正/负各自总和、每主体/事件的总权重、重复候选、无正类别主体、错label/归属/未知sid拒绝；实现`sample_weights(candidates, labels, groups, truths, owners) -> ndarray`和`probabilities(train, validation, weights, seed)`，真实pipeline接sample_weight。
- [ ] Task2：register锁定A9代码/本文、原P1 registeredfiles和T1证据/manifest、protected tree；提交后policy阶段严格复现B、拟合W、选择W政策，资源与政策共同封存提交；health前拒绝未提交/篡改seal。
- [ ] Task3：health同结构评分与GT迁移，核对B=66/20/31；评gate并写报告，不改dist。专项回归、核对上游hash/保护树，更新README、清理本阶段临时文件、提交结果。

Review Focus：best-IoU归属与候选label完全一致；少餐subject的大权重可能增大方差，需要披露有效样本量；pipeline参数必须model__sample_weight且不改class_weight；policy与health边界；权重不允许接触验证GT。

Ruling：用户最新授权取代重复计划审批；保持发布的证据标准，不因时间紧迫把开发成绩包装为正式F1。
