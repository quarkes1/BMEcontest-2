# HALFSTEP-V6：固定半步召回送测对照

状态：2026-10-28按用户最终收口指令取消。未实现probe、未运行模型/评价、没有新F1或EXE；未完成的测试已删除。后文仅为历史方案，不再执行。

2026-10-26，用户要求抓紧产出EXE，预计28日最晚29日最终提交。CAP8-V4与FUSION-POLICY-V5已失败，分别TP+5/FP+22/F1−.01897和TP−9/FP−12/F1−.02061；不导出这两个失败算法，不降低它们原门。

本轮是**独立试测风险对照，不是正式promotion**。官方最佳.533333（56/42/56）的原EXE永久保留。只走一个事前固定方向：主档F1阈值与已有F1.5高召回档阈值取logit空间中点，即odds几何平均，权重固定.5，不搜索插值位置。四模型、blend.25、cap6、admission.2、NMS.3全不变，不训练、不重算组件分数、不根据官方CSV时间/ID设计规则。

各fold用各自先前train-only F1/F1.5阈值取同一公式中点；最终EXE取两个已有部署阈值.4582664763345077与.26383368823631487的同一中点。它不是对原full-data模型的无偏CV，actual_exe_cv=False，只报重复冻结折模型的政策开发诊断。两个端点各有已封存来源，不能用outer回扫最终阈值。

先T24三组160餐，再39人五折254餐。试测风险门：净TP至少1/3，新增FP≤2.5×新增TP，F1不得比主档下降超过.005；至少2/3或3/5折TP不减。它和前两项的算法改善门不同，必须如实披露是否本地F1实际提高。失败不换插值权重、不再加试验，不能称改善。

通过后模型/feature_schema字节保持与主版相同，独立版本official_halfstep_policy_v6，dist/official_trials/halfstep_policy_v6/EatingDetection.exe；最终threshold为固定中点，不修改原模型/EXE/正式submission/frontend。只发送EXE，CPU单文件，官方无参数ACC_test_data→results.csv接口不变。

75保护文件、两端点bundle与原始score/图/分区/阈值seals前后SHA核对。用户删除主README、新增submission.zip保持原样不stage。新source与tests稳定后完整pytest与独立编译/真实中文EXE-only验收，接受缓存清理的执行策略限制、不绕过删除拒绝。打包叶函数复用现有builder，不修改canonical算法。零模型fit、每probe10分钟/RSS12GiB/16MiB证据、每编译30分钟、每验收子进程300秒、剩余10GiB。

最终材料选择仍以已收到的官方成绩为准：若新试测反馈未及时返回，不凭本地小波动自动替换.533333最佳版。这是用户要求的下一次送测文件，不能保证官方更高。

## Ledger

- Ruling：拒绝两项明显失败变体，改成既有两端点之间的固定半步，不再扩大参数面——控制最后一次试测的改动幅度；代价是增益可能很小，官方仍未知。
- 最终交付review涵盖probe/两端点身份/打包/验收；未通过不得伪报完成。
