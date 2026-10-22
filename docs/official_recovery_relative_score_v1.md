# RELATIVE-SCORE-V1 — 冻结模型的主体相对决策

2026-10-22登记。用户授权限时自主推进；不修改正式release、第一试测或dist/submission。本项无模型训练，不重算LR/LGBM、窗口分数或候选。

## 固定假设及公式

全局event threshold可能牺牲整体分数偏低的主体；这只是待验证假设，不把官方63FN归因于此。与E02 percentile输入新verifier、旧MAX-FUSION或day-budget不同，本项仅改变准入后的最终event score。

对当前一个producer的每个subject全部raw union candidates，p为冻结.25LR+.75LGBM分数。固定eps=1e-6，按clip(p,eps,1-eps)求logit。anchor为该subject raw p的90% quantile(method='linear')后同样clip/logit；新分数r=sigmoid(logit(p)−.5logit(anchor))。只用该subject无标签预测；不读GT、subject identity特征、不跨主体估计anchor。无候选时空输出；单候选照公式，不强制报餐。固定.5和.90，禁止在health搜索。

保留部分绝对证据而非纯rank，减少把所有无餐主体强行抬至同一高分的风险；是否仍产生这种FP成本必须通过事件评价检验。r只称决策分数，不称校准概率。

## 严格冻结与选择

- 原A scores严格复用negative_clean_v1/negative_clean_fivefold_v1完整候选来源、geometry、role graph、原LR/LGBM融合恒等及来源指纹。每个producer的anchor只由其score subject自己的raw scores形成，无模型fit。训练policy来自原三OOF producer预测，每subject覆盖一次。
- NMS/admission threshold/subject cap沿用原A各组已冻结policy，且**仍使用原p**执行。本项绝不把r交给admit_candidates。只在已准入集合以r选最终事件；没有新最终event cap。原A逐事件回放须精确89/62/71或五折138/80/116。
- 新final threshold只在train OOF上按完整GT/strict IoU>.25/每subject完整GT一次计数，固定排序(F1,PPV,较高threshold)选择；候选阈值为已准入r的唯一值+0/1/nextafter(1,2)，与既有规则一致。三policy全部seal/commit后才health。
- 图应显式连接各producer score节点→label-free relative_score→最终policy；不能只存一个无实际输入父节点的policy标记。score角色/禁入主体严格核对。
- T24完整160餐门F1差≥.01、≥2/3严格提高、TP不减，若FP增加则deltaTP>deltaFP。通过才39主体/254餐五折，同门但≥3/5改善；五折父级必须是本ID通过的T24并复算gate。五组策略也先seal/commit再outer。重复开发证据，非新独立/官方F1。

## 交付、预算与步骤

两门通过后另注册独立full-target policy/refit及runtime可选relative-score模式，原生产默认逐位不变。官方runner每subject已统一trace_texts处理多文件，能与本研究的subject anchor保持同一输入范围。必须确认全目标refit合同和EXE-only验收再导出；不把本研究分数直接写进旧bundle。失败关闭固定公式，不改quantile/alpha、不导出。

新代码relative_score_probe.py，纯公式/准入后选择也在同模块。输出outputs/official_recovery/relative_score_v1_t24及_five。仅小型策略/报告/来源图，无模型/特征缓存。每phase≤10min、RSS12GiB、新证据16MiB、空闲10GiB；输入代码/spec/cache/Data完整SHA、资格来源与66保护跨phase核查，禁止覆盖，异常INCOMPLETE。

TDD：同主体手算logit公式、permutation/跨主体独立、0/1/空/非法分数；原p准入而非r、完全相同原admission集合；train-only threshold、严格IoU边界/GT一次计数；新gate/父级身份、防止未seal或dirty source评分。专项回归后Luna medium一次只读审查，严重项一次RED→GREEN，小项延期。先提交代码/任务书，policy封存提交，再统一评分；源冻结，结束才更新文档。

Ruling：暂停新模型拟合，优先固定主体决策的低成本机制检验——当前模型/监督/增量表示探索均未可靠改善；代价是无标签anchor仍可能放大无餐主体噪声，因此必须计FP且原准入保持冻结，失败不投送。
