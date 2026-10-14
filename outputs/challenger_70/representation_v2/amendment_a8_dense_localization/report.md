# A8 D0 source-only定位诊断

仅现有227会话缓存；不训练、不调整规则、不评价最终F1。

|wrapper|GT|缓存覆盖GT|原可达|新可达|联合可达|新增GT|新候选|无GT重叠候选|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|0|32|32|32|25|32|0|433|408|
|1|31|31|30|27|30|0|448|420|
|2|34|34|31|28|32|1|436|407|

汇总：{'n_true': 97, 'cached_gt': 97, 'old_candidate_count': 794, 'new_candidate_count': 1317, 'old_reachable': 93, 'new_reachable': 80, 'union_reachable': 94, 'candidate_without_gt_overlap': 1235, 'one_to_one_matched': 80, 'unmatched_candidates': 1237, 'additional_gt': 1}

新候选FP成本不是最终事件FP；source概率未经目标校准。既有缓存会话由旧候选筛选，存在覆盖选择偏差，缺缓存不当作阴性。正式F1仍.6514285714，dist未改变。
