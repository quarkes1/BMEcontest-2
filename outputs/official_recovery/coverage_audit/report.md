# 记录长度与缓存窗口支持审计

不训练、不改变评分或策略、不产生挑战模型成绩。窗口覆盖是已缓存窗口区间的并集，不是原始信号有效率；索引覆盖也不等于有效IMU。

| 观察 | 数量 |
|---|---:|
| truths | 254 |
| legacy_excluded_truths | 101 |
| excluded_no_index_overlap | 4 |
| excluded_partial_index_union | 2 |
| excluded_with_80pct_macro_window_support | 70 |
| excluded_with_80pct_micro_window_support | 72 |
| excluded_no_cached_micro_support | 19 |
| truths_with_discarded_micro_support | 1 |
| micro_windows_without_macro_session | 59 |
| subjects_with_more_than6_valid_labels | 18 |
| theoretical_unavoidable_fn_with_six_cap | 30 |

每人6候选只作部署配置的理论上限检查；旧五折策略并非全部cap6，不能将此上限当作实际漏检归因。
微窗支持被宏窗会话过滤丢弃，不等于恢复该支持一定形成正确事件；须在训练内重新验证候选、评分和FP成本。
完整有效标签保留未覆盖餐；未知原始采样情况不作推断。所有输入和保护哈希见summary.json。
