# A7 P2 冻结先验残差结果

T24开发证据，政策固定；outer_calls=0。

|arm|F1|TP|FP|FN|
|---|---:|---:|---:|---:|
|b|0.721311|66|20|31|
|q|0.709302|61|14|36|
|p|0.712644|62|15|35|

## GT迁移与可用性

{'b': {'gained': 0, 'lost': 0, 'matched': 66}, 'q': {'gained': 0, 'lost': 5, 'matched': 61}, 'p': {'gained': 0, 'lost': 4, 'matched': 62}}

{'eligible': 340, 'total': 794}

Wrapper 0: {'b': {'matched_indices': [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30], 'gained_indices': [], 'lost_indices': []}, 'q': {'matched_indices': [0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 30], 'gained_indices': [], 'lost_indices': [10, 26]}, 'p': {'matched_indices': [0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 30], 'gained_indices': [], 'lost_indices': [10, 13, 26]}}

Wrapper 1: {'b': {'matched_indices': [1, 2, 4, 8, 9, 13, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30], 'gained_indices': [], 'lost_indices': []}, 'q': {'matched_indices': [1, 2, 4, 8, 9, 13, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 27, 28, 29, 30], 'gained_indices': [], 'lost_indices': [26]}, 'p': {'matched_indices': [1, 2, 4, 8, 9, 13, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30], 'gained_indices': [], 'lost_indices': []}}

Wrapper 2: {'b': {'matched_indices': [0, 1, 7, 8, 10, 11, 13, 15, 16, 17, 18, 20, 24, 30, 31, 33], 'gained_indices': [], 'lost_indices': []}, 'q': {'matched_indices': [1, 7, 8, 10, 11, 13, 15, 16, 18, 20, 24, 30, 31, 33], 'gained_indices': [], 'lost_indices': [0, 17]}, 'p': {'matched_indices': [1, 7, 8, 10, 11, 13, 15, 16, 17, 18, 20, 24, 30, 31, 33], 'gained_indices': [], 'lost_indices': [0]}}

## 逐人诊断

{'b': {'median': 0.6666666666666666, 'iqr': [0.5535714285714286, 0.8571428571428571], 'p10': 0.35333333333333333, 'improved': 0, 'worsened': 0, 'tied': 24}, 'q': {'median': 0.6666666666666666, 'iqr': [0.43333333333333335, 0.8650793650793651], 'p10': 0.10000000000000009, 'improved': 4, 'worsened': 4, 'tied': 16}, 'p': {'median': 0.6666666666666666, 'iqr': [0.4861111111111111, 0.8392857142857143], 'p10': 0.10000000000000009, 'improved': 3, 'worsened': 4, 'tied': 17}}

Gate: {'passed': False, 'delta_vs_base': -0.008667797248916531, 'delta_vs_quality': 0.0033413525795241705, 'nonworse_wrappers': 1, 'outer_calls': 0, 'next_validation_authorized': False}

正式F1仍.6514285714，未变更dist。
