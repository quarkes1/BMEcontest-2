# A6 frozen V7 validation

TCN primary; Transformer secondary. Same historical competition cohort, not pristine independent evidence. No outer/dist/release change.

| Arm | F1 | TP | FP | FN |
| --- | --- | --- | --- | --- |
| b | 0.745763 | 22 | 4 | 11 |
| c | 0.689655 | 20 | 5 | 13 |
| t | 0.724138 | 21 | 4 | 12 |

Primary TCN delta: -0.056108; gate passed: False.
Secondary success cannot override a failed primary gate. A pass only supports requesting separately approved outer evaluation.

Per-subject, short-event and candidate counts are in summary.json; final models/policies were committed before V7 inference.
