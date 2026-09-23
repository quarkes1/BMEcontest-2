# E04.1 — Frozen LR/LGBM disagreement direction

Diagnostic only. Scores, threshold, models, and populations are frozen; no F1 was computed.

| Population | N | LR−LGBM ≥ .20 | LGBM−LR ≥ .20 | Neither |
|---|---:|---:|---:|---:|
| matched_tp | 114 | 1 | 5 | 108 |
| final_fp | 83 | 9 | 17 | 57 |
| reachable_fn | 31 | 12 | 3 | 16 |

Reachable-FN LR-high share exceeds FP in 4/5 folds.
Evidence flag for later adaptive-fusion consideration: **yes**. This is not a challenger result.

