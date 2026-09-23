# E05.1 Decision Churn Audit

Final status: `GYRO_RESIDUAL_NOT_SUPPORTED`

Exact replay counts: `{'TP_RECOVERED': 11, 'FP_REMOVED': 16, 'FP_INTRODUCED': 20, 'TP_LOST': 9}`.

1. TP recovered is concentrated below the incumbent boundary: median margin `-0.058679`.
2. TP lost is above that boundary, but only narrowly: median margin `0.009570`; it is not a clearly high-confidence-loss population. H1 therefore has boundary ordering but only partial support for its stated high-confidence-loss mechanism.
3. FP removed and FP introduced are not usefully separated by GYRO5. Median variance is `93468.072` versus `102116.484`, with very wide overlapping interquartile ranges.
4. The monotonic GYRO interaction is contradicted: TP-recovered median variance (`33879.103`) is lower, not higher, than FP-introduced (`102116.484`).
Fold direction consistency: `3/5`.

5. Three folds have the desired recovered-TP versus introduced-FP direction, one has the opposite direction, and one has no recovered TP; the effect is not robust enough when combined with the aggregate contradiction.
6. Frozen residual is **not supported**.
7. Uncertainty gating alone is **not supported**: margin ordering exists, but the GYRO correction direction is neither monotonic nor cleanly separable. No threshold was searched.
No new model, threshold, policy, outer F1, release, or dist artifact was created.
