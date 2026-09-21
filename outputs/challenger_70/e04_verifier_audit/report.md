# E04 Verifier Failure Audit

> **DIAGNOSTIC ONLY / USES GT.** This sealed-outer evidence explains frozen incumbent scores; it is not a challenger result.

## Raw-reachable false-negative population

The E01-bound target population contains 31 truths: 27 C1 and 4 C2 (fold counts 3/1/10/6/11).

## Mechanism counts

best-IoU witness labels are non-exclusive: A=3, B=0, C=15, D=0, B_model_disagreement=1, unsupported=0, unresolved=14. best-IoU fold support: fold 0: 3, fold 1: 1, fold 2: 10, fold 3: 6, fold 4: 11.
best-score witness labels are non-exclusive: A=3, B=0, C=15, D=0, B_model_disagreement=1, unsupported=0, unresolved=14. best-score fold support: fold 0: 3, fold 1: 1, fold 2: 10, fold 3: 6, fold 4: 11.

## LR/LGBM and Context findings

The fixed |LR−LGBM| ≥ 0.20 descriptive flag appears on 15 best-score witnesses. Pooled best-score Context median contributions are LR=0.9572889121582079 and LGBM=0.812908802177234 in each fitted model's raw-margin space.
Stable descriptive feature directions among best-score witnesses: base_02, base_03, base_05, base_06, base_07, base_08, base_09, base_10, base_11, base_12, base_14, base_15, base_16, base_20, base_21, base_28, base_29, base_31, base_32, base_41, base_42, base_43, base_44, base_45, base_46, base_47, base_48, base_49, base_50, base_52, base_53, macro_candidate_above_fraction, macro_candidate_center_half_mass_fraction, macro_candidate_max, macro_candidate_mean, macro_candidate_minus_post_above_fraction, macro_candidate_minus_post_mean, macro_candidate_minus_post_std, macro_neighbor_run_max_duration_s, macro_post_max, macro_post_mean, macro_post_std, macro_pre_coverage, macro_pre_mean, macro_pre_std, micro_candidate_above_fraction, micro_candidate_max, micro_candidate_mean, micro_candidate_minus_post_above_fraction, micro_candidate_minus_post_mean, micro_candidate_minus_post_std, micro_candidate_minus_pre_mean, micro_candidate_std, micro_pre_max, micro_pre_std.

## Limits

Witness rows are GT-aware. Descriptive score margins, contributions, and feature directions neither tune a threshold nor select a policy or model.
