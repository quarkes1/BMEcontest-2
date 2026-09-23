# Challenger data characterization sprint

> Diagnostic only: no training, threshold tuning, policy change, outer F1, release, README headline, or dist change.

**End state: `MULTIPLE_SIGNALS_FOUND_NEEDS_HUMAN_SELECTION`** — frozen LR/LGBM direction; fold-stable short-scale proxy.

The official headline remains `0.6514285714 / 114 / 153 / 197`.

## Required decisions

1. Adaptive fusion: **retain as a Tier 1 candidate**.
2. New temporal-scale signal: **yes**; strongest proxy `gyro_variance` at 5 s.
3. Repetition/periodicity signal: **not established**; strongest proxy `burst_count`.
4. Temporal morphology: PCA remains descriptive and does not independently justify a sequence model.
5. Subject-relative shift: **not established**.
6. Fold 0/2 shared mechanism: descriptive outliers are `{"fp_density_per_session": 1.3646041902507107}`; no fold-specific intervention is permitted.
7. Dominant FP mode: `[["long_continuous_movement", 39], ["low_frequency_long_movement", 27], ["fragmented_high_activity", 11]]`.
8. Next challenger mechanism: **MULTIPLE_SIGNALS_FOUND_NEEDS_HUMAN_SELECTION**; human review is required before any implementation.
9. Stop directions: any signal that failed its predeclared fold-stability/robust-separation gate; no further outer-derived mining.
10. Representation re-extraction: **likely required for temporal candidate**.

## Population contract

Matched TP: 114 final events; final FP: 83 events; reachable FN: 31 unique GT (highest frozen-score witness from 32 pairs); C0 FN: 6 GT. All folds and scores are frozen incumbent evidence.
