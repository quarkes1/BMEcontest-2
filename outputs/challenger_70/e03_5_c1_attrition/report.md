# E03.5 C1 Attrition Audit

> **DIAGNOSTIC ONLY / USES GT for oracle rows.** This sealed-outer audit is not a challenger result and has no performance headline.

## Cause counts

| First total-loss stage | C1 short truths | terminal-reason sets |
| --- | ---: | --- |
| admission_threshold | 5 | — |
| subject_cap | 6 | — |

## Frozen-score single-operation counterfactuals

| Intervention | C1 newly reachable | C1 newly matched | C1 downstream still lost | Non-C1 newly matched/lost | Extra nonmatching candidates | Gross introduced final FP | Removed incumbent FP | Net final FP |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| disable_nms_only | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 | 0 |
| disable_admission_threshold_only | 1 | 0 | 1 | 0 / 0 | 32 | 0 | 0 | 0 |
| disable_subject_cap_only | 6 | 1 | 5 | 0 / 0 | 158 | 21 | 0 | 21 |

No parameter or challenger recommendation is made here. E04/E05 remain paused pending human review.
