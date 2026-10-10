# S6: teammate-inspired geometry probe and release boundary compatibility

2026-10-10. **Exploratory diagnostic only.** Code `934a84b`. No training, threshold tuning, health access, new outer predictions/F1, dist or release change. Preliminary counts were observed before the written protocol; this is not presented as a preregistered success.

## Fixed geometry probe

`union_candidates` uses the macro interval when it pairs macro and micro candidates. This probe first applies the original B score/admission/event threshold, then intersects selected dual-source intervals with micro support. All scores, selected raw indices, and single-source intervals remain unchanged. Use raw indices rather than looking up candidates by equal interval geometry.

| Policy wrapper | Changed intervals | Baseline TP/FP/FN | Clipped TP/FP/FN | GT recovered/lost |
| --- | ---: | --- | --- | --- |
| 0 | 5 | 48/30/17 | 48/30/17 | 0/0 |
| 1 | 10 | 38/7/28 | 38/7/28 | 0/0 |
| 2 | 2 | 50/10/13 | 50/10/13 | 0/0 |

These wrappers reuse subjects and GT; do not pool occurrences as independent observations. No candidate AP or event F1 headline is produced.

Matched GT identities are unchanged, not merely the TP counts. Therefore own-match and common-GT MAE comparisons use the same truths. Start/end MAE (seconds):

| Wrapper | Baseline start/end | Clipped start/end |
| --- | --- | --- |
| 0 | 292.73 / 154.22 | 297.89 / 155.47 |
| 1 | 306.04 / 114.47 | 271.30 / 139.54 |
| 2 | 384.51 / 142.89 | 383.91 / 142.59 |

FP total duration decreases by 82.5s/877.5s/0s respectively, but **FP event counts do not decrease**. This is not an improvement in the competition's primary metric. No consistent secondary boundary benefit supports continuing this exact decoder. Close it without consuming health or trying another clipping parameter on the same data. This does not rule out event splitting/merging or learned boundary models; those need separate training-only evidence and a distinct registered hypothesis.

## Exact IoU boundary

Resources/试题.txt specifies `IoU > 0.25`. Current README/production evaluator historically used `>=`. All raw candidate/GT and final prediction/GT pairs in the three policy wrappers contain zero exact 1/4 boundary pairs; integer strict/inclusive matching produces identical final counts there.

Separately, the sealed E04 TP export contains exactly 114 unique prediction matches with fold counts 16/30/15/25/28, matching the release. Its SHA256 matches the previously recorded `e963fd43964c948f05f11c199923469200d4ab1eda433b473fdb5a405809cc41`. Recompute IoU from integer prediction and GT endpoints: zero exact-quarter matches; minimum IoU 0.2604783149166265. Since every matched edge is strictly above the boundary and lower-weight exact-quarter edges occur later in greedy matching, strict comparison preserves these **already-frozen predictions'** TP count. This is compatibility evidence for the existing release, not a new performance claim.

This does not establish that correcting every training-label or threshold-selection use of the comparison would select the same model/policy. Production evaluator correction should be versioned and audited separately; no production algorithm change was made here.

## Teammate comparison and score transport limits

Reviewed public `Lancercodermaker/Bme_Eating_Detection` main `dd05aa128f8d62a85c20a83f8af72fe13fcfa27b`. Its README reports 234 eligible GT and later A2 best pooled F1 .623482. The corresponding later experiment branch/artifacts are explicitly unpublished; only main is publicly listed. Our 153-GT result cannot be ranked against that denominator. Its 15s/7.5s candidate/HGB verifier architecture largely overlaps components we already have.

Candidate-model-relative quantile thresholds remain an interesting design hypothesis, but the public prototype derives one full-training threshold and applies it to OOF as well; the newer design is not equivalent to a validated calibration implementation. Our saved S2 baseline ledgers lack original verifier training-score reference distributions. Held-out subject probability distributions alone do not diagnose OOF-to-refit calibration shift. Do not build a calibration experiment on an unsupported causal claim or use health labels to fit it.

Practical next input is the teammate's actual later branch plus split/eligibility manifests and per-event artifacts. That allows a common event-population comparison and tests whether it contributes complementary meals rather than duplicating our errors. No model download or remote execution was performed.

## Verification and resources

All twelve geometry inputs match the pinned S2 commit byte-for-byte; actual producer/policy ancestry validated; baseline selection counts replayed. Inputs unchanged. luna/medium read-only review found no blocker in the geometry path. 52 focused geometry/stability/event tests passed, including strict-boundary handling, session isolation, GT identity swaps and non-mutating clipping. Runtime 0.61s after imports, approximately 16KiB of evidence before summary/report; zero new caches, models, GPU use or copied checkpoint. Repository-wide full-suite testing was not rerun for this isolated research addition.
