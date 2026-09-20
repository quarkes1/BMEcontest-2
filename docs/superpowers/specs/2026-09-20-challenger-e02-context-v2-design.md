# Challenger E02 — Context-v2 Design

## Scope and evidence

E02 is one preregistered verifier-feature challenger on branch "research/f1-070-challenger". It appends a small deterministic session-local Context-v2 block to the frozen macro-v1, micro-v1, candidate-control and 116D Context-v1 verifier pipeline. It does not change raw features, window models, candidate generation, matcher, policy grids, incumbent release, dist, or promoted artifacts.

E00 reproduces run 160afaf81debf1ee: TP / eligible / pred 114 / 153 / 197, FP 83, FN 39, F1 0.6514285714285715. E01 finds 27/39 FN are C1 raw-candidate reachable; 61/83 FP have macro+micro support and 38/83 are dense-cluster FPs. This supports a narrow, FP-oriented session-relative experiment. E01 Context-v1 FP-versus-TP separation tables are explicitly excluded from feature selection.

The predeclared list below is fixed before any E02 inner or outer result. Any redesign is a new experiment ID.

## Hypothesis and data boundary

A raw-union candidate that is weak relative to unlabelled probability evidence in its own session, or has nearby stronger same-session candidates, is more likely to be a false positive. Context-v2 lets the existing LogisticRegression plus restricted LightGBM verifier express this distinction. The mechanism claim requires FP removal and TP-retention accounting.

The producer sees only raw candidate geometry and unlabelled macro/micro probability windows from the same exact session. It must not read GT, candidate labels, verifier/admission/final scores, thresholds, NMS, caps, another session, or another subject. Competition is calculated across the complete raw-union candidate set before any verifier score; therefore no pre-score feedback loop exists.

## Fixed Context-v2 schema

"context_features_version='v2'" is a strict Context-v1 increment: it appends exactly 25 columns after the existing immutable 56 legacy plus 60 Context-v1 columns, giving E02 verifier width 141. No-context remains 56 and v1 remains 116. Version v1 must retain bit-for-bit behavior.

For candidate c and source s in {macro,micro}, mean_s(c) and max_s(c) are the arithmetic mean/maximum of finite source CandidateEvent probabilities. An absent source or no finite probability is missing. coverage(c) is the mean finite observed_fraction across present sources. duration(c) is end_ms - start_ms; support_count(c) is the number of present sources.

The complete raw-union population is exactly the output of the existing `union_candidates(macro, micro, merge_iou=0.25)` semantic authority, after validation.  Every member must have positive event duration, one exact session sid, at least one non-null source and at least one source with a nonempty finite probability stream.  Invalid members are a fail-closed error, never silently filtered.  No second deduplication is allowed: macro candidates are canonically ordered by the existing CandidateEvent canonical key; micro candidates are ordered by `(sid, -max(probabilities), canonical_key)`; each micro candidate greedily consumes at most one still-unmatched same-sid macro with IoU >= .25, choosing maximum `(IoU, -macro_start, -macro_end, macro_index)`; output is then sorted `(sid,start,end)`.  A Context-v2 candidate identity is its output event plus the complete canonical source identity of macro and micro, so coincident event intervals with different sources remain distinct just as `union_candidates` keeps them.

All candidate comparison populations contain only such complete raw-union candidates with the same sid as c and the required finite value. A sid is a complete acquisition session, not merely a subject identifier. Window baselines use finite scores only from the same sid/source stream. Authoritative session_bounds_by_sid is mandatory and ordered. The source index stores whole-second endpoints while cached macro/micro windows retain millisecond, stride-aligned timestamps; therefore candidate containment permits one fixed 15,000 ms acquisition-grid quantization margin at either nominal index endpoint. This is a data-format reconciliation, not a feature hyperparameter: it cannot join sids, bridge an acquisition gap, or expand any context region/rate denominator beyond the authoritative bounds. A candidate farther outside this fixed margin fails closed.

For finite x and nonempty finite V, E02 requires NumPy 2.4.6 and calls `np.percentile(V, 90, method="linear")` explicitly:
- pct(x,V) = (count(V < x) + 0.5 * count(V == x)) / len(V). This midrank makes a singleton or all tie group 0.5.
- rz(x,V) = (x - median(V)) / (1.4826 * median(abs(V - median(V))) + 0.05).
- relative(x,b) = (x - b) / max(abs(b), 0.05).
- ratio_to_max(x,V) = x / max(max(V), 0.05).

The fixed 0.05 floor is schema, not a tuned parameter. p90 uses NumPy default linear percentile behavior. A stronger d has finite mean_s(d) > mean_s(c), strictly; ties are not stronger. distance_s(c,d) is zero for overlap, otherwise the nonnegative interval gap in seconds. "20m" means distance <= 1200 seconds.

Source-specific required data missing produces NaN. If source mean exists with no stronger candidate, stronger_count is 0.0, nearest_stronger_distance_s is NaN and best_mean_margin is 0.0 for a tied/top candidate. All finite inputs produce finite outputs; no infinity is allowed. The existing fold-contained imputer handles NaNs.

Ordered columns:
1. macro_candidate_mean_percentile
2. macro_candidate_max_percentile
3. macro_candidate_mean_robust_z
4. macro_candidate_mean_relative_to_session_median
5. macro_candidate_max_relative_to_session_p90
6. macro_candidate_mean_to_session_max
7. macro_best_mean_margin
8. macro_stronger_count_20m
9. macro_nearest_stronger_distance_s
10. micro_candidate_mean_percentile
11. micro_candidate_max_percentile
12. micro_candidate_mean_robust_z
13. micro_candidate_mean_relative_to_session_median
14. micro_candidate_max_relative_to_session_p90
15. micro_candidate_mean_to_session_max
16. micro_best_mean_margin
17. micro_stronger_count_20m
18. micro_nearest_stronger_distance_s
19. session_candidate_count
20. session_candidate_rate_per_hour
21. candidate_duration_percentile
22. candidate_duration_robust_z
23. candidate_support_count
24. candidate_support_coverage_percentile
25. candidate_support_coverage_robust_z

For each source prefix, percentile/robust-z/best margin/competition use same-session raw-union source candidates. Session median, p90, and max use the full finite same-session probability stream. session_candidate_count counts raw-union candidates for sid. session_candidate_rate_per_hour divides this count by max((session_end-session_start)/3600000, 1/3600), a fixed one-second guard. Duration and coverage fields compare only candidates with finite relevant values.

Emit an atomic descriptor at outputs/challenger_70/context_v2_schema.json before any formal outer run; the formal E02 directory may contain an identical content-addressed copy. It includes name, definition, data_dependency, label_dependency=false, session_local=true, learned=false, width, ordered-column SHA-256.

## Session and gap safety

Context-v2 is pure: (candidates, macro_windows_by_sid, micro_windows_by_sid, session_bounds_by_sid) -> candidate-aligned float64 matrix. It rejects unknown/missing bounds, out-of-bound candidates, nonfinite/unordered time or windows, and infinity. It does not bridge, generate or interpolate candidates/windows. Existing candidate construction remains the gap-safe semantic authority.

The implementation canonical-sorts its same-session working view while returning rows in caller order. Output is invariant to candidate/window input order and to candidates in other sessions. It never joins sessions even when timestamps touch.

## Fixed E02 selection surface

Only context_features_version changes from v1 to v2. The formal execution requires NumPy 2.4.6. Preserve the entire incumbent registered surface: experiment key 160afaf81debf1ee; fold config hashes `(afcf9609a2f6925a, 8b3e27bd2796d038, b463226db1e6073a, 3f26fcb2172cd882, 0199041f39a09f5d)`; four subject-disjoint inner splits; seed 20260908; workers 5; no_tcn true; device auto; macro-v1 63D; gravity-aligned micro-v1 47D; candidate control true; verifier feature mode probability; Logistic C `(0.1)`; external FD weight `(0.0)`; micro threshold `(0.10,0.20,0.30,0.40,0.50)`; blend `(0.0,0.25,0.5,0.75,1.0)`; admission NMS IoU `(0.3,0.5,0.7)`; admission threshold `(0.2,0.35,0.5,0.65)`; admission cap `(3,4,5,6,8)`; admission minimum recall exactly `0.80`; no final subject cap; density `{stride=15000, window=240000, bridge=60000, density=600000, min_positive=10, coverage_min=.80, merge=120000, context=1200000, coverage_fix=false, window_threshold=.28838}`; micro candidate `{stride=7500, window=15000, smooth_sigma=30000, smooth_radius=60000, merge=180000, min_duration=60000, context=1200000}`; and micro_positive_middle_fraction null. Existing inner-OOF policy authority selects thresholds, blend, admission and event policy. There is no Context-v2-specific grid.

The mechanism gate is an inner-only paired v1/v2 comparison for each outer fold. `OuterTrainOnlyData` is a distinct immutable data type containing only outer-train batches, truths, session bounds, subject-by-session mapping and train subject IDs; it has no validation field. `FilesystemDataSource.load_outer_train_only(config)` is a new loader that reads only train-role paths and must not call or wrap `load_outer_fold`. Tests install sentinels on every validation-path/window/label accessor and assert no call occurs. Each branch accepts only OuterTrainOnlyData, makes subject-disjoint inner OOF candidates/scores and policy decisions, then discards them. Recompute v1 through this new path and require parity with frozen incumbent inner evidence before considering v2.

The gate passes only if:
1. v1 parity and Gate-0 tests pass;
2. pooled v2 inner F1 is strictly greater than paired v1;
3. v2 inner FP is lower and v2 inner TP loss <= 2;
4. at least 3/5 outer-train partitions are non-worse in inner F1.

Failure records E02 as stopped_inner_gate, with no formal outer run and no feature revision. The gate is development/mechanism evidence, not an independent estimate: its five outer-train partitions and the later formal outer folds belong to the same repeated-development programme. It must be reported separately and cannot be used to claim confirmatory generalization. This is not E03: no label weighting changes, so no threshold-only challenger belongs in E02.

## Formal evidence and safety

Before a full run, atomically create plan.json containing base run, code SHA, NumPy version, 25-name schema hash, all fixed grids, hypothesis, selection protocol, aggregate episode F1 primary metric and declared secondaries. The E02 cache key is the SHA-256 of a canonical object containing the literal namespace `e02-context-v2`, complete RunConfig, code SHA, exact NumPy version, and 25-column schema hash; it is stored only at cache/challenger_70/e02-context-v2/<key>/. Never read/write incumbent cache keys, outputs/crossfit, release registry, models, or dist.

After a passed gate, run exactly one strict subject-disjoint five-fold evaluation. Fold fitting, imputation, normalization, policy selection and caching remain outer-train/inner-OOF contained. Outer labels do not enter selection. The full command fails closed if a terminal E02 registry row exists or if the E02 output directory contains anything beyond the immutable preregistration allowlist `{plan.json, context_v2_schema.json, inner_gate.json}`; it has no `--force`, overwrite, resume, or rerun mode. Record per-fold and pooled counts/F1/PPV/recall/slices/candidate volume, stability, runtime/RSS and the full event accounting against frozen E00: TP retained/recovered/lost; FP retained/removed/added; event_diff.csv, fp_removed.csv, fp_added.csv, tp_recovered.csv, tp_lost.csv, comparison.md and leakage_audit.md.

Append one terminal record, including a failed gate, to outputs/challenger_70/experiment_registry.jsonl. A completed outer run is immutable; future ideas require a new experiment ID. Even F1 >= .70 only returns READY FOR HUMAN PROMOTION REVIEW. E02 never promotes, releases, changes the incumbent registry, packages, or rebuilds dist.

## Required Gate-0 tests

Test exact schema/hash/widths 56/116/141; no labels in API/schema; byte/value v1 parity; stable ties/input-order invariance; missing/zero semantics; session isolation/bounds/gap safety; no infinity; pre-score competition; fold-contained cache identity; outer-subject exclusion from inner-only gate; preregistration before full execution; and E00 v1 parity at 114/153/197. CPU-only runtime records per-session duration and peak RSS, flagging but not retuning a >2x incumbent-duration increase.
