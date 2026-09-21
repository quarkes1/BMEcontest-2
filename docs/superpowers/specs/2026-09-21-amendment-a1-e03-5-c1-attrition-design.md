# Amendment A1 — E03.5 C1 Attrition Audit

## Authority and scope

This amendment supplements `docs/challenge设计任务书.md`; it does not replace the
incumbent-protection or strict subject-disjoint rules. E01 found 19 short
false negatives, of which 5 are C0 and 11 are C1. E02 and E03 failed their
registered inner gates. Therefore E04/E05 micro-v2 work is paused. E03.5 is a
sealed-outer, read-only **diagnostic**, followed by human review before any
new challenger hypothesis is selected.

E03.5 does not train, refit, recalibrate, tune, or promote a model. It must not
change `release/`, the incumbent bundle, `dist/`, canonical v1 schemas,
candidate-generation code, or the official matcher. It produces no new F1
headline, no `outer_f1` registry field, and no release candidate. Its numbers
are development diagnostics from already-used outer folds, not an independent
generalization estimate.

## Frozen evidence and baseline checks

Use only `replay_outer_fold(root, fold)` for folds 0–4 and its attested
incumbent bundles. Each trace must reproduce its sealed TP/eligible/pred and
the pooled 114/153/197. The initial E01 classification must again identify
exactly 11 short FN-C1 truth keys using the unchanged IoU 0.25 rule. Abort on
any discrepancy before writing an output. Protect incumbent roots with the
existing snapshot guard. Record code SHA, run key, source/evidence hashes and
the amendment spec hash in output provenance.

For every replay trace, the raw-union candidate list, macro/micro support,
candidate geometry, 116D Context-v1 features, Logistic and LGBM scores,
incumbent blend scores, selected admission tuple, event threshold and event
cap are immutable. No operation may recompute scores or choose a new policy.

## Candidate and truth attrition

For each of the 11 short C1 truths, enumerate every raw-union candidate with
IoU >= 0.25 and retain its stable `(outer_fold, raw_index)` coordinate.
Export score, source support, IoU, NMS suppressor, admission-threshold pass,
subject-cap rank, admission/final indices, and the existing ledger terminal
reason. Independently replay the frozen admission pipeline in the registered
order `NMS -> admission threshold -> subject cap` and assert that its final
indices equal `trace.admitted_indices`; the ledger is evidence, not the sole
authority.

For each truth, report reachability after every stage. Its **first total-loss
stage** is the first stage at which no IoU-reachable candidate survives.
Separately report the set of terminal reasons among all its reachable raw
candidates; when multiple reasons occur, label the truth `mixed` rather than
forcing a single causal explanation. Candidate-level counts and truth-level
counts must remain separate because one truth can have multiple candidates.

## Frozen-score, one-operation counterfactuals

Exactly three interventions are registered:

1. `disable_nms_only`: start from all raw-union candidates, then apply the
   incumbent admission threshold and subject cap.
2. `disable_admission_threshold_only`: apply incumbent NMS, then the incumbent
   subject cap without an admission score threshold.
3. `disable_subject_cap_only`: apply incumbent NMS and admission threshold,
   without the admission subject cap.

All non-disabled operations retain the incumbent constants and canonical tie
orders. Each intervention starts independently from the same frozen raw union,
never from another intervention's result. The incumbent event threshold and
event cap then run unchanged on the intervention's admitted candidates with
the original frozen blend scores. No verifier call, score recomputation,
policy search, or threshold optimization is permitted.

For each intervention, report candidate-stage and final-event effects:

- newly admitted raw candidates, including how many have no IoU >= 0.25 to
  any eligible truth (`extra_nonmatching_candidates`, GT-aware diagnostic);
- C1 short truths newly reachable after admission, C1 short truths newly
  matched by the unchanged official matcher, and C1 short truths still lost
  by the downstream event policy or matching;
- newly introduced unmatched final events, removed incumbent unmatched
  events, and net FP difference. Event identity uses `(fold, sid, start_ms,
  end_ms)` with multiplicity-safe coordinates when duplicates occur; counts
  come from the official matcher, not from candidate labels;
- newly matched and lost truths outside the C1 set, so a rescue is not
  presented without its collateral effects.

The report must show both gross introduced FP and net FP. A changed event
set can remove an old FP while introducing a new one; reporting only net FP
would conceal that cost. No intervention is a challenger result or F1 claim.

## GT-aware oracle upper bound

Mark every oracle artifact `DIAGNOSTIC ONLY / USES GT`. For each C1 truth
independently, force **one** of its raw IoU-reachable candidates into the
incumbent admitted set. Do not alter that candidate's score or geometry.
Apply the unchanged incumbent event threshold/cap and official matcher. The
oracle asks whether **any** such single-candidate choice yields a match for
the target truth. Deterministic witness choice: among successful choices,
select highest target IoU, then highest frozen blend score, then lowest raw
index; if none succeeds, report the best attempted candidate by the same
ordering and its downstream rejection reason. Restore the baseline before
testing the next truth; oracle choices are never combined into a synthetic
run. This prevents an aggregate GT-aware F1 from being mistaken for a result.

Distinguish `control_limited_and_downstream_capable` from
`downstream_still_rejects`. The oracle is not training data, a feature,
a policy-selection criterion, or an input to a future inner gate.

## Output contract

Write only new immutable files under
`outputs/challenger_70/e03_5_c1_attrition/`:

- `manifest.json`: provenance, protection snapshot, frozen policy hashes,
  exact scope, and prominent diagnostic-only warning;
- `candidate_attrition.csv`: one row per C1 truth/raw reachable candidate;
- `truth_attrition.csv`: one row per C1 truth, stage reachability and mixed
  reasons;
- `counterfactual_events.csv`: multiplicity-safe per-event differences for
  each registered intervention;
- `counterfactual_summary.json`: the three frozen-score interventions and
  their GT-recovery/FP-cost counts;
- `oracle.csv`: one independent GT-aware upper-bound row per C1 truth;
- `report.md`: compact cause/intervention table and evidence-backed possible
  hypotheses, with **no automatic parameter or challenger recommendation**.

Do not write an official-performance row to
`outputs/challenger_70/experiment_registry.jsonl`. Existing output is
immutable: no resume, overwrite, force, or silent replacement. A failed
diagnostic attempt gets a separately named failure record, not a modified
result. Do not retain a new cache when the report is complete.

## Correctness gates

Tests must prove: exact 11-key target set; fixed scores and policy; the three
single-operation ablations; canonical admission parity in baseline; NMS,
threshold and cap tie behavior; mixed-cause truth accounting; official
matcher-based TP/FP event differences; oracle one-truth-at-a-time reset and
unchanged downstream threshold/cap; no model `fit` anywhere and no additional
`predict_proba` after sealed replay has produced the incumbent scores; no
F1/`outer_f1` headline or registry write; immutable
output and protected-snapshot parity. Run the existing incumbent parity suite
before accepting the artifact. Any mismatch or leakage concern stops E03.5.

After E03.5, a human reviews the cause/FP-cost table. Only then may a new,
narrow candidate-control challenger be designed and preregistered for an
inner-only gate. E04/E05 remain paused.
