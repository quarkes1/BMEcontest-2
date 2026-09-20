# Challenger E03 Hard-Negative Verifier Design

## Decision

E03 is warranted, but it must use the frozen **Context-v1, 116-column**
representation only.  The incumbent OOF atlas identifies 83 false positives
with stable, actionable structure: 61 have macro+micro support, 18 are
micro-high/macro-low, 38 occur in dense candidate clusters, and 32 are short
spikes.  This is enough evidence to test a weighting mechanism, rather than a
new representation or a new candidate generator.

E02-005 is not a positive representation result.  Its pooled inner result is
only 0.765665 versus the paired v1 0.765599, while it loses eight TP (446 vs
454) and its registered gate failed.  It is therefore excluded from E03, not
used as an optional arm, and must not be combined with hard-negative weighting.
That choice keeps the causal question single-factor: does training emphasis on
hard *outer-train* negatives improve over the same v1 model and policy search?

This is an explicit taskbook deviation and derived experiment, recorded as
`experiment_id="E03-001"`, `experiment_slug="e03-v1-hard-negative"`, and
`derived_from="E02-005"`.  The taskbook names Track B after the atlas but does
not prescribe that it must consume the failed Track A representation.  The
preregistration and registry must say why: E02-005 failed its loss-of-TP gate,
so E03-001 intentionally isolates a v1 weighting hypothesis rather than
testing the unidentifiable compound change `Context-v2 + hard negatives`.

## Scope and immutable inputs

E03 may add challenger-only code and evidence below these roots:

```text
src/pipeline/challenger/hard_negative.py
scripts/run_challenger_e03.py
tests/pipeline/test_challenger_hard_negative.py
tests/pipeline/test_challenger_runner_e03.py
cache/challenger_70/e03-v1-hard-negative/
outputs/challenger_70/experiments/E03-001/
outputs/challenger_70/experiment_registry.jsonl
```

It must not modify the incumbent pointer, `models/event_stack/160afaf81debf1ee`,
its attestation, existing v1/v2 definitions, `dist/**`, the official matcher,
or deployment inference.  E03 has no promotion or release path.

The fixed representation and candidate/policy surface are copied from the
incumbent/E02 v1 control:

```text
outer folds=0..4; inner_splits=4; seed=20260908; no_tcn=true
macro=incumbent DensityConfig; micro=15 s/7.5 s incumbent path
context=v1; verifier width=116; admission_minimum_recall=.80
verifier C=(.1,); blend=(0,.25,.5,.75,1)
micro threshold=(.10,.20,.30,.40,.50)
admission NMS IoU=(.3,.5,.7); threshold=(.2,.35,.5,.65);
subject cap=(3,4,5,6,8)
```

No E03 option may add a feature, alter candidate geometry, alter model class,
alter class weight, add external data, or alter this policy grid.

## Leakage-safe hard-negative construction

The E03 orchestrator alone receives `OuterTrainOnlyData`.  It builds the
ordinary v1 inner raw-union candidates, the 116D v1 matrix, candidate labels,
and subject-group vector with current runner helpers.  It then obtains an
out-of-fold **logistic-only** score for every candidate by a four-way
subject-grouped cross-fit of the unweighted incumbent LogisticRegression
(`C=.1`, existing balanced class weight).  No final fitted verifier model is
ever used to mine hardness.

The pure miner accepts exactly `(candidates, v1_features, labels, groups,
oof_logistic_scores)`.  It does **not** accept `OuterTrainOnlyData`,
`FoldDataset`, truths, a subject map, outer-validation arrays, or a final
model.  `groups` is permitted solely for aligned provenance reporting and is
not used to derive a predicate.  The score used to decide hardness is never an
in-sample score and never a final outer prediction.  The restricted LGBM is
deliberately not used to mine; this avoids a blend/policy circular dependency.

For negative candidates only (`label == 0`), set `hard=True` when one or more
of these fixed, label-free-after-negative-identification predicates holds:

| Category | Exact predicate |
| --- | --- |
| `high_score` | logistic OOF probability `>= 0.50` |
| `near_threshold` | probability in `[0.40, 0.60]` inclusive |
| `disagreement` | `abs(macro_mean - micro_mean) >= 0.35`, where `macro_mean=nan_to_num(v1_features[:,0], nan=0.0)` and `micro_mean=nan_to_num(v1_features[:,41], nan=0.0)` |
| `dense` | at least two *other raw-union candidates* with the same session id and candidate-center distance `<= 600000 ms` |

`v1_features[:,0]` and `v1_features[:,41]` are the incumbent legacy
`macro_mean` and `micro_mean` columns, respectively; only missing (`NaN`)
values are replaced by `0.0`, before subtraction, and all other finite values
remain unchanged.  Candidate centers are `(start_ms + end_ms) / 2`; `dense`
uses the raw-union candidate list, same session id, inclusive `<=600000 ms`,
and explicitly excludes the candidate itself.  Each candidate is counted once
even if several predicates match.  Positive
candidates always have weight `1.0`; ordinary negatives have `1.0`; hard
negatives have the fixed weight `2.0`.  There is no sampling, no per-subject
normalisation, no weight grid, and no category-specific tuning.  The same
vector is passed as `model__sample_weight` to both final verifier pipelines.
Existing `class_weight="balanced"` remains intact in every arm.

The miner's candidate order is the canonical raw-union order from
`union_candidates`, retained unchanged through features, labels, groups,
cross-fit scores, weights, final fit and evidence rows.  Median imputation,
LogisticRegression/LGBM parameter dictionaries, all seeds, the existing
policy-ranking tie order, and input candidate ordering are immutable parts of
the E03 identity.

The final outer-train fit repeats the same nested process using only
outer-train groups: generate OOF mining scores/weights there, then fit the
final logistic and restricted LGBM on all outer-train candidates with those
weights.  The outer-validation candidates receive no labels, no mining score,
and no weight.  Their scores are produced exactly once by the fitted model.

## Three locked comparison arms

All arms share raw candidates, v1 features, candidate-control configuration,
and the exact policy grid above.

1. `incumbent_evidence` is the verified, read-only E00/release result
   (114/153/197; F1 .6514285714).  It is a reporting anchor, not reselected.
2. `threshold_only` reruns the ordinary unweighted v1 Logistic+restricted
   LGBM stack.  It is the paired contemporaneous control: it selects admission
   and event thresholds from its own inner OOF scores using the frozen grid.
3. `hard_negative_v1` differs from `threshold_only` only by the fixed
   outer-train hard-negative sample weights above.  It independently selects
   its own policy from the same grid and never imports the control's selected
   threshold.

Thus any difference between arms cannot be attributed to threshold search:
both control and challenger receive the same threshold/admission freedom.
The report must state this explicitly and record both selected policies per
fold.

## Gate, formal run, and artifacts

Before any outer evaluation, write immutable `plan.json` with the full
constants, categories, model parameters, code SHA, base run, v1 schema hash,
the E02-005 derivation rationale, cache identity and the statement
`selection_protocol=outer-train inner-OOF only`.  Cache identity is SHA-256
of literal namespace `e03-v1-hard-negative`, arm, complete `RunConfig`, code
SHA, NumPy version, Context-v1 schema hash, mining-spec hash, exact current
estimator parameter dictionaries, imputer contract, canonical candidate-order
contract, policy tie-order contract, and exact source input fingerprints for
the selected outer-train split/session/window files.  Cache reads/writes are
restricted to the E03 cache root and are atomic.

The inner gate executes five outer-train partitions (outer folds 0–4), each
with exactly `inner_splits=4`, and aggregates their held-out inner predictions.
For each arm, pooled `TP`, `eligible`, and `pred` are integer sums across all
five partitions, and `F1 = 2*TP/(eligible+pred)`.  `strictly improves` means
the resulting finite pooled float is `>` with no tolerance; `non-worse` means
the per-partition finite F1 is `>=` with no tolerance.  Existing candidate and
policy ranking ties retain the incumbent tuple order, so no new tie-break is
introduced.  `hard_negative_v1` passes only if all conditions hold against
`threshold_only`:

```text
pooled F1 strictly improves
pooled FP decreases by at least 4
pooled TP loses no more than 2
at least 3/5 outer-train partitions have non-worse F1
```

This is development evidence, not independent validation.  A failed gate
writes its immutable plan, one gate report and one terminal registry row, then
stops; it creates no `fold_*.json`, `summary.json`, event diff, comparison or
formal outer cache artifact.  The validated disposable E03 cache is removed.
If it passes, E03 runs the complete paired `threshold_only` and
`hard_negative_v1` five-fold strict outer evaluation exactly once.  It may not
resume, overwrite, force, or modify its hypothesis after any outer score.

Formal output contains per-arm/fold metrics, selected policy, subject sets,
hard-negative category/count/weight summaries, inner-to-outer gap, fold
stability, short/medium/long recall, candidate volume, and the mandatory
event-level accounting relative to the frozen incumbent:

```text
TP retained/new/lost; FP retained/removed/newly introduced
```

It also writes `event_diff.csv`, `comparison.md`, `leakage_audit.md`,
`summary.json`, `inner_gate.json`, and a terminal registry row.  No output
can drive promotion; a later promotion decision requires a separate user
approved release task.

## Safety and test requirements

Tests must prove the predicates and exact weights, OOF-only score provenance,
and that hard-negative training groups are disjoint from each outer-validation
subject set.  They must also reject labels/scores of wrong length, unknown
categories, duplicate/escaped output roots, an E02/v2 representation, missing
or changed preregistration, input snapshot changes, existing terminal output,
and promotion/release flags.  They must include one fit smoke test for each of
the LogisticRegression and LGBM pipelines confirming that
`model__sample_weight` reaches the model while `class_weight="balanced"` and
all registered estimator parameters remain intact.  A v1 parity test must
demonstrate that the unweighted arm invokes the same 116D feature surface and
baseline estimator parameters as the incumbent path.

## Non-goals

E03 does not implement Context-v2, micro-v2, a deep model, external data,
session-specific thresholds, new features, threshold-only tuning beyond the
already registered grid, dist changes, or cleanup/reorganisation work.
