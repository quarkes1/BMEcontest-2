# Challenger E03 Hard-Negative Verifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test exactly one frozen-v1 change—outer-train, inner-OOF hard-negative sample weighting—against an unweighted threshold-only control without looking at outer validation data.

**Architecture:** A small challenger-only miner converts inner cross-fitted logistic scores and fixed v1 candidate evidence into one deterministic sample-weight vector.  A challenger runner orchestrates the paired unweighted and weighted arms using the existing nested runner mechanics, but owns preregistration, cache/evidence isolation, the fixed gate, and no-promotion semantics.  The incumbent release remains only a verified reporting anchor.

**Tech Stack:** Python 3.12, NumPy 2.4.6, scikit-learn pipelines, LightGBM, existing strict subject-disjoint nested CV, pytest.

**Spec:** `docs/superpowers/specs/2026-09-21-challenger-e03-hard-negative-verifier-design.md`

## Global Constraints

- E03-001 (`e03-v1-hard-negative`) is derived from failed E02-005: it is v1-only (63 macro, 47 micro and 116 verifier columns); v2 is forbidden and the plan/registry must record that E02's TP-loss gate failed.
- `release/event_stack_incumbent.json`, `models/event_stack/160afaf81debf1ee/**`, incumbent attestation and `dist/**` are read-only.
- Fixed mining: OOF logistic C=.1; high `>=.50`; near `[.40,.60]`; disagreement uses `nan_to_num(v1[:,0], nan=0.0)` versus `nan_to_num(v1[:,41], nan=0.0)` at `>=.35`; dense is `>=2` other raw-union candidates in the same session within inclusive 600000 ms; negative hard weight 2.0; all others 1.0.
- Existing balanced class weights, LogisticRegression, restricted LGBM and all incumbent policy grids remain unchanged.
- Only the E03 orchestrator takes `OuterTrainOnlyData`; the pure miner takes exactly candidates, 116D v1 features, labels, groups and OOF logistic scores.  No outer validation signal, label, score, subject map, truth, dataset object or final model can enter that miner.
- Write only atomically under `cache/challenger_70/e03-v1-hard-negative/`, `outputs/challenger_70/experiments/E03-001/`, and one appended E03-001 registry row.  No force/resume/overwrite/promotion option exists.
- Formal E03 runs only after immutable plan/schema/code identity checks and passed inner gate.  It runs paired full v1 threshold-only and v1 hard-negative arms once.

---

## File structure

- `src/pipeline/challenger/hard_negative.py`: pure hardness predicates and sample weights; it has no dataset loading, truth matching, model fitting or source-map dependency.
- `src/pipeline/runner.py`: narrowly accepts optional aligned verifier sample weights for cross-fit/final fitting while preserving the incumbent call paths bit-for-bit.
- `src/pipeline/challenger/e03.py`: restricted E03 inner gate, full paired evaluator, cache/output/registry guards and accounting.
- `scripts/run_challenger_e03.py`: one command-line entrypoint limited to `--phase inner-gate|full`.
- `tests/pipeline/test_challenger_hard_negative.py`: pure predicate, weight, schema and isolation unit tests.
- `tests/pipeline/test_challenger_runner_e03.py`: protected runner, gate, artifact and once-only behavioural tests.

### Task 1: Deterministic hard-negative miner

**Files:**
- Create: `src/pipeline/challenger/hard_negative.py`
- Create: `tests/pipeline/test_challenger_hard_negative.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class HardNegativeEvidence:
    candidates: Sequence[MultiScaleCandidate]
    v1_features: np.ndarray
    labels: np.ndarray
    oof_logistic_scores: np.ndarray
    groups: np.ndarray

@dataclass(frozen=True)
class HardNegativeResult:
    sample_weight: np.ndarray
    category_mask: Mapping[str, np.ndarray]
    hard_mask: np.ndarray

def mine_hard_negatives(evidence: HardNegativeEvidence) -> HardNegativeResult: ...
def assert_outer_subject_disjoint(train_subjects: Iterable[str], outer_subjects: Iterable[str]) -> None: ...
def hard_negative_spec_descriptor() -> dict[str, object]: ...
```

- [ ] **Step 1: Write failing predicate/edge tests.**

```python
def test_negative_union_of_fixed_categories_is_weighted_once():
    evidence = _evidence(scores=[.5, .4, .9, .1, .1], labels=[0, 0, 0, 1, 0],
                         v1_columns={0: [.8, .1, .1, .8, .1], 41: [.1, .1, .1, .1, .1]},
                         centers=[0, 600_000, 1_200_000, 1_210_000, 1_215_000])
    result = mine_hard_negatives(evidence)
    assert result.sample_weight.tolist() == [2., 2., 2., 1., 2.]
    assert result.hard_mask.tolist() == [True, True, True, False, True]

def test_dense_uses_same_session_inclusive_center_distance_and_two_others():
    result = mine_hard_negatives(_three_neighbours_at_centers([0, 600_000, 1_200_000]))
    assert result.category_mask["dense"].tolist() == [False, True, False]

def test_disagreement_uses_exact_v1_indices_and_missing_zero_semantics():
    evidence = _evidence(scores=[.1, .1], labels=[0, 0],
                         v1_columns={0: [np.nan, .35], 41: [.35, np.nan]})
    assert mine_hard_negatives(evidence).category_mask["disagreement"].tolist() == [True, True]

def test_outer_subject_intersection_is_rejected():
    with pytest.raises(ValueError, match="outer-validation"):
        assert_outer_subject_disjoint({"s1"}, {"s1", "s2"})
```

- [ ] **Step 2: Run RED.**

```text
D:/Anaconda3/envs/bme/python.exe -m pytest tests/pipeline/test_challenger_hard_negative.py -v
```

Expected: import failure because the module does not yet exist.

- [ ] **Step 3: Implement validation and pure miner.**

```python
negative = evidence.labels == 0
high = negative & (evidence.oof_logistic_scores >= .50)
near = negative & (evidence.oof_logistic_scores >= .40) & (evidence.oof_logistic_scores <= .60)
macro_mean = np.nan_to_num(evidence.v1_features[:, 0], nan=0.0)
micro_mean = np.nan_to_num(evidence.v1_features[:, 41], nan=0.0)
disagreement = negative & (np.abs(macro_mean - micro_mean) >= .35)
hard = high | near | disagreement | dense
weights = np.where(hard, 2.0, 1.0).astype(np.float64)
```

Validate equal non-empty candidate/feature/label/group/score lengths, exact 116 feature width, binary labels, finite scores and valid intervals.  Permit the incumbent's canonical Context-v1 NaNs anywhere in the 116D feature matrix, reject +/-Inf and all-missing columns exactly as the candidate-control verifier fit does, and replace NaN with zero only when reading the two registered disagreement inputs.  Calculate centers as float64 from candidates, never compare candidates from different sessions, and count other rows only.

Implementation correction after frozen `E03-004`: Context-v1 deliberately emits
NaN for unavailable session-local summaries and the incumbent folds them with a
fold-fitted median imputer.  The prior wording incorrectly limited missingness
to columns 0 and 41, which rejected the canonical v1 matrix before mining.

- [ ] **Step 4: Run GREEN.**

```text
D:/Anaconda3/envs/bme/python.exe -m pytest tests/pipeline/test_challenger_hard_negative.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit.**

```text
git add src/pipeline/challenger/hard_negative.py tests/pipeline/test_challenger_hard_negative.py
git commit -m "feat(challenger): add deterministic hard-negative mining"
```

### Task 2: Weight-aware verifier fitting without incumbent regression

**Files:**
- Modify: `src/pipeline/runner.py`
- Modify: `tests/pipeline/test_challenger_hard_negative.py`
- Modify: `tests/pipeline/test_runner.py`

**Interfaces:**

```python
def _fit_final_models(..., verifier_sample_weight: np.ndarray | None = None) -> _FinalModelFit: ...
def crossfit_predict_proba(..., fit_sample_weight: np.ndarray | None = None) -> CrossfitResult: ...
```

`None` must retain every existing fit call exactly.  A provided vector must be float64, finite, strictly positive and row-aligned; each grouped inner fit receives the indexed training slice through `model__sample_weight`.  Both LogisticRegression and LGBM pipelines retain `class_weight="balanced"`, their current fixed seeds and every registered estimator parameter.

- [ ] **Step 1: Write failing alignment, slice and parity tests.**

```python
def test_weighted_crossfit_never_passes_validation_weights_to_fit(monkeypatch):
    seen = []
    monkeypatch.setattr(Pipeline, "fit", lambda self, x, y, **kw: seen.append(kw["model__sample_weight"]) or self)
    crossfit_predict_proba(x, y, groups, x, groups, 2, factory, fit_sample_weight=np.array([1.,2.,3.,4.]))
    assert all(len(weight) == 2 for weight in seen)

def test_unweighted_estimator_fit_arguments_are_unchanged(monkeypatch):
    observed = _capture_pipeline_fit_kwargs(monkeypatch)
    crossfit_predict_proba(x, y, groups, x, groups, 2, factory, fit_sample_weight=None)
    assert all("model__sample_weight" not in kwargs for kwargs in observed)

def test_bad_weight_length_zero_or_nan_rejects_before_model_fit():
    for bad in (np.array([]), np.array([1., np.nan, 1., 1.])):
        with pytest.raises(ValueError):
            crossfit_predict_proba(x, y, groups, x, groups, 2, factory, fit_sample_weight=bad)

@pytest.mark.parametrize("factory", [_verifier_estimator, _verifier_lgbm_estimator])
def test_weighted_verifier_pipeline_fit_smoke_preserves_balanced_class_weight(factory):
    pipeline = factory(20260911) if factory is _verifier_estimator else factory()
    pipeline.fit(x, y, model__sample_weight=np.array([1., 2., 1., 2.]))
    assert pipeline.named_steps["model"].get_params()["class_weight"] == "balanced"
```

- [ ] **Step 2: Run RED.**

```text
D:/Anaconda3/envs/bme/python.exe -m pytest tests/pipeline/test_challenger_hard_negative.py tests/pipeline/test_runner.py -k "weight or unweighted" -v
```

Expected: failures because the baseline fitting APIs have no sample-weight path.

- [ ] **Step 3: Implement the narrow optional argument.**

Construct `fit_kwargs = {}` for `None`; otherwise use only
`{"model__sample_weight": weights[train_indices]}`.  Apply this rule to
logistic and LGBM cross-fit/final verifier fits, never macro or micro models.
Thread the optional value only through challenger-specific calls; leave every
incumbent `RunConfig` result, serialization and cache key unchanged.

- [ ] **Step 4: Run focused GREEN and frozen baseline parity.**

```text
D:/Anaconda3/envs/bme/python.exe -m pytest tests/pipeline/test_challenger_hard_negative.py tests/pipeline/test_runner.py tests/parity/test_promoted_release_baseline.py -v
```

Expected: PASS; the promoted release test still reports F1 .6514285714.

- [ ] **Step 5: Commit.**

```text
git add src/pipeline/runner.py tests/pipeline/test_challenger_hard_negative.py tests/pipeline/test_runner.py
git commit -m "feat(challenger): support isolated verifier sample weights"
```

### Task 3: E03 inner-only paired gate

**Files:**
- Create: `src/pipeline/challenger/e03.py`
- Create: `scripts/run_challenger_e03.py`
- Modify: `tests/pipeline/test_challenger_runner_e03.py`

**Interfaces:**

```python
def registered_e03_config(outer_fold: int, *, workers: int) -> RunConfig: ...
def write_e03_preregistration(root: Path, output: Path) -> Path: ...
def run_e03_inner_gate(root: Path, output: Path, *, workers: int) -> dict[str, object]: ...
def e03_gate_passes(control: Sequence[EventMetrics], hard: Sequence[EventMetrics]) -> bool: ...
```

- [ ] **Step 1: Write failing guard and gate tests.**

```python
def test_gate_requires_all_four_preregistered_predicates():
    assert not e03_gate_passes(_metrics(tp=100, fp=20, f1=.80, folds=5),
                               _metrics(tp=97, fp=10, f1=.81, folds=5))
    assert e03_gate_passes(_metrics(tp=100, fp=20, f1=.80, folds=5),
                           _metrics(tp=99, fp=16, f1=.81, folds=5, nonworse=3))

def test_inner_only_function_rejects_fold_dataset_and_outer_subjects():
    with pytest.raises(TypeError):
        build_e03_inner_weights(FoldDataset(...), config)

def test_plan_records_v1_only_all_constants_and_mining_spec_hash(tmp_path):
    plan = json.loads(write_e03_preregistration(root, tmp_path).read_text())
    assert plan["feature_version"] == "v1"
    assert plan["allowed_hyperparameter_grid"]["hard_negative_weight"] == [2.0]
```

- [ ] **Step 2: Run RED.**

```text
D:/Anaconda3/envs/bme/python.exe -m pytest tests/pipeline/test_challenger_runner_e03.py -v
```

Expected: import/API failures.

- [ ] **Step 3: Implement gate only.**

For every outer fold, load only `FilesystemDataSource.load_outer_train_only`.
The orchestrator generates normal v1 candidates/features/labels/groups, then
cross-fits unweighted logistic mining scores by groups and passes only those
five aligned inputs to the pure miner.  It derives weights, asserts groups are outer-subject-disjoint,
and independently produce threshold-only and weighted inner scores with the
same fixed selection grid.  Write only `plan.json` and then atomic
`inner_gate.json`; a failure appends `stopped_inner_gate` to the registry and
does not create an outer artifact.  Require NumPy 2.4.6, protected snapshots,
clean working tree except allowed untracked bytecode, a fresh output root, and
a cache identity containing source input fingerprints, exact estimator
parameter dictionaries/seeds, imputer contract, raw-union order contract and
policy tie-order contract.  The gate aggregates five outer-train runs, each
with exactly four inner splits: pooled counts are integer sums and use
`2*TP/(eligible+pred)`; strict improvement is `>` and non-worse is `>=`, both
with no tolerance.

- [ ] **Step 4: Run GREEN.**

```text
D:/Anaconda3/envs/bme/python.exe -m pytest tests/pipeline/test_challenger_hard_negative.py tests/pipeline/test_challenger_runner_e03.py tests/pipeline/test_challenger_protection.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit.**

```text
git add src/pipeline/challenger/e03.py scripts/run_challenger_e03.py tests/pipeline/test_challenger_runner_e03.py
git commit -m "feat(challenger): register E03 hard-negative inner gate"
```

### Task 4: Paired, once-only strict outer evidence and accounting

**Files:**
- Modify: `src/pipeline/challenger/e03.py`
- Modify: `scripts/run_challenger_e03.py`
- Modify: `tests/pipeline/test_challenger_runner_e03.py`
- Modify: `tests/pipeline/test_challenger_atlas.py`

**Interfaces:**

```python
def run_e03_full_outer(root: Path, output: Path, *, workers: int) -> dict[str, object]: ...
def compare_e03_arms_and_incumbent(...) -> dict[str, object]: ...
```

- [ ] **Step 1: Write failing once-only, paired-arm and outer-isolation tests.**

```python
def test_full_requires_passed_immutable_gate_and_runs_each_arm_once(tmp_path):
    _write_passed_plan_and_gate(tmp_path)
    run_e03_full_outer(root, tmp_path, workers=1)
    with pytest.raises(FileExistsError, match="already terminal"):
        run_e03_full_outer(root, tmp_path, workers=1)

def test_full_records_control_and_weighted_event_diffs_without_promotion(monkeypatch, tmp_path):
    result = run_e03_full_outer(root, tmp_path, workers=1)
    assert {"threshold_only", "hard_negative_v1"} <= result["arms"].keys()
    assert not (tmp_path / "promotion.json").exists()

def test_outer_subject_is_never_in_weighted_fit_subjects():
    with pytest.raises(ValueError, match="outer-validation"):
        assert_outer_subject_disjoint({"held_out"}, {"held_out"})
```

- [ ] **Step 2: Run RED.**

```text
D:/Anaconda3/envs/bme/python.exe -m pytest tests/pipeline/test_challenger_runner_e03.py tests/pipeline/test_challenger_atlas.py -v
```

Expected: missing full-evaluation/artifact behaviour.

- [ ] **Step 3: Implement full evidence with no selection from outer results.**

Validate plan hash/code/schema/gate/snapshots before allocating a formal
output.  For folds 0–4, run both independent selected arms from the same
outer-train partition and score each once on untouched outer validation.
Persist per-fold subject hashes, final mining summaries, v1 feature width,
selected admission/event policy and metrics.  Aggregate using the existing
official `.25` matcher.  Compute per-arm incumbent event accounting with
stable event identifiers; do not feed those diffs into any selection.  Emit
all named output files atomically, append one terminal `complete` row, and
resnapshot protected paths.

- [ ] **Step 4: Run GREEN.**

```text
D:/Anaconda3/envs/bme/python.exe -m pytest tests/pipeline/test_challenger_hard_negative.py tests/pipeline/test_challenger_runner_e03.py tests/pipeline/test_challenger_atlas.py tests/pipeline/test_challenger_protection.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit.**

```text
git add src/pipeline/challenger/e03.py scripts/run_challenger_e03.py tests/pipeline/test_challenger_runner_e03.py tests/pipeline/test_challenger_atlas.py
git commit -m "feat(challenger): record paired E03 outer evidence"
```

### Task 5: Verification and one registered experiment

**Files:** Generate only `outputs/challenger_70/experiments/E03-001/**` and append `outputs/challenger_70/experiment_registry.jsonl`.

- [ ] **Step 1: Run complete relevant verification.**

```text
D:/Anaconda3/envs/bme/python.exe -m pytest tests/pipeline/test_event_stack.py tests/pipeline/test_runner.py tests/pipeline/test_crossfit.py tests/pipeline/test_challenger_hard_negative.py tests/pipeline/test_challenger_runner_e03.py tests/pipeline/test_challenger_protection.py tests/pipeline/test_challenger_atlas.py tests/parity/test_promoted_release_baseline.py -v
```

Expected: all selected tests pass and incumbent parity remains 0.6514285714.

- [ ] **Step 2: Register and run the inner gate once.**

```text
D:/Anaconda3/envs/bme/python.exe scripts/run_challenger_e03.py --root . --output outputs/challenger_70/experiments/E03-001 --phase inner-gate --workers 5
```

Expected: immutable plan and a terminal gate report.  If gate fails, commit
only that evidence and stop; remove only the validated disposable E03 cache.

- [ ] **Step 3: If and only if the recorded gate passes, run full paired evidence once.**

```text
D:/Anaconda3/envs/bme/python.exe scripts/run_challenger_e03.py --root . --output outputs/challenger_70/experiments/E03-001 --phase full --workers 5
```

Expected: complete paired five-fold evidence, with no release or dist change.

- [ ] **Step 4: Commit immutable evidence.**

```text
git add outputs/challenger_70/experiments/E03-001 outputs/challenger_70/experiment_registry.jsonl
git commit -m "data(challenger): record E03 hard-negative result"
```

## Plan self-review

The design's v1-only decision, four exact mining categories, single weight,
same policy grid controls, inner gate, full paired outer once-only execution,
protected release/dist, isolation, cache identity, diagnostics and no-promotion
requirement map to Tasks 1–5.  No Context-v2, micro-v2, new model, threshold
surface, outer-driven selection, unbounded cache or release action appears in
the plan.
