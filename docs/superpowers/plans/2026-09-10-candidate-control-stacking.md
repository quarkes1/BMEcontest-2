# Candidate Control and OOF Stacking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce multiscale candidate noise and improve leakage-safe aggregate F1 beyond 0.478632 using train-only candidate admission and verifier stacking, then atomically promote every improvement into reproducible model artifacts and `dist/`.

**Architecture:** A pure candidate-control module performs stable same-session NMS and per-subject admission. The runner generates subject-disjoint OOF scores from logistic and LightGBM candidate verifiers, selects one blend/admission/event policy on outer-train only, and evaluates outer validation once. A separate artifact layer serializes accepted models/configuration and atomically rebuilds a CPU inference package.

**Tech Stack:** Python 3.11, NumPy, scikit-learn, LightGBM 4.7, joblib, pytest, PowerShell/Git Bash packaging.

**Spec:** `docs/superpowers/specs/2026-09-10-candidate-control-stacking-design.md`

## Global Constraints

- Use `D:/Anaconda3/envs/bme/python.exe` for every test, training, and packaging command.
- Preserve the frozen 153-event eligible denominator and exact macro/micro session-ID alignment.
- No outer-validation truth, labels, slices, or metrics may select any hyperparameter.
- Keep LightGBM `n_jobs=1`; bound fold-level parallelism by physical CPU count.
- `external_fd_weight_grid` remains exactly `(0.0,)` throughout this plan.
- `micro_enabled`, candidate control, and stacking remain opt-in until every registered default gate passes.
- Do not add one-off scripts. Delete verified smoke/temp/bytecode outputs and keep the worktree clean after each commit.
- Subagents must use Terra for runner/model/artifact integration and Luna for pure utilities, cleanup, and documentation.
- Any aggregate F1 above 0.478632 triggers artifact serialization, `dist/` refresh, README/architecture updates, and one coherent promotion commit.

---

### Task 1: Pure deterministic candidate admission

**Files:**
- Create: `src/pipeline/candidate_control.py`
- Create: `tests/pipeline/test_candidate_control.py`

**Interfaces:**
- Consumes: `EventRef`, `MultiScaleCandidate`, `EventMetrics`, `compute_event_metrics` from `src.pipeline.event_stack`.
- Produces: `CandidateAdmissionConfig`, `CandidateAdmissionSelection`, `suppress_overlapping_candidates`, `admit_candidates`, and `select_candidate_admission`.

- [ ] **Step 1: Write failing configuration and NMS tests**

```python
def multi(sid, start, end, source):
    evidence = CandidateEvent(EventRef(sid, start, end), (0.8,), 1.0, 0, 0, 1, 1)
    return MultiScaleCandidate(
        EventRef(sid, start, end),
        evidence if source in {"macro", "both"} else None,
        evidence if source in {"micro", "both"} else None,
    )

def geometry(candidates):
    return tuple((row.event.sid, row.event.start_ms, row.event.end_ms) for row in candidates)

def test_same_session_nms_is_score_first_and_input_order_independent():
    candidates = (multi("s1", 0, 100, "micro"), multi("s1", 10, 110, "macro"), multi("s2", 0, 100, "micro"))
    scores = np.array([0.8, 0.9, 0.7])
    expected = (("s1", 10, 110), ("s2", 0, 100))
    kept = suppress_overlapping_candidates(candidates, scores, 0.5)
    reversed_kept = suppress_overlapping_candidates(candidates[::-1], scores[::-1], 0.5)
    assert geometry(tuple(candidates[index] for index in kept)) == expected
    assert geometry(tuple(candidates[::-1][index] for index in reversed_kept)) == expected

@pytest.mark.parametrize("kwargs", [
    {"nms_iou": -0.1}, {"nms_iou": 1.1}, {"threshold": float("nan")},
    {"max_candidates_per_subject": 0},
])
def test_admission_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        CandidateAdmissionConfig(**kwargs)
```

- [ ] **Step 2: Run the focused RED test**

Run: `D:/Anaconda3/envs/bme/python.exe -m pytest -p no:cacheprovider tests/pipeline/test_candidate_control.py -q`

Expected: collection fails because `src.pipeline.candidate_control` does not exist.

- [ ] **Step 3: Implement immutable contracts and stable NMS**

```python
@dataclass(frozen=True)
class CandidateAdmissionConfig:
    nms_iou: float = 0.5
    threshold: float = 0.5
    max_candidates_per_subject: int | None = None

@dataclass(frozen=True)
class CandidateAdmissionSelection:
    config: CandidateAdmissionConfig
    metrics: EventMetrics
    candidate_count: int
    candidate_recall: float

def _stream_key(candidate: CandidateEvent | None) -> tuple[object, ...]:
    if candidate is None:
        return (0,)
    return (
        1, candidate.event.sid, candidate.event.start_ms, candidate.event.end_ms,
        candidate.probabilities, candidate.observed_fraction,
        candidate.bridged_gap_count, candidate.bridged_gap_ms,
        candidate.pre_observed_count, candidate.post_observed_count,
    )

def _multiscale_key(candidate: MultiScaleCandidate) -> tuple[object, ...]:
    return (
        candidate.event.sid, candidate.event.start_ms, candidate.event.end_ms,
        _stream_key(candidate.macro), _stream_key(candidate.micro),
    )

def suppress_overlapping_candidates(
    candidates: Sequence[MultiScaleCandidate], scores: np.ndarray, iou_threshold: float,
) -> tuple[int, ...]:
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (len(candidates),) or not np.isfinite(values).all():
        raise ValueError("scores must be finite and align with candidates")
    ranked = sorted(
        range(len(candidates)),
        key=lambda index: (
            -values[index], _multiscale_key(candidates[index]),
        ),
    )
    kept: list[int] = []
    for index in ranked:
        event = candidates[index].event
        if all(
            event.sid != candidates[other].event.sid
            or event_iou(event.interval, candidates[other].event.interval) < iou_threshold
            for other in kept
        ):
            kept.append(index)
    return tuple(
        index
        for index in sorted(
            kept,
            key=lambda row: (
                candidates[row].event.sid,
                candidates[row].event.start_ms,
                candidates[row].event.end_ms,
            ),
        )
    )
```

Validate finite aligned scores, use same-session IoU only, sort by score then complete candidate evidence and geometry, and return original row indices in canonical `(sid,start,end)` order. `admit_candidates` also returns original row indices so callers slice candidates, features, labels, groups, and both model-score arrays with one shared index vector.

- [ ] **Step 4: Write failing subject-budget and selection tests**

```python
def test_admission_budget_is_per_subject_not_per_session():
    admitted = admit_candidates(candidates, scores, groups=np.array(["p1", "p1", "p2"]), config=CandidateAdmissionConfig(1.0, 0.0, 1))
    assert [candidates[index].event.sid for index in admitted] == ["s1", "s3"]

def test_selection_prefers_recall_floor_then_f1_then_lower_burden():
    selection = select_candidate_admission(candidates, scores, labels, truths, groups, configs, minimum_recall=0.88)
    assert selection.config == CandidateAdmissionConfig(nms_iou=0.5, threshold=0.4, max_candidates_per_subject=4)
```

- [ ] **Step 5: Implement admission and registered selection ranking**

`select_candidate_admission` must evaluate only supplied train candidates/truths. Rank feasible choices by `(event_f1, ppv, -candidate_count/truth_count, canonical_config_rank)`; if none meets recall 0.88, prepend candidate recall to the same rank. Do not inspect global manifests or validation data.

- [ ] **Step 6: Run focused and pipeline tests**

Run: `D:/Anaconda3/envs/bme/python.exe -m pytest -p no:cacheprovider tests/pipeline/test_candidate_control.py tests/pipeline/test_event_stack.py -q`

Expected: all pass without warnings.

- [ ] **Step 7: Commit**

```bash
git add src/pipeline/candidate_control.py tests/pipeline/test_candidate_control.py
git commit -m "feat: add deterministic candidate admission"
```

---

### Task 2: Versioned runner contracts for stacked admission

**Files:**
- Modify: `src/pipeline/runner.py`
- Modify: `scripts/crossfit_event_stack.py`
- Modify: `tests/pipeline/test_runner.py`

**Interfaces:**
- Consumes: Task 1 admission contracts.
- Produces: registered `RunConfig` grids, schema-5 cache identity, extended `FoldResult`, strict CLI parsers, and two verifier factories.

- [ ] **Step 1: Write failing config, hash, result round-trip, and CLI tests**

```python
def test_stacking_settings_change_schema5_cache_identity():
    base = RunConfig(outer_fold=0, candidate_control_enabled=True)
    assert cache_key(base) != cache_key(replace(base, verifier_blend_weight_grid=(0.0, 1.0)))
    assert cache_key(base) != cache_key(replace(base, admission_threshold_grid=(0.3, 0.5)))

def test_stacked_result_round_trip_preserves_admission_diagnostics():
    base = run_outer_fold(RunConfig(outer_fold=0), synthetic_runner_dataset())
    payload = fold_result_to_dict(replace(base, selected_blend_weight=0.25, raw_union_candidate_count=700, admitted_candidate_count=120))
    assert fold_result_to_dict(_fold_result_from_dict(payload)) == payload
```

Also test invalid/duplicate/nonfinite CLI grids and reject candidate control unless `--micro-enabled` is set.

- [ ] **Step 2: Run RED**

Run: `D:/Anaconda3/envs/bme/python.exe -m pytest -p no:cacheprovider tests/pipeline/test_runner.py -q`

Expected: failures for absent fields, parsers, schema and serialization.

- [ ] **Step 3: Add exact registered settings**

```python
RUNNER_SCHEMA_VERSION = 5
VERIFIER_LGBM_PARAMETERS = {
    "n_estimators": 200, "num_leaves": 15, "max_depth": 4,
    "min_child_samples": 40, "learning_rate": 0.03,
    "colsample_bytree": 0.8, "reg_lambda": 5.0,
    "class_weight": "balanced", "n_jobs": 1, "verbosity": -1,
}

candidate_control_enabled: bool = False
admission_nms_iou_grid: tuple[float, ...] = (0.3, 0.5, 0.7)
admission_threshold_grid: tuple[float, ...] = (0.2, 0.35, 0.5, 0.65)
admission_subject_cap_grid: tuple[int, ...] = (3, 4, 5, 6, 8)
verifier_blend_weight_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
admission_minimum_recall: float = 0.88
```

Add `selected_blend_weight`, selected admission values, `raw_union_candidate_count`, `admitted_candidate_count`, and logistic/LightGBM timing fields to `FoldResult`. Historical schema-4 JSON receives neutral defaults but never matches schema 5 cache identity.

- [ ] **Step 4: Add the LightGBM verifier factory and strict CLI propagation**

Implement `_verifier_lgbm_estimator(seed)` with median imputation and exact parameters. Add explicit CLI flags for enabling control and overriding registered grids; all parsers require finite unique canonical values.

- [ ] **Step 5: Run focused and full tests**

Run: `D:/Anaconda3/envs/bme/python.exe -m pytest -p no:cacheprovider tests/pipeline/test_runner.py -q`

Run: `D:/Anaconda3/envs/bme/python.exe -m pytest -p no:cacheprovider -q`

Expected: all pass without warnings and legacy config behavior remains unchanged.

- [ ] **Step 6: Commit**

```bash
git add src/pipeline/runner.py scripts/crossfit_event_stack.py tests/pipeline/test_runner.py
git commit -m "feat: register stacked candidate control"
```

---

### Task 3: Nested OOF verifier stacking and admission

**Files:**
- Modify: `src/pipeline/runner.py`
- Create: `tests/pipeline/test_stacked_runner.py`

**Interfaces:**
- Consumes: Task 1 control functions and Task 2 runner contracts.
- Produces: train-only selected blend/admission/event policy and untouched outer-fold metrics.

- [ ] **Step 1: Write the failing isolation and one-score-per-row tests**

```python
def test_both_verifiers_crossfit_each_candidate_once_without_subject_overlap(monkeypatch):
    real = runner.crossfit_predict_proba
    calls = []
    def recording_crossfit(*args, **kwargs):
        output = real(*args, **kwargs)
        calls.append(output)
        return output
    monkeypatch.setattr(runner, "crossfit_predict_proba", recording_crossfit)
    config = registered_stacking_config()
    run_outer_fold(config, multiscale_dataset())
    assert len(calls) == 4
    verifier_calls = calls[-2:]
    assert len(verifier_calls) == 2
    assert all(np.all(call.score_counts == 1) for call in verifier_calls)
    assert all(
        set(fold.train_groups).isdisjoint(fold.validation_groups)
        for call in verifier_calls for fold in call.folds
    )
```

`registered_stacking_config()` is a test-local constructor returning
`RunConfig(outer_fold=0, inner_splits=3, micro_enabled=True, candidate_control_enabled=True)`;
`multiscale_dataset()` is imported from `tests.pipeline.test_multiscale_runner`. The registered
runner order is macro OOF, micro OOF, logistic-verifier OOF, then LightGBM-verifier OOF, so the last
two recorded calls are the verifier calls; also assert `len(calls) == 4` to make order drift fail.

- [ ] **Step 2: Run RED**

Run: `D:/Anaconda3/envs/bme/python.exe -m pytest -p no:cacheprovider tests/pipeline/test_stacked_runner.py -q`

Expected: runner lacks stacked execution and diagnostics.

- [ ] **Step 3: Produce aligned two-model OOF predictions**

Call `crossfit_predict_proba` twice with identical candidate features, labels, groups and splits, using seeds `config.seed+20` and `config.seed+21`. Assert both probability arrays are finite, aligned, and scored once.

- [ ] **Step 4: Write the failing outer-label-independence test**

```python
def test_outer_labels_cannot_change_stacking_or_admission_settings():
    original = run_outer_fold(config, dataset)
    changed = run_outer_fold(config, replace(dataset, validation_truths=(), validation_truth_slices={}, validation=replace(dataset.validation, labels=1-dataset.validation.labels)))
    assert selected_tuple(changed) == selected_tuple(original)
```

The selected tuple contains blend weight, NMS IoU, admission threshold/cap, verifier C, final event threshold and final event cap.

- [ ] **Step 5: Implement train-only joint selection**

For every registered blend weight, blend aligned OOF probabilities, enumerate Task 1 admission configs, admit train candidates, then call existing `select_event_policy` on admitted train events only. Select by `(event_f1, ppv, -admitted_count, -model_complexity_rank, canonical_settings)` while enforcing the admission recall rule. Store every selected value in `FoldResult`.

- [ ] **Step 6: Fit both final verifiers and score outer validation once**

Fit both estimators on all outer-train candidates. Produce exactly one outer probability vector per estimator, blend with the frozen weight, apply the frozen admission config without outer labels, and then apply the frozen event policy. Record raw union and admitted counts separately.

- [ ] **Step 7: Add legacy equivalence and timing tests**

Candidate control disabled must reproduce the current `0fe59d5/87748a9` multiscale synthetic outputs field-for-field. Enabled timing must include logistic OOF, LightGBM OOF, admission selection, final fit, and outer inference with nonnegative finite values.

- [ ] **Step 8: Run focused and full tests**

Run: `D:/Anaconda3/envs/bme/python.exe -m pytest -p no:cacheprovider tests/pipeline/test_stacked_runner.py tests/pipeline/test_multiscale_runner.py -q`

Run: `D:/Anaconda3/envs/bme/python.exe -m pytest -p no:cacheprovider -q`

Expected: all pass without warning.

- [ ] **Step 9: Commit**

```bash
git add src/pipeline/runner.py tests/pipeline/test_stacked_runner.py
git commit -m "feat: run nested verifier stacking"
```

---

### Task 4: Atomic model artifact contract

**Files:**
- Create: `src/pipeline/artifacts.py`
- Create: `scripts/promote_event_stack.py`
- Create: `tests/pipeline/test_artifacts.py`
- Modify: `src/pipeline/runner.py`

**Interfaces:**
- Consumes: fitted macro/micro/logistic/LightGBM models and selected policy from Task 3.
- Produces: `EventStackBundle`, atomic `write_event_stack_bundle`, verified `load_event_stack_bundle`, SHA-256 manifest, and a registered promotion CLI.

- [ ] **Step 1: Write failing artifact round-trip and checksum tests**

```python
def fitted_tiny_bundle():
    model = DummyClassifier(strategy="prior").fit(np.array([[0.0], [1.0]]), np.array([0, 1]))
    return EventStackBundle(
        models={name: model for name in ("macro", "micro", "verifier_logistic", "verifier_lgbm")},
        policy={"blend_weight": 0.25, "admission_threshold": 0.5},
        run_config={"outer_fold": 0, "candidate_control_enabled": True},
        feature_schema={"macro": 63, "micro": 47, "verifier": 56},
        metrics={"n_tp": 1, "n_true": 1, "n_pred": 1, "f1": 1.0},
        source_fingerprints=({"path": "fixture.npz", "size": 1, "mtime_ns": 1},),
        role="outer-fold-evidence",
    )

def test_bundle_round_trip_preserves_predictions_and_manifest(tmp_path):
    bundle = fitted_tiny_bundle()
    destination = tmp_path / "models" / "event_stack" / "run-key"
    write_event_stack_bundle(destination, bundle)
    loaded = load_event_stack_bundle(destination)
    probe = np.array([[0.25], [0.75]])
    for name in bundle.models:
        assert np.array_equal(
            loaded.models[name].predict_proba(probe),
            bundle.models[name].predict_proba(probe),
        )
    assert verify_bundle_manifest(destination) == ()

def test_tampered_model_is_rejected(tmp_path):
    destination = tmp_path / "models" / "event_stack" / "run-key"
    write_event_stack_bundle(destination, fitted_tiny_bundle())
    (destination / "verifier_lgbm.joblib").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256"):
        load_event_stack_bundle(destination)
```

- [ ] **Step 2: Run RED**

Run: `D:/Anaconda3/envs/bme/python.exe -m pytest -p no:cacheprovider tests/pipeline/test_artifacts.py -q`

Expected: module and APIs absent.

- [ ] **Step 3: Implement manifest and atomic directory promotion**

```python
@dataclass(frozen=True)
class EventStackBundle:
    models: Mapping[str, object]
    policy: Mapping[str, object]
    run_config: Mapping[str, object]
    feature_schema: Mapping[str, int]
    metrics: Mapping[str, object]
    source_fingerprints: tuple[Mapping[str, object], ...]
    role: str
```

Serialize models with joblib into a sibling unique temporary directory. Write `policy.json`, `run_config.json`, `feature_schema.json`, and `manifest.json` last. Manifest includes Git SHA, run/experiment key, source-input fingerprints, dependency versions, metrics, development-evidence warning and SHA-256 for every file. Verify before renaming; if destination exists, move it to a same-parent backup, rename the complete temp directory, then remove the verified backup only after success. On failure restore the backup.

- [ ] **Step 4: Add failure-recovery and stale-temp tests**

Inject a rename failure and assert the prior destination remains byte-identical. Assert cleanup targets only exact same-parent directories matching the run key and never follows symlinks outside `models/event_stack`.

- [ ] **Step 5: Expose final-fit artifacts without changing evaluation**

Refactor runner training into an internal result carrying models and policy; `run_outer_fold` still returns only `FoldResult`. The promotion CLI retrains registered five folds and one full-target deployment bundle using frozen selected settings; full-target models are labeled `deployment` and cannot be loaded as outer-fold evidence.

- [ ] **Step 6: Run tests and commit**

Run: `D:/Anaconda3/envs/bme/python.exe -m pytest -p no:cacheprovider tests/pipeline/test_artifacts.py tests/pipeline/test_stacked_runner.py -q`

Run: `D:/Anaconda3/envs/bme/python.exe -m pytest -p no:cacheprovider -q`

```bash
git add src/pipeline/artifacts.py scripts/promote_event_stack.py tests/pipeline/test_artifacts.py src/pipeline/runner.py
git commit -m "feat: persist verified event stack bundles"
```

---

### Task 5: Deployable `dist/event_stack` package

**Files:**
- Create: `scripts/package_event_stack.py`
- Create: `scripts/predict_event_stack.py`
- Create: `tests/pipeline/test_event_stack_dist.py`
- Modify: `dist/README.md`
- Retain atomically: `dist/event_stack/`

**Interfaces:**
- Consumes: Task 4 deployment bundle.
- Produces: standalone CPU inference package and repository/dist prediction parity.

- [ ] **Step 1: Write failing package-content and parity tests**

```python
def test_dist_package_is_complete_and_matches_repository_prediction(tmp_path):
    dist = package_fixture_bundle(tmp_path)
    assert required_paths(dist) == EXPECTED_EVENT_STACK_PATHS
    assert run_dist_prediction(dist, fixture_session()) == run_repo_prediction(fixture_session())

def test_failed_package_build_keeps_previous_dist(tmp_path, monkeypatch):
    previous = snapshot_dist(tmp_path)
    monkeypatch.setattr(package_module, "verify_packaged_bundle", raise_error)
    with pytest.raises(RuntimeError):
        package_event_stack(bundle_path=bundle_path, destination=dist_path)
    assert snapshot_dist(tmp_path) == previous
```

- [ ] **Step 2: Run RED**

Run: `D:/Anaconda3/envs/bme/python.exe -m pytest -p no:cacheprovider tests/pipeline/test_event_stack_dist.py -q`

- [ ] **Step 3: Implement one registered atomic packager**

Package only runtime modules, `predict_event_stack.py`, requirements, bundle files and manifest into a sibling temp directory. Validate imports in an isolated subprocess, verify checksums and fixture parity, then atomically replace only `dist/event_stack`. Never recreate or delete the unrelated legacy files elsewhere in `dist/`.

- [ ] **Step 4: Implement CPU inference and schema rejection**

The entrypoint accepts raw session directories, uses the bundle feature schema, outputs canonical JSON events, refuses incompatible schema/model hashes, and has no training-data or CUDA dependency.

- [ ] **Step 5: Run focused/full tests and commit**

Run: `D:/Anaconda3/envs/bme/python.exe -m pytest -p no:cacheprovider tests/pipeline/test_event_stack_dist.py -q`

Run: `D:/Anaconda3/envs/bme/python.exe -m pytest -p no:cacheprovider -q`

```bash
git add scripts/package_event_stack.py scripts/predict_event_stack.py tests/pipeline/test_event_stack_dist.py dist/README.md
git commit -m "feat: package event stack inference"
```

---

### Task 6: Registered five-fold experiment, promotion gate, documentation, and cleanup

**Files:**
- Modify: `README.md`
- Modify: `docs/三阶段重构设计.md`
- Modify when F1 improves: `dist/README.md`
- Retain when F1 improves: `models/event_stack/<run_key>/`, `dist/event_stack/`
- Retain: compact `outputs/crossfit/*.json`

**Interfaces:**
- Consumes: registered CLI from Tasks 2-3 and promotion/package CLIs from Tasks 4-5.
- Produces: one locked target-domain result, gate decision, promoted model/dist when improved, current architecture documentation, and a clean tree.

- [ ] **Step 1: Verify the implementation baseline**

Run: `D:/Anaconda3/envs/bme/python.exe -m pytest -p no:cacheprovider -q`

Run: `git status --short`

Expected: all tests pass without warnings; no paths are reported except the known inaccessible `.pytest_cache` warning.

- [ ] **Step 2: Run fold 0 runtime/geometry diagnostic only**

Run the registered CLI with `--fold 0 --inner-splits 4 --no-tcn --workers 1 --micro-enabled --candidate-control-enabled --force`. Verify finite 63/47/56 dimensions, disjoint subject sets, selected settings from their registered grids, and no outer-label-dependent selection. Do not alter grids afterward.

- [ ] **Step 3: Run the unchanged five-fold configuration**

Run: `D:/Anaconda3/envs/bme/python.exe scripts/crossfit_event_stack.py --fold all --inner-splits 4 --no-tcn --workers 0 --micro-enabled --candidate-control-enabled --force`

Record exact per-fold and aggregate TP/true/pred, F1, PPV, sensitivity, short recall, raw/admitted candidate counts, selected blend/admission policies and stage timings.

- [ ] **Step 4: Apply the pre-registered comparison**

Improvement means aggregate F1 strictly greater than `0.47863247863247865`. Recommended-default acceptance additionally requires PPV `>=0.35555555555555557`, short recall `>=20/39`, admitted candidates `<=612`, and five-fold wall clock `<=600s` excluding one-time feature extraction.

- [ ] **Step 5: Promote every F1 improvement**

If F1 improves by any amount, run `scripts/promote_event_stack.py` for the exact experiment key, then `scripts/package_event_stack.py`. Verify checksums, repository/dist fixture parity and isolated CPU inference. If F1 does not improve, do not replace the current model or `dist/event_stack`; retain only compact JSON evidence.

- [ ] **Step 6: Update all documentation surfaces**

Update README and architecture documentation with the exact result, comparison, candidate burden, selection protocol, development-evidence warning, artifact run key, dist reproduction command and current recommendation. If promoted, update `dist/README.md` in the same commit and ensure it describes the actual packaged architecture rather than legacy models.

- [ ] **Step 7: Audit and clean generated storage**

Measure retained production micro cache, model bundle, dist bundle and compact JSON sizes. Resolve every deletion target to an exact absolute path under its intended parent before removing only smoke/temp/failed model exports, bytecode, and superseded non-best experimental bundles. Preserve raw data, `cache/sessions`, all 20 current production micro NPZs, the accepted model, current dist and compact evidence.

- [ ] **Step 8: Final verification and commit**

Run full tests, `git diff --check`, artifact checksum verification and isolated dist smoke. Stage only the coherent model/dist/docs changes. Commit `feat: promote stacked event detector` if improved, otherwise `docs: record candidate control ablation`. Confirm `git status --short` reports no project paths.

---

## Follow-up Boundary

Only after Task 6 establishes the new target-domain baseline may a separate spec introduce FD-I/FD-II self-supervised pretraining. That work must retain random-initialization and `external_weight=0` controls and must not be folded into this plan.
