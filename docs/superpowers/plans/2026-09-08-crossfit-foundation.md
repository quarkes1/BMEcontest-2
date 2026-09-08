# Cross-Fit Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Build a subject-disjoint nested cross-fit evaluation foundation with frozen event thresholds, coverage-aware candidates, compact diagnostics, and bounded CPU execution, establishing the trustworthy baseline on the path to locked outer-CV F1 >= 0.65.

**Architecture:** Pure event and cross-fit behavior lives in src/pipeline/; one CLI loads existing slide caches and writes compact fold JSON. Inner group folds generate OOF window and verifier scores for threshold selection, while the untouched outer fold provides the only reported evaluation. Candidate modes share one explicit configuration so training and validation distributions cannot silently diverge.

**Tech Stack:** Python 3.11, NumPy, pandas, SciPy, scikit-learn, pytest, and threadpoolctl through scikit-learn. Optional PyTorch CUDA is reserved for later sequence-model plans.

**Spec:** docs/superpowers/specs/2026-09-08-crossfit-event-stack-design.md

## Global Constraints

- Final research acceptance requires aggregate event F1 >= 0.65 on subject-disjoint outer CV with each threshold frozen from inner OOF predictions.
- No outer-validation subject may influence a window model, verifier, threshold, feature selection, or variant selection.
- Foundation runs default to CPU/no-TCN because current target-trained TCN caches are not inner-fold cross-fitted.
- Reusable logic belongs in src/pipeline/; scripts/ receives one CLI; tests belong in tests/.
- Generated arrays live under cache/crossfit/ and compact metrics under outputs/crossfit/.
- Parallel fold workers limit native BLAS/OpenMP pools to one thread.
- Every production behavior is introduced by a failing test and the smallest passing implementation.
- Existing dirty files are preserved and never staged with these task commits.

---

### Task 1: Official event threshold selection

**Files:**
- Create: src/pipeline/__init__.py
- Create: src/pipeline/event_stack.py
- Create: tests/pipeline/test_event_stack.py

**Interfaces:**
- Consumes: src.eval.metrics.event_iou
- Produces: EventRef, EventMetrics, ThresholdSelection, compute_event_metrics(), select_event_threshold()

- [ ] **Step 1: Write failing tests for one-to-one matching and threshold tie-breaking**

    import numpy as np
    from src.pipeline.event_stack import EventRef, compute_event_metrics, select_event_threshold

    def test_compute_event_metrics_matches_once_per_truth():
        truths = [EventRef("s1", 100, 200)]
        predictions = [EventRef("s1", 100, 200), EventRef("s1", 110, 190)]
        metrics = compute_event_metrics(predictions, truths, iou_threshold=0.25)
        assert (metrics.n_tp, metrics.n_pred, metrics.n_true) == (1, 2, 1)
        assert metrics.f1 == 2 / 3

    def test_select_event_threshold_prefers_stricter_equal_f1_choice():
        candidates = [
            EventRef("s1", 100, 200),
            EventRef("s2", 100, 200),
            EventRef("s3", 100, 200),
            EventRef("s4", 100, 200),
        ]
        truths = [EventRef("s1", 100, 200), EventRef("s2", 100, 200)]
        selected = select_event_threshold(
            candidates, np.array([0.9, 0.7, 0.7, 0.7]), truths
        )
        assert selected.threshold == 0.9
        assert selected.metrics.n_tp == selected.metrics.n_pred == 1

- [ ] **Step 2: Run the focused tests and verify RED**

Run: D:/Anaconda3/envs/bme/python.exe -m pytest tests/pipeline/test_event_stack.py -v

Expected: collection fails with ModuleNotFoundError for src.pipeline.

- [ ] **Step 3: Implement immutable event types and metrics**

    @dataclass(frozen=True)
    class EventRef:
        sid: str
        start_ms: int
        end_ms: int

    @dataclass(frozen=True)
    class EventMetrics:
        n_tp: int
        n_pred: int
        n_true: int
        sensitivity: float
        ppv: float
        f1: float

    def compute_event_metrics(
        predictions: Sequence[EventRef],
        truths: Sequence[EventRef],
        iou_threshold: float = 0.25,
    ) -> EventMetrics:
        # Form same-session IoU pairs, sort descending, greedily consume each
        # prediction and truth once, and compute global ratios.

- [ ] **Step 4: Implement event-level threshold selection**

    @dataclass(frozen=True)
    class ThresholdSelection:
        threshold: float
        metrics: EventMetrics

    def select_event_threshold(candidates, scores, truths, iou_threshold=0.25):
        thresholds = np.unique(np.concatenate(
            (scores, [0.0, 1.0, np.nextafter(1.0, 2.0)])
        ))
        # Rank choices by (event_f1, ppv, threshold).

- [ ] **Step 5: Run focused and full tests**

Run: D:/Anaconda3/envs/bme/python.exe -m pytest tests/pipeline/test_event_stack.py -v

Expected: 2 tests pass.

Run: D:/Anaconda3/envs/bme/python.exe -m pytest -q

Expected: all tests pass.

- [ ] **Step 6: Commit Task 1**

    git add -- src/pipeline/__init__.py src/pipeline/event_stack.py tests/pipeline/test_event_stack.py
    git commit -m "feat: add official event threshold selection"

### Task 2: Density candidates and coverage metadata

**Files:**
- Modify: src/pipeline/event_stack.py
- Modify: tests/pipeline/test_event_stack.py

**Interfaces:**
- Consumes: EventRef and per-session windows shaped (start_ms, end_ms, probability)
- Produces: DensityConfig, CandidateEvent, density_candidates(), verifier_features()

- [ ] **Step 1: Add failing density and coverage tests**

    from src.pipeline.event_stack import DensityConfig, density_candidates

    def test_density_boundaries_use_threshold_support():
        windows = {
            "s1": [(i * 15_000, i * 15_000 + 240_000,
                    0.8 if 10 <= i <= 19 else 0.1) for i in range(40)]
        }
        result = density_candidates(
            windows, DensityConfig(density_ms=600_000, min_positive=10)
        )
        # Characterized against scripts/slide_verifier.py legacy semantics.
        assert [(c.event.start_ms, c.event.end_ms) for c in result] == [
            (180_000, 525_000)
        ]

    def test_coverage_fix_does_not_zero_pad_segment_edges():
        windows = {
            "s1": [(i * 15_000, i * 15_000 + 240_000, 0.8) for i in range(10)]
        }
        assert density_candidates(windows, DensityConfig(coverage_fix=False)) == []
        fixed = density_candidates(windows, DensityConfig(coverage_fix=True))
        assert len(fixed) == 1
        assert fixed[0].observed_fraction == 1.0

    def test_coverage_metadata_counts_internal_gap():
        starts = [0, 15_000, 30_000, 60_000, 75_000, 90_000,
                  105_000, 120_000, 135_000, 150_000]
        result = density_candidates(
            {"s1": [(s, s + 240_000, 0.8) for s in starts]},
            DensityConfig(coverage_fix=True, min_positive=10),
        )
        assert result[0].bridged_gap_count == 1
        assert result[0].bridged_gap_ms == 15_000

    def test_verifier_feature_dimensions_are_explicit():
        candidates = density_candidates(sample_windows, DensityConfig())
        assert verifier_features(candidates, sample_windows, include_coverage=False).shape[1] == 37
        assert verifier_features(candidates, sample_windows, include_coverage=True).shape[1] == 42

- [ ] **Step 2: Run new tests and verify RED**

Run: D:/Anaconda3/envs/bme/python.exe -m pytest tests/pipeline/test_event_stack.py -k "density or coverage" -v

Expected: import fails because DensityConfig and density_candidates do not exist.

- [ ] **Step 3: Implement explicit candidate types**

    @dataclass(frozen=True)
    class DensityConfig:
        stride_ms: int = 15_000
        window_ms: int = 240_000
        bridge_ms: int = 60_000
        density_ms: int = 600_000
        min_positive: int = 10
        coverage_min: float = 0.80
        merge_ms: int = 120_000
        window_threshold: float = 0.28838
        context_ms: int = 1_200_000
        coverage_fix: bool = False

    @dataclass(frozen=True)
    class CandidateEvent:
        event: EventRef
        probabilities: tuple[float, ...]
        observed_fraction: float
        bridged_gap_count: int
        bridged_gap_ms: int
        pre_observed_count: int
        post_observed_count: int

- [ ] **Step 4: Implement legacy parity and corrected timestamp-local coverage**

    def density_candidates(windows_by_sid, config):
        # Split gaps larger than stride+bridge. Insert missing slots only for
        # smaller bridged gaps. In corrected mode use the observed local
        # timestamp neighbourhood as denominator. Merge nearby output events.

    def verifier_features(candidates, windows_by_sid, include_coverage=False,
                          tcn_scores_by_sid=None):
        # Reproduce the existing 37 features with explicit inputs. Append the
        # five CandidateEvent coverage fields only when include_coverage=True.

- [ ] **Step 5: Run focused and full tests**

Run: D:/Anaconda3/envs/bme/python.exe -m pytest tests/pipeline/test_event_stack.py -v

Expected: all event-stack tests pass.

Run: D:/Anaconda3/envs/bme/python.exe -m pytest -q

Expected: all tests pass.

- [ ] **Step 6: Commit Task 2**

    git add -- src/pipeline/event_stack.py tests/pipeline/test_event_stack.py
    git commit -m "feat: add coverage-aware density candidates"

### Task 3: Subject-disjoint cross-fit primitives

**Files:**
- Create: src/pipeline/crossfit.py
- Create: tests/pipeline/test_crossfit.py

**Interfaces:**
- Consumes: NumPy arrays, subject groups, estimator factories with fit() and predict_proba()
- Produces: GroupFold, OOFProbabilities, make_group_folds(), crossfit_predict_proba()

- [ ] **Step 1: Write failing group-exclusion and score-coverage tests**

    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from src.pipeline.crossfit import crossfit_predict_proba, make_group_folds

    def test_group_folds_never_split_subjects():
        groups = np.array(["a", "a", "b", "b", "c", "c", "d", "d"])
        for fold in make_group_folds(groups, n_splits=2):
            assert set(groups[fold.train_rows]).isdisjoint(
                set(groups[fold.validation_rows])
            )

    def test_crossfit_scores_every_requested_row_once():
        x = np.arange(16, dtype=float).reshape(8, 2)
        y = np.array([0, 0, 0, 1, 0, 1, 1, 1])
        groups = np.array(["a", "a", "b", "b", "c", "c", "d", "d"])
        result = crossfit_predict_proba(
            x, y, groups, x, groups, n_splits=2,
            estimator_factory=lambda: LogisticRegression(random_state=42),
        )
        assert result.probabilities.shape == (8,)
        assert np.isfinite(result.probabilities).all()
        assert (result.score_counts == 1).all()

- [ ] **Step 2: Run focused tests and verify RED**

Run: D:/Anaconda3/envs/bme/python.exe -m pytest tests/pipeline/test_crossfit.py -v

Expected: collection fails because src.pipeline.crossfit does not exist.

- [ ] **Step 3: Implement deterministic group fold manifests**

    @dataclass(frozen=True)
    class GroupFold:
        index: int
        train_rows: np.ndarray
        validation_rows: np.ndarray
        train_groups: tuple[str, ...]
        validation_groups: tuple[str, ...]

    def make_group_folds(groups, n_splits):
        # Validate enough unique groups, call GroupKFold, and assert disjoint sets.

- [ ] **Step 4: Implement generic cross-fitted probabilities**

    @dataclass(frozen=True)
    class OOFProbabilities:
        probabilities: np.ndarray
        score_counts: np.ndarray
        folds: tuple[GroupFold, ...]

    def crossfit_predict_proba(
        train_features, train_labels, train_groups,
        score_features, score_groups, n_splits, estimator_factory,
    ):
        # Each inner model excludes held-out score groups from fitting and
        # scores all rows belonging to those groups exactly once.

- [ ] **Step 5: Add failing-then-passing validation for unknown score groups**

    def test_crossfit_rejects_unknown_score_group():
        with pytest.raises(ValueError, match="score groups absent from training groups"):
            crossfit_predict_proba(
                np.ones((4, 1)), np.array([0, 1, 0, 1]),
                np.array(["a", "a", "b", "b"]),
                np.ones((1, 1)), np.array(["missing"]), 2,
                lambda: LogisticRegression(),
            )

- [ ] **Step 6: Run focused and full tests**

Run: D:/Anaconda3/envs/bme/python.exe -m pytest tests/pipeline/test_crossfit.py -v

Expected: all cross-fit tests pass.

Run: D:/Anaconda3/envs/bme/python.exe -m pytest -q

Expected: all tests pass.

- [ ] **Step 7: Commit Task 3**

    git add -- src/pipeline/crossfit.py tests/pipeline/test_crossfit.py
    git commit -m "feat: add subject-disjoint crossfit primitives"

### Task 4: Nested runner, cache contract, and CLI

**Files:**
- Create: src/pipeline/runner.py
- Create: tests/pipeline/test_runner.py
- Create: scripts/crossfit_event_stack.py

**Interfaces:**
- Consumes: fold train/meal_train/no_meal_train/val NPZ caches and session-to-subject metadata
- Produces: RunConfig, FoldResult, run_outer_fold(), run_folds(), content-addressed caches, compact fold JSON

- [ ] **Step 1: Write failing cache-key and leakage-guard tests**

    from src.pipeline.event_stack import DensityConfig
    from src.pipeline.runner import RunConfig, cache_key, validate_outer_isolation

    def test_cache_key_changes_with_candidate_semantics():
        base = RunConfig(
            outer_fold=0, inner_splits=3,
            density=DensityConfig(coverage_fix=False),
        )
        fixed = RunConfig(
            outer_fold=0, inner_splits=3,
            density=DensityConfig(coverage_fix=True),
        )
        assert cache_key(base) != cache_key(fixed)

    def test_validate_outer_isolation_rejects_overlap():
        with pytest.raises(ValueError, match="outer subject leakage"):
            validate_outer_isolation({"a", "b"}, {"b", "c"})

- [ ] **Step 2: Run runner tests and verify RED**

Run: D:/Anaconda3/envs/bme/python.exe -m pytest tests/pipeline/test_runner.py -v

Expected: collection fails because src.pipeline.runner does not exist.

- [ ] **Step 3: Implement explicit run and result types**

    @dataclass(frozen=True)
    class RunConfig:
        outer_fold: int
        inner_splits: int = 4
        seed: int = 20260908
        no_tcn: bool = True
        workers: int = 1
        device: str = "auto"
        density: DensityConfig = field(default_factory=DensityConfig)

    @dataclass(frozen=True)
    class FoldResult:
        config_hash: str
        threshold: float
        inner_metrics: EventMetrics
        outer_metrics: EventMetrics
        candidate_count: int
        candidate_match_recall: float
        slices: Mapping[str, EventMetrics]
        timings_seconds: Mapping[str, float]
        cache_hits: Mapping[str, bool]
        outer_subjects: frozenset[str]
        window_fit_subjects: frozenset[str]
        verifier_fit_subjects: frozenset[str]

- [ ] **Step 4: Implement content-addressed cache keys**

cache_key() serializes RunConfig, feature dimensions, model parameters, seed, and
input cache file size/mtime into sorted canonical JSON, then keeps the first 16
hexadecimal characters of SHA-256. Cache writes use a temporary sibling file and
atomic Path.replace() so interrupted runs do not leave a valid-looking cache.

- [ ] **Step 5: Implement the nested no-TCN CPU runner**

    def run_outer_fold(config, data_source=None):
        # Load arrays once and map sid to externalid.
        # Cross-fit HGB probabilities for every outer-train meal/no-meal row.
        # Generate OOF candidates and matching verifier features.
        # Cross-fit the L2 verifier by subject and select one event threshold.
        # Fit final outer-train HGB/verifier and score untouched outer val.
        # Return global metrics, slices, timings, cache hits, and audit subjects.

Use current HGB parameters: learning_rate=0.05, max_iter=150,
max_leaf_nodes=15, max_depth=4, min_samples_leaf=100,
l2_regularization=1.0, early_stopping=False. Use the current verifier pipeline
SimpleImputer -> StandardScaler -> LogisticRegression with C=0.1,
class_weight="balanced", and max_iter=3000.

- [ ] **Step 6: Implement bounded parallel execution**

    def run_folds(configs, workers):
        # workers==1 runs inline. workers>1 uses ProcessPoolExecutor and wraps
        # each fold in threadpoolctl.threadpool_limits(limits=1).

The CLI exposes --fold 0|1|2|3|4|all, --inner-splits, --coverage-fix,
--no-tcn, --workers, --device auto|cpu|cuda, and --force. workers=0 resolves
to min(physical CPU count, requested fold count). In this CPU foundation an
explicit --device cuda exits with a clear message; auto and cpu are accepted.

- [ ] **Step 7: Add a synthetic end-to-end leakage test**

    def test_outer_subjects_never_enter_fit_sets(tmp_path):
        dataset = synthetic_runner_dataset(tmp_path, subjects=8, windows_per_subject=40)
        result = run_outer_fold(
            RunConfig(outer_fold=0, inner_splits=3), data_source=dataset
        )
        assert result.outer_subjects.isdisjoint(result.window_fit_subjects)
        assert result.outer_subjects.isdisjoint(result.verifier_fit_subjects)

- [ ] **Step 8: Run tests and CLI help**

Run: D:/Anaconda3/envs/bme/python.exe -m pytest -q

Expected: all tests pass.

Run: D:/Anaconda3/envs/bme/python.exe scripts/crossfit_event_stack.py --help

Expected: exits 0 and lists all registered arguments.

- [ ] **Step 9: Commit Task 4**

    git add -- src/pipeline/runner.py tests/pipeline/test_runner.py scripts/crossfit_event_stack.py
    git commit -m "feat: add nested event stack runner"

### Task 5: Baseline, coverage ablation, hygiene, and documentation

**Files:**
- Modify: .gitignore
- Modify: README.md
- Modify: docs/三阶段重构设计.md
- Modify: scripts/slide_verifier.py
- Create: tests/pipeline/test_slide_verifier_parity.py
- Generate but do not commit: outputs/crossfit/*.json

**Interfaces:**
- Consumes: Tasks 1-4 and current slide caches
- Produces: clean nested baseline, coverage-fix decision, legacy adapter parity, updated reproduction commands

- [ ] **Step 1: Write a failing legacy parity test**

    from scripts import slide_verifier
    from src.pipeline.event_stack import DensityConfig, density_candidates

    def test_slide_verifier_density_delegates_with_legacy_parity(sample_windows):
        old = slide_verifier.density_candidates(sample_windows, 0.28838)
        new = density_candidates(sample_windows, DensityConfig(coverage_fix=False))
        assert [(c[0], c[1], c[2]) for c in old] == [
            (c.event.sid, c.event.start_ms, c.event.end_ms) for c in new
        ]

- [ ] **Step 2: Run parity test and verify RED**

Run: D:/Anaconda3/envs/bme/python.exe -m pytest tests/pipeline/test_slide_verifier_parity.py -v

Expected: fails because slide_verifier has not delegated to the new routine.

- [ ] **Step 3: Delegate candidate and threshold behavior**

Retain the existing CLI/output schema. Replace the duplicated density body with
an adapter around src.pipeline.event_stack.density_candidates. Keep
BME_DENS_COVFIX only as a compatibility adapter that creates DensityConfig.
Replace validation threshold scanning with select_event_threshold and label its
result diagnostic_per_fold_optimum.

- [ ] **Step 4: Add exact artifact ignore rules**

Append:

    /cache/crossfit/
    /outputs/crossfit/
    /outputs/slide_cand_fold*.npz
    /outputs/slide_diag_fold*.json

Do not delete tracked historical experiment artifacts. Do not add another script.

- [ ] **Step 5: Run fold 0 twice and verify cache reuse**

Run twice:

    D:/Anaconda3/envs/bme/python.exe scripts/crossfit_event_stack.py --fold 0 --inner-splits 4 --no-tcn --workers 1

Expected: identical metrics/thresholds; the second run reports score cache hits
and lower wall-clock time.

- [ ] **Step 6: Run five-fold baseline and coverage ablation**

    D:/Anaconda3/envs/bme/python.exe scripts/crossfit_event_stack.py --fold all --inner-splits 4 --no-tcn --workers 0
    D:/Anaconda3/envs/bme/python.exe scripts/crossfit_event_stack.py --fold all --inner-splits 4 --no-tcn --workers 0 --coverage-fix

Expected: compact JSON under outputs/crossfit/, stage timings, cache status, and
outer F1 calculated only with inner-derived frozen thresholds.

- [ ] **Step 7: Apply the registered coverage decision rule**

Adopt coverage_fix=True only if candidate misses fall, aggregate PPV does not
fall, and aggregate outer F1 improves by at least 0.01. Otherwise retain legacy
behavior and mark the result rejected or inconclusive.

- [ ] **Step 8: Update README and architecture documentation**

README must record config hash, inner metrics, outer TP/eligible/pred/F1,
candidate recall, short-meal recall, non-dominant recall, runtime, and cache
status. docs/三阶段重构设计.md must record the nested data-flow contract,
coverage decision, and remaining gap to F1 0.65. Historical leakage values remain
only in explicitly labelled history.

- [ ] **Step 9: Run final verification**

Run: D:/Anaconda3/envs/bme/python.exe -m pytest -q

Expected: all tests pass.

Run: git diff --check

Expected: no whitespace errors in plan-owned files.

Run: git status --short

Expected: generated crossfit directories are ignored; only pre-existing user
changes and intended documentation changes remain.

- [ ] **Step 10: Commit Task 5**

    git add -- .gitignore README.md "docs/三阶段重构设计.md" scripts/slide_verifier.py tests/pipeline/test_slide_verifier_parity.py
    git commit -m "refactor: validate slide stack with nested crossfit"

## Plan self-review

- Spec coverage: Tasks 1-5 cover official event metrics, subject isolation,
  frozen thresholds, matching candidate modes, cache/timing controls, slice
  metrics, baseline reproduction, documentation, and folder hygiene.
- Scope boundary: short-meal modelling, reviewed hard negatives, target-trained
  TCN cross-fitting, and PPG remain separate plans because each changes the
  learned representation and needs the foundation delivered here.
- Type consistency: EventRef, EventMetrics, ThresholdSelection, DensityConfig,
  CandidateEvent, GroupFold, OOFProbabilities, RunConfig, and FoldResult are
  introduced once and consumed under the same names.
- Placeholder scan: each task names exact APIs, paths, commands, expected
  failures, model parameters, output locations, and decision rules.
