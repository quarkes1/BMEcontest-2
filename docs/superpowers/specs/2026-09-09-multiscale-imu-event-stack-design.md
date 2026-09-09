# Multi-scale IMU Event Stack Design

Date: 2026-09-09

## 1. Objective

Raise the leakage-safe, subject-disjoint locked aggregate event F1 from the current CPU
baseline of 0.416 toward the project target of at least 0.65. The first milestone adds a
15-second ACC+GYRO micro-motion branch without replacing the existing 240-second ACC
context branch. External data is permitted by the competition and enters only after the
target-domain micro branch proves useful.

Official scoring remains greedy one-to-one event matching at IoU≥0.25. No outer-validation
subject may influence feature selection, model fitting, score fusion, post-processing,
threshold selection, or variant selection.

“Locked” below means each run is nested and subject-disjoint: a fold's validation labels cannot
affect that fold's predictions. Repeatedly comparing project variants on the same outer folds can
still create evaluation-adaptive bias. Those aggregate results are therefore development CV; a
final F1≥0.65 claim requires one pre-registered frozen pipeline evaluated on untouched competition
labels or a separately reserved audit split.

## 2. Evidence and chosen approach

The current nested baseline misses 32 of 39 meals shorter than 10 minutes at final selection,
and its 240-second training window requires more than 120 seconds of meal overlap to become a
positive. The audited teammate pipeline uses 15-second ACC+GYRO windows and supplies two useful
ideas: local motion windows and gyroscope statistics. Its reported F1 is not adopted because
pure-negative sessions are absent and several hyperparameters are selected on pooled test OOF.

Three approaches were considered:

1. **Selected: multi-scale target-domain model, then gated external transfer.** Add a small
   15-second LightGBM branch, form permissive micro candidates, union them with macro candidates,
   and train one nested event verifier over both score streams. This directly targets candidate
   misses while retaining long context for false-positive control.
2. Add GYRO features only to 240-second windows. This is cheaper but does not remove the positive
   label geometry that suppresses short meals.
3. Start with external pretraining. This delays target-domain evidence and risks spending compute
   on a representation whose task or scale does not transfer.

## 3. Components and boundaries

### 3.1 Micro feature extraction

Create `src/pipeline/imu_features.py` as the reusable feature implementation. It exposes:

```python
@dataclass(frozen=True)
class MicroFeatureConfig:
    window_ms: int = 15_000
    stride_ms: int = 7_500
    coverage_min: float = 0.80
    gravity_align: bool = True

def gravity_rotation(gravity: np.ndarray) -> np.ndarray: ...

def extract_micro_features(
    acc: np.ndarray,
    gyro: np.ndarray,
    sample_rate_hz: float,
) -> np.ndarray: ...
```

`acc` and `gyro` are shaped `(3, n)`. The output is exactly 47 finite float32 values:

- ACC x/y/z: mean, std, min, max, RMS, skewness, excess kurtosis, zero-crossing rate
  (`3 × 8 = 24`);
- ACC signal magnitude area and pairwise axis correlations (`4`);
- ACC magnitude spectrum above 0.5Hz: dominant frequency, dominant-power ratio, spectral
  centroid, normalized entropy, and relative energy in 0.5–2, 2–5, 5–10Hz (`7`);
- GYRO x/y/z: mean, std, RMS, energy (`3 × 4 = 12`).

Undefined correlations, empty frequency bands, and near-constant channels map to zero. The
gravity rotation is computed once per session from the median valid ACC vector and applied to
both ACC and GYRO. The rotation-free variant uses the identity transform but the same feature
schema, enabling a controlled ablation.

Create `scripts/build_micro_features.py` only as a thin CLI. It reads `cache/sessions`, uses the
existing fold/session manifests, and writes NPZ files to `cache/micro15/` with the same
`feat`, `label`, and `wid` arrays as `cache/slide/`, plus a versioned `metadata` JSON scalar.
Supported splits are `train`, `meal_train`, `no_meal_train`, and `val`.

Labels are:

- positive when a 15-second window has at least 50% overlap with an outer-train meal;
- negative when it has zero overlap and is at least 300 seconds from every meal;
- ignored otherwise.

Validation and candidate-training caches retain all windows, including ignored labels. The
sampled training cache keeps every positive and at most three negatives per positive per
session, using the existing fixed seed. Pure-negative sessions contribute at least one sampled
negative to the window model and remain fully represented in `no_meal_train`.

Extraction uses `ProcessPoolExecutor`, defaults to at most eight processes, writes a temporary
file beside the destination, and atomically replaces the final NPZ. Cache metadata contains the
feature-config hash, source-session signatures, feature count, and extraction version. A cache
whose metadata does not match is rejected instead of silently reused.

### 3.2 Micro window model

Extend `src/pipeline/runner.py` with `micro_window_train`, `micro_candidate_train`, and
`micro_validation` `WindowBatch` fields in `FoldDataset`. The filesystem data source loads the
four cache files and combines micro `meal_train` plus `no_meal_train` into
`micro_candidate_train`, matching the existing macro data boundary. The micro model is
`lightgbm.LGBMClassifier` with fixed starting parameters:

```python
dict(
    n_estimators=300,
    num_leaves=31,
    min_child_samples=100,
    learning_rate=0.05,
    colsample_bytree=0.8,
    reg_lambda=5.0,
    class_weight="balanced",
    n_jobs=1,
    random_state=20260909,
    verbosity=-1,
)
```

Each outer fold generates micro scores for outer-train candidates through inner GroupKFold OOF.
The final micro model fits only outer-train subjects and scores the untouched outer-validation
subjects. The existing macro HGB branch is unchanged, so its cached baseline remains reproducible.

Outer folds may run in parallel. Native LightGBM threads remain one per fold worker to avoid
oversubscription. CUDA is reserved for a later neural external-data encoder; this tree-model
milestone stays CPU-first because its feature table is small and fold parallelism is faster to
iterate.

### 3.3 Micro candidate generation

Add pure event functions to `src/pipeline/event_stack.py`:

```python
@dataclass(frozen=True)
class MicroCandidateConfig:
    smooth_sigma_ms: int = 30_000
    smooth_radius_ms: int = 60_000
    merge_ms: int = 180_000
    min_duration_ms: int = 60_000

@dataclass(frozen=True)
class MultiScaleCandidate:
    event: EventRef
    macro: CandidateEvent | None
    micro: CandidateEvent | None

def micro_candidates(
    windows_by_sid: Mapping[str, Sequence[tuple[int, int, float]]],
    threshold: float,
    config: MicroCandidateConfig,
) -> list[CandidateEvent]: ...

def union_candidates(
    macro: Sequence[CandidateEvent],
    micro: Sequence[CandidateEvent],
    merge_iou: float = 0.25,
) -> list[MultiScaleCandidate]: ...
```

Micro windows are processed independently per sensor session. Scores are placed on the observed
7.5-second grid and smoothed with a normalized Gaussian (`sigma=30` seconds, truncated at
`±60` seconds). Gaps larger than two strides split the grid and are never filled or bridged by
smoothing. Runs above threshold are merged when their gap is at most 180 seconds and retained at
duration at least 60 seconds. Event boundaries use the first and last supporting micro windows.

The inner-OOF threshold grid is fixed before outer evaluation:
`(0.10, 0.20, 0.30, 0.40, 0.50)`. Selection maximizes candidate recall subject to no more than
`3 × max(number of eligible inner truths, 1)` pooled micro candidates; ties prefer fewer
candidates and then a higher threshold. If no threshold satisfies the budget, choose the threshold
with the highest candidate F1, then PPV, then threshold. A zero-truth inner pool therefore has an
explicit maximum of three diagnostic candidates rather than a division-by-zero special case.

Candidate union is session-local and one-to-one. Micro candidates are processed by descending
maximum window probability, then start time. When a micro candidate has IoU≥0.25 with one or more
unmatched macro candidates, it is paired with the highest-IoU macro candidate; ties use earlier
start time. The macro event geometry is preserved, while both `CandidateEvent` objects are
attached as evidence.
An unmatched micro candidate keeps its own geometry and can therefore rescue a macro miss;
unmatched macro candidates also remain. Preserving macro geometry avoids boundary expansion that
could lower official IoU. The result is sorted by `(sid, start_ms, end_ms)` and is deterministic
under input ordering.

### 3.4 Multi-scale verifier

Each union candidate receives the existing 37 macro probability/context features plus these
micro features:

- source flags: macro present, micro present;
- number of micro windows and candidate duration;
- micro mean, max, std, P10, median, P90;
- fraction at or above 0.30 and 0.50;
- longest run at or above 0.30;
- micro mean and max contrast against the 20-minute pre/post context;
- macro–micro mean and max differences, with missing-stream indicators.

The 37 macro features are recomputed against the macro score stream over every union event's
geometry, including micro-only events. When fewer than two macro windows support that geometry,
the macro block is zero-filled and a missing-macro indicator is set; rows are never silently
dropped. The micro block follows the same rule. Context features always use observed windows from
the same session and never bridge acquisition gaps. Source flags describe which generator emitted
the candidate; the two missing-stream indicators describe whether enough score samples existed,
so they are not redundant. The verifier width is exactly 56 (`37 + 19`).

The first implementation does not concatenate all 47 raw features at event level because the
previous 349/354-dimensional raw-summary experiment amplified cross-subject scale shift. The
event verifier remains imputer + scaler + balanced logistic regression. Its C value, micro
candidate threshold, optional per-subject event budget, and final verifier threshold are selected
only from inner OOF. The final outer score is reported once.

### 3.5 Deterministic positive-purity ablation

After the multi-scale baseline is complete, one registered variant trains the micro model using
only positive windows whose centers lie in the middle 60% of each outer-train meal. All outer-train
negative windows remain. This selection uses labels only, needs no learned OOF selector, and never
touches outer-validation labels. It is accepted only under the same locked gate as other variants.

## 4. External-data phase

Competition rules permit external data. Dataset-specific licenses and attribution still apply.
The first external experiment uses the already-present FD-I/FD-II files; no new download is needed.

Create an FD micro-cache adapter only after the target-domain multi-scale branch passes its gate.
It produces the same 47-feature schema from 15-second windows. FD-I supplies eating and confirmed
non-eating windows; FD-II supplies positive meal/intake windows only because its outside-meal zero
labels are not confirmed negatives. External subject identifiers are namespaced and can never be
confused with target subjects.

External samples are added only to inner/final training pools, never to target validation pools.
External sample weights `(0.0, 0.1, 0.25, 0.5)` are selected by target-domain inner OOF event F1;
`0.0` is mandatory so transfer can be rejected. No external dataset may change official target
truths, eligibility, or the outer folds.

WIMID and Clemson All-Day are deferred until their dataset licenses are confirmed. If approved,
WIMID is used for 100Hz micro-motion transfer and Clemson for all-day hard negatives and sequence
calibration. FIC/OREBA remain gesture-pretraining sources and do not calibrate event PPV.

## 5. Configuration and outputs

Extend `RunConfig` with explicit, hashed fields:

```python
micro_enabled: bool = False
micro_gravity_align: bool = True
micro_threshold_grid: tuple[float, ...] = (0.10, 0.20, 0.30, 0.40, 0.50)
micro_candidate: MicroCandidateConfig = field(default_factory=MicroCandidateConfig)
micro_positive_middle_fraction: float | None = None
external_fd_weight_grid: tuple[float, ...] = (0.0,)
```

`FoldResult` adds selected micro threshold, micro candidate count/recall, short-meal candidate
recall, external weight, feature counts, and separate extraction/window/verifier timings. The
runner schema version increments so stale fold results cannot be reused.

`scripts/crossfit_event_stack.py` exposes matching CLI flags and writes only compact JSON under
`outputs/crossfit/`. No experiment-specific runner script is created. README and the architecture
document record every five-fold result, including rejected variants.

## 6. Testing and leakage invariants

Unit tests cover:

- gravity alignment for parallel, anti-parallel, and zero gravity vectors;
- 47-feature shape, finiteness, constant-channel behavior, and dominant-frequency recovery;
- label boundaries at 50% overlap and 300-second negative buffer;
- session gaps never bridged during smoothing;
- micro run merging, minimum duration, deterministic union, and source flags;
- exact multi-scale verifier width and missing-stream indicators;
- cache metadata mismatch rejection;
- inner micro model scoring subjects are disjoint from its fit subjects;
- external subjects enter fit pools only and never outer validation;
- config hashes change for every new behavior-setting field.

Integration tests use synthetic `FoldDataset` instances and verify that outer subjects do not enter
macro fit, micro fit, verifier fit, fusion selection, or threshold selection. Existing macro-only
tests and cached results must remain unchanged when `micro_enabled=False`.

## 7. Experiment gates

The sequence is fixed:

1. Unit/integration tests and a limited-session cache smoke test.
2. Fold 0 micro-only diagnostic to validate runtime, positive coverage, and candidate geometry.
3. Five-fold target-domain multi-scale locked run.
4. Gravity-alignment ablation only if the multi-scale run is stable.
5. Middle-60% purity ablation only if micro candidate recall improves but verifier precision is weak.
6. Write a separate FD-transfer implementation plan only if the target-domain multi-scale run
   passes the acceptance gate; external-data code is not part of the first implementation plan.

Relative to the matching macro-only nested run, a variant is accepted only when:

- aggregate locked F1 improves by at least 0.02;
- short-meal final recall improves by at least 0.10;
- aggregate PPV falls by no more than 0.03;
- candidate count and runtime remain practical for full five-fold iteration.

The project target remains F1≥0.65. Passing an intermediate gate does not mean the project target
is complete; it only determines whether the branch becomes the new development default for the
next experiment. Gravity, positive-purity, and external-weight choices that survive diagnostics
must be moved into inner-OOF selection before the next locked run; an outer-fold comparison is
never used to choose a setting inside that same reported run.

## 8. Repository hygiene

Every successful algorithm increment ends with this exact lifecycle:

1. run focused tests, full pipeline tests, and the relevant locked experiment;
2. delete throwaway caches/logs and any one-off helper script;
3. retain reusable code only under `src/`, thin durable CLIs under `scripts/`, tests under `tests/`,
   compact results under the ignored `outputs/crossfit/`, and decisions under `docs/`/README;
4. stage and commit the coherent change immediately;
5. verify `git status --short` is empty.

Rejected experiments are documented and their reusable primitive may remain only when it has tests
and a clear interface. Otherwise the code and cache are removed in the same commit that records the
rejection.
