# Challenger E02 Context-v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Add the fixed 25-column session-local Context-v2 verifier block and evaluate it only through the preregistered E02 inner gate and, if passed, a single strict five-fold challenger evidence run.

**Architecture:** Context-v1 remains untouched. A pure Context-v2 builder appends 25 values to the existing 56 legacy plus 60 Context-v1 verifier features; the existing fold-contained verifier, imputer and policy machinery remains authoritative. A challenger-only E02 runner owns preregistration, isolated cache/output paths, inner-only gate, evidence accounting and registry append; it has no promotion path.

**Tech Stack:** Python 3.11, NumPy, scikit-learn, LightGBM, pytest, existing RunConfig/candidate-control/event matcher/atomic JSON helpers.

**Spec:** docs/superpowers/specs/2026-09-20-challenger-e02-context-v2-design.md

## Global Constraints

- Work only on research/f1-070-challenger. Do not alter main, release registry, incumbent models, dist, raw model features, candidate construction, matcher, dependencies, or incumbent cache contract.
- v1 is bit-for-bit immutable. None/v1/v2 verifier widths are exactly 56/116/141.
- Context-v2 uses precisely the fixed 25-name schema/constants in the spec. Never inspect E01 Context-v1 separation or E02 outer results to change features.
- Fits, imputers, policies and cache use only outer-train/inner-OOF data. `OuterTrainOnlyData` and `FilesystemDataSource.load_outer_train_only` are mandatory; this loader must not call/wrap load_outer_fold and inner-gate APIs cannot receive outer validation data.
- Write E02 only below cache/challenger_70/e02-context-v2 and outputs/challenger_70/experiments/E02 atomically. Cache identity is SHA-256 over literal `e02-context-v2`, full config, code SHA, NumPy version 2.4.6 and the 25-column schema hash.
- E02 uses NumPy 2.4.6 and `np.percentile(..., method="linear")`; formal run rejects a different version. The full command is exactly-once and has no force/resume/overwrite mode.
- Use TDD, a focused test cycle and git add/commit after every successful code task. Do not execute five-fold outer evaluation until plan/schema and Gate-0 tests exist.
- No release/promotion/package/dist command is authorized. A terminal result is reported, never promoted.

---

### Task 1: Pure Context-v2 schema and builder

**Files:**
- Modify: src/pipeline/context_features.py
- Test: tests/pipeline/test_context_features.py

**Interfaces:**

    CONTEXT_V2_COLUMNS: tuple[str, ...]
    CONTEXT_V2_SCHEMA_HASH: str

    def context_v2_features(candidates, macro_windows_by_sid,
                            micro_windows_by_sid, session_bounds_by_sid) -> np.ndarray: ...
    def context_schema_descriptor(version: str) -> dict[str, object]: ...

**Produces:** caller-order float64 (n,25) append block with only finite entries or NaN; versioned schema descriptor. The v2 verifier matrix is 141D because it includes the frozen 116D v1 surface.

- [ ] Step 1: Write failing tests asserting the 25 names/order/hash/descriptor metadata and exact 56/116/141 contracts.
- [ ] Step 2: Run RED: D:/Anaconda3/envs/bme/python.exe -m pytest tests/pipeline/test_context_features.py -k context_v2 -v
- [ ] Step 3: Write failing numerical tests from real CandidateEvent/MultiScaleCandidate fixtures: macro-only, micro-only and dual support; hand-computed midranks, robust-z, relative/max floors, tie/no-stronger behavior, 20-minute boundary and overlap distance.
- [ ] Step 4: Write failing safety tests for singleton/zero baseline/missing source NaNs/no infinity, permutation invariance, other-session non-interference, touching sessions, missing/out-of-range session bounds, unordered/nonfinite inputs and acquisition gaps. Add raw-union contract cases: zero-duration event, no source, all-nonfinite/empty source probability streams all fail; coincident intervals with distinct canonical source identities remain distinct; the candidate population is exactly the existing union_candidates output and is not deduplicated again.
- [ ] Step 5: Implement only the pure builder and descriptor. Validate each complete raw-union member exactly as specified, then canonical-sort a working session view but return rows in input order. Reuse validation helpers only if that leaves v1 behavior byte/value unchanged. Never access labels or verifier scores.
- [ ] Step 6: Run GREEN: D:/Anaconda3/envs/bme/python.exe -m pytest tests/pipeline/test_context_features.py tests/pipeline/test_event_stack.py -v
- [ ] Step 7: Commit:
    git add src/pipeline/context_features.py tests/pipeline/test_context_features.py
    git commit -m "feat(challenger): add deterministic context v2 features"

### Task 2: Versioned verifier path and inner-only seam

**Files:**
- Modify: src/pipeline/event_stack.py
- Modify: src/pipeline/runner.py
- Modify: scripts/crossfit_event_stack.py
- Test: tests/pipeline/test_event_stack.py
- Test: tests/pipeline/test_runner.py
- Test: tests/pipeline/test_crossfit.py

**Interfaces:**

    RunConfig.context_features_version: None | "v1" | "v2"
    expected_feature_dimensions(config) -> (..., 56 | 116 | 141, ...)
    multiscale_verifier_features(..., context_features_version="v2", ...) -> ndarray
    @dataclass(frozen=True)
    class OuterTrainOnlyData: ...  # no validation field
    FilesystemDataSource.load_outer_train_only(config) -> OuterTrainOnlyData
    run_outer_train_inner_oof(data: OuterTrainOnlyData, config: RunConfig) -> InnerOOFResult

**Produces:** v2 141D dispatch (legacy 56 + v1 60 + v2 25) and an inner-only API whose input type excludes outer-validation batches.

- [ ] Step 1: Add RED configuration tests: v2 needs micro/bounds, unknown version rejects, v2 width is 141, and None/v1 values/dimensions stay unchanged.
- [ ] Step 2: Run RED: D:/Anaconda3/envs/bme/python.exe -m pytest tests/pipeline/test_event_stack.py tests/pipeline/test_runner.py -k "context or feature_dimension" -v
- [ ] Step 3: Add minimal dispatch: preserve v1 branch unchanged, add v2 branch, validation and metadata/CLI recognition. Do not expose arbitrary grids.
- [ ] Step 4: Write RED protocol tests with sentinels on every validation path/window/label accessor and on load_outer_fold; prove load_outer_train_only and inner-only execution make no calls. Require newly recomputed v1 inner parity with incumbent-compatible result.
- [ ] Step 5: Implement a real OuterTrainOnlyData loader that reads only canonical train-role inputs, then extract existing outer-train OOF/verifier/policy steps into the typed inner-only seam. Do not construct a FoldDataset wrapper. Have current run_outer_fold invoke compatible shared internals and preserve its outer behavior exactly.
- [ ] Step 6: Run GREEN: D:/Anaconda3/envs/bme/python.exe -m pytest tests/pipeline/test_context_features.py tests/pipeline/test_event_stack.py tests/pipeline/test_runner.py tests/pipeline/test_crossfit.py -v
- [ ] Step 7: Commit:
    git add src/pipeline/event_stack.py src/pipeline/runner.py scripts/crossfit_event_stack.py tests/pipeline/test_event_stack.py tests/pipeline/test_runner.py tests/pipeline/test_crossfit.py
    git commit -m "feat(challenger): integrate context v2 verifier path"

### Task 3: Preregistration, cache isolation and mechanism gate

**Files:**
- Create: src/pipeline/challenger/context_v2.py
- Create: scripts/run_challenger_e02.py
- Test: tests/pipeline/test_challenger_context_v2.py
- Test: tests/pipeline/test_challenger_runner.py

**Interfaces:**

    def registered_e02_config(outer_fold: int, *, workers: int) -> RunConfig: ...
    def write_e02_preregistration(root: Path, output: Path) -> Path: ...
    def run_e02_inner_gate(root: Path, output: Path, *, workers: int) -> dict[str, object]: ...
    def gate_passes(report: Mapping[str, object]) -> bool: ...

**Produces:** immutable plan.json/schema JSON/inner_gate.json and a legal/illegal full-run decision.

- [ ] Step 1: Add RED preregistration tests: plan/schema must precede fitting; list base run, code SHA, schema hash and every fixed incumbent grid; reject changed overwrite.
- [ ] Step 2: Add RED boundary tests: E02 cache keys include literal e02-context-v2/full config/code SHA/NumPy 2.4.6/schema hash; paths outside challenger roots fail; no v1/incumbent cache read/write; wrong NumPy version rejects before writes.
- [ ] Step 3: Add RED inner-gate tests with synthetic subject partitions: v1/v2 pairing; outer subject absent from both branches; strictly-greater pooled F1, FP reduction, TP-loss<=2 and >=3/5 non-worse predicates all required; failure suppresses full execution.
- [ ] Step 4: Run RED: D:/Anaconda3/envs/bme/python.exe -m pytest tests/pipeline/test_challenger_context_v2.py tests/pipeline/test_challenger_runner.py -v
- [ ] Step 5: Implement hard-coded registered v1-control/v2 configs differing only in context version. Copy the full incumbent grid/hash set from the spec, including admission minimum recall exactly .80. Recompute five outer-train inner-only reports. Use frozen evidence only for incumbent identity/parity validation, not policy selection, and label the gate development evidence rather than independent validation.
- [ ] Step 6: Implement command modes. --phase inner-gate writes preregistration then gate; --phase full refuses unless the exact immutable gate passed with matching code/config/schema. No release flags.
- [ ] Step 7: Run GREEN: D:/Anaconda3/envs/bme/python.exe -m pytest tests/pipeline/test_challenger_context_v2.py tests/pipeline/test_challenger_runner.py tests/pipeline/test_challenger_protection.py -v
- [ ] Step 8: Commit:
    git add src/pipeline/challenger/context_v2.py scripts/run_challenger_e02.py tests/pipeline/test_challenger_context_v2.py tests/pipeline/test_challenger_runner.py
    git commit -m "feat(challenger): register context v2 inner gate"

### Task 4: Strict outer evidence and accounting

**Files:**
- Modify: src/pipeline/challenger/context_v2.py
- Modify: scripts/run_challenger_e02.py
- Modify: src/pipeline/challenger/atlas.py (only reusable accounting helpers; E01 output behavior unchanged)
- Test: tests/pipeline/test_challenger_context_v2.py
- Test: tests/pipeline/test_challenger_atlas.py

**Interfaces:**

    def run_e02_full_outer(root: Path, output: Path, *, workers: int) -> dict[str, object]: ...
    def compare_e02_to_incumbent(...) -> dict[str, object]: ...
    def append_experiment_registry(path: Path, row: Mapping[str, object]) -> None: ...

**Produces:** one five-fold immutable evidence package and one append-only terminal registry row.

- [ ] Step 1: Add RED evidence tests using synthetic five folds/E00 traces: pooled metrics, fold stability/deltas, slices/candidate diagnostics, all required files, and exact TP/FP accounting partitions.
- [ ] Step 2: Add RED safety tests: reject absent/mismatched plan/schema/gate, protected input changes, output escapes, duplicate E02 registry ID and promotion/release attempts. For full execution, allow only immutable plan/schema/gate beforehand; any existing summary/fold/diff output or terminal registry E02 row fails closed. Assert no force/resume flag exists.
- [ ] Step 3: Implement run only after gate pass. It may run once only; atomically write summary.json, fold_0..4.json, diff CSVs, short_meal_analysis.json, fold_stability.json, schema/plan, leakage_audit.md and comparison.md. Use E01 stable official matching only for accounting; never use outer diff for selection.
- [ ] Step 4: Implement append-only canonical JSONL registry with ID/timestamp/SHA/hypothesis/F1/TP/FP/gate status/output hashes.
- [ ] Step 5: Run GREEN: D:/Anaconda3/envs/bme/python.exe -m pytest tests/pipeline/test_challenger_context_v2.py tests/pipeline/test_challenger_atlas.py tests/pipeline/test_challenger_protection.py -v
- [ ] Step 6: Commit:
    git add src/pipeline/challenger/context_v2.py scripts/run_challenger_e02.py src/pipeline/challenger/atlas.py tests/pipeline/test_challenger_context_v2.py tests/pipeline/test_challenger_atlas.py
    git commit -m "feat(challenger): record context v2 outer evidence"

### Task 5: Verification and registered execution

**Files:** Generate only outputs/challenger_70/experiments/E02/** and outputs/challenger_70/experiment_registry.jsonl.

- [ ] Step 1: Run Gate-0:
    D:/Anaconda3/envs/bme/python.exe -m pytest tests/pipeline/test_context_features.py tests/pipeline/test_event_stack.py tests/pipeline/test_runner.py tests/pipeline/test_crossfit.py tests/pipeline/test_challenger_context_v2.py tests/pipeline/test_challenger_replay.py tests/pipeline/test_challenger_atlas.py tests/parity/test_promoted_release_baseline.py -v
- [ ] Step 2: Reproduce incumbent and confirm protected paths unchanged:
    D:/Anaconda3/envs/bme/python.exe scripts/reproduce_release.py --run-key 160afaf81debf1ee
    git diff --exit-code -- release/event_stack_incumbent.json models/event_stack/160afaf81debf1ee dist/inference dist/submission dist/visual dist/event_stack
- [ ] Step 3: Run only registered inner gate:
    D:/Anaconda3/envs/bme/python.exe scripts/run_challenger_e02.py --root . --output outputs/challenger_70/experiments/E02 --phase inner-gate --workers 5
  If it fails, append the terminal registry row and stop E02; delete only validated disposable E02 cache.
- [ ] Step 4: Only after pass, run formal outer once:
    D:/Anaconda3/envs/bme/python.exe scripts/run_challenger_e02.py --root . --output outputs/challenger_70/experiments/E02 --phase full --workers 5
  Do not tune from partial fold output.
- [ ] Step 5: Re-run Gate-0/reproduction/protected diff and a real single-session CPU timing/RSS smoke. Report, never retune, >2x runtime.
- [ ] Step 6: Commit evidence:
    git add outputs/challenger_70/experiments/E02 outputs/challenger_70/experiment_registry.jsonl
    git commit -m "data(challenger): record E02 context v2 result"
- [ ] Step 7: Stop for parent/user decision: DO NOT PROMOTE if F1 < .70 or evidence is unstable; otherwise only READY FOR HUMAN PROMOTION REVIEW. Never release or rebuild dist.

## Plan self-review

The plan fixes one 25-feature schema before data are scored, leaves v1 unmodified, provides type-level isolation for inner-only selection, retains the incumbent policy surface, isolates cache/output/evidence and forbids release. E03 hard-negative weighting and E04 micro-v2 recovery are deliberately excluded.
