# Amendment A2 E04 Verifier Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce an immutable, sealed-outer E04 explanation of reachable false-negative verifier scores and fold-wise separability without training or changing the release.

**Architecture:** Add a read-only feature view to `ReplayTrace`, calculate fitted-model LR/LightGBM contributions and score/reference rows in pure modules, then aggregate five attested folds and atomically install one diagnostic artifact root. Existing replay/official matcher, A1 protection and sealed bundle contracts remain the authority.

**Tech Stack:** Python 3.11, NumPy 2.4.6, scikit-learn 1.9.0, LightGBM 4.7.0, pytest, Git.

**Spec:** `docs/superpowers/specs/2026-09-21-amendment-a2-e04-verifier-failure-audit-design.md`

## Global Constraints

- **NO TRAINING, NO THRESHOLD TUNING, NO POLICY CHANGE, NO NEW OUTER F1, NO DIST CHANGE, NO RELEASE CHANGE, NO MODEL PROMOTION.**
- Incumbent run `160afaf81debf1ee`, IoU 0.25, 116D feature order, exact pooled 114/153/197.
- E04 is diagnostic only; `outputs/challenger_70/experiment_registry.jsonl` is untouched.
- Original E04 micro-v2 and E05 remain paused; no future challenger is selected here.
- Each new code task writes a failing test first, observes RED, then implements, verifies GREEN and commits.
- Output root is exactly `outputs/challenger_70/e04_verifier_audit/`, created once atomically and never overwritten.

---

### Task 1: Frozen 116D replay view and model contribution reconstruction

**Files:**
- Modify: `src/pipeline/challenger/replay.py`
- Create: `src/pipeline/challenger/verifier_attribution.py`
- Test: `tests/pipeline/test_challenger_verifier_attribution.py`

**Interfaces:**
- `ReplayTrace.verifier_features: np.ndarray | None` exposes the pre-existing raw 116D matrix with a read-only NumPy flag; historical synthetic fixtures may keep `None`.
- `explain_fitted_verifiers(features: np.ndarray, logistic_pipeline: object, lgbm_pipeline: object, expected_lr: np.ndarray, expected_lgbm: np.ndarray) -> dict[str, np.ndarray]` returns per-row LR and LightGBM base/context/bias raw-margin contributions, with strict identity assertions.
- `ReplayTrace.verifier_models: Mapping[str, object]` carries read-only references to the already-fitted verifier pipelines for audit use, never estimators to fit.

- [ ] Write synthetic fitted-pipeline tests for immutable 116D view, LR imputer/scaler coefficient reconstruction, LGBM `pred_contrib` mapping when the imputer drops a column, and fail-closed probability mismatch. Explicitly assert no `fit` call from `explain_fitted_verifiers`.
- [ ] Run `D:/Anaconda3/envs/bme/python.exe -m pytest -q -p no:cacheprovider tests/pipeline/test_challenger_verifier_attribution.py`; confirm RED due to missing interface.
- [ ] Add the replay view at the existing feature computation site. Validate feature width 116 and freeze a copy; store only the two existing fitted verifier references. In the pure attribution helper, apply frozen transforms, map LightGBM transformed columns via the imputer's retained feature indices, partition `[:56]`/`[56:116]`, and validate raw-margin/probability identities at absolute `1e-5`/`1e-6` tolerances. Never call `fit` or `predict_proba` in the helper.
- [ ] Re-run targeted tests and existing replay unit tests; commit as `feat(challenger): expose frozen verifier contributions`.

### Task 2: Pure FN/TP/FP score autopsy and fold-wise statistics

**Files:**
- Create: `src/pipeline/challenger/verifier_audit.py`
- Test: `tests/pipeline/test_challenger_verifier_audit.py`

**Interfaces:**
- `audit_verifier_trace(trace: ReplayTrace) -> dict[str, object]` consumes Task 1 features/attributions and existing `classify_false_negatives`, `official_match_indices` and candidate ledger; returns per-fold `reachable_fn_candidates`, `tp_reference`, `fp_reference`, witness keys, margin/disagreement/context/separability input rows, and deterministic examples. It is pure and does no I/O.
- `summarize_verifier_audits(audits: Sequence[Mapping[str, object]]) -> dict[str, object]` aggregates the fixed margin bins, disagreements, Context distributions, four non-exclusive mechanism flags, and per-feature fold signs without fitting or scoring.

- [ ] Write synthetic traces with multiple raw candidates per FN, tied IoU/score witnesses, an FN-C0 exclusion, matched TP and final FP. Assert all reachable FN pairs are retained, dual witness ties are deterministic, units remain distinct, complete 116D values are exported, and scores/margins agree exactly with frozen ledger values.
- [ ] Run targeted pytest and confirm RED due to missing audit functions.
- [ ] Implement pure row construction using `(fold, raw_index)` coordinates; FN membership from official matches plus IoU 0.25 and E01 stage, TP/FP from official match pairs. Use `base_00..base_55` and `CONTEXT_V1_COLUMNS`; preserve NaN as missing values. Select per-fold examples by most negative and closest negative best-score FN witness with stable ties.
- [ ] Add failing tests for five fixed margin bins, signed LR-minus-LGBM distribution, fixed disagreement flag 0.20, A/B/C/D non-exclusive rules and `unresolved`, empty-fold unsupported quantiles, and `stable_descriptive` only for ≥4/5 same nonzero fold signs. Then implement summary; no AUC or classifier.
- [ ] Re-run targeted tests; commit as `feat(challenger): add pure E04 verifier autopsy`.

### Task 3: Sealed five-fold runner and atomic diagnostic contract

**Files:**
- Create: `src/pipeline/challenger/verifier_audit_run.py`
- Create: `scripts/run_challenger_e04.py`
- Test: `tests/pipeline/test_challenger_verifier_audit_run.py`

**Interfaces:**
- `run_verifier_audit(root: Path, output: Path) -> Path` calls five `replay_outer_fold` traces, Task 2 pure audit and summary, validates E01 raw-reachable FN identity and 114/153/197, then installs the spec's files once.
- CLI `scripts/run_challenger_e04.py --root . --output outputs/challenger_70/e04_verifier_audit` wires the canonical runner.

- [ ] Write injected-runner tests for exact five-fold parity/E01 target identity, output root and symlink/overwrite rejection, protected snapshot unchanged, atomic cleanup on failure, exact file set including data-only `candidate_examples/`, no registry/release/dist writes, no `fit` and no new outer-F1 headline. Confirm RED.
- [ ] Implement runner with existing `protected_snapshot` guard and exact output confinement. Build provenance manifest with code/run/spec/E01/A1/attestation/source/policy/score hashes and group definitions. Validate before staging and again before atomic install; never keep a new cache.
- [ ] Run targeted Task 1–3 tests; commit as `feat(challenger): add sealed E04 audit runner`.

### Task 4: One diagnostic execution, independent verification and documentation

**Files:**
- Create only: `outputs/challenger_70/e04_verifier_audit/**`
- Modify: `README.md`

- [ ] Run Task 1–3 tests plus `tests/pipeline/test_challenger_replay.py`, `tests/pipeline/test_challenger_atlas.py`, `tests/pipeline/test_challenger_protection.py`, and `tests/parity/test_promoted_release_baseline.py`; require no failures and 114/153/197 parity.
- [ ] Verify target absent and worktree clean, then invoke the canonical CLI **once**. If any reconstruction identity or E01 set fails, stop without an installed result.
- [ ] Independently recount FN candidate/TP/FP rows, margin bins, example count and fold signs from installed CSV/JSON; check no `f1`/`outer_f1` field, registry unchanged, protected roots unchanged, no cache/staging leftovers.
- [ ] Update README challenge status with the diagnostic's evidence and limitations, without a new F1 claim or automatic challenger choice; commit only README plus immutable evidence as `data(challenger): record E04 verifier audit`.
- [ ] Report A/B/C/D descriptive breakdown, LR/LGBM and Context findings, fold consistency and limitations. Stop for human review; do not start a challenger.

## Self-review

Tasks 1–2 cover frozen feature/score/attribution identity and pure diagnostic populations; Task 3 covers provenance, immutability and protection; Task 4 covers one-shot evidence, independent arithmetic, documentation and human handoff. No training API appears in the planned production path. All substantive output fields in the approved spec map to a task and test.
