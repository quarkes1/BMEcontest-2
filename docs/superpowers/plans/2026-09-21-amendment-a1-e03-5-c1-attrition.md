# Amendment A1 E03.5 C1 Attrition Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce immutable, sealed-outer diagnostic evidence explaining the 11 short C1 false negatives and the FP cost of three frozen-score single-operation ablations.

**Architecture:** Extend the existing read-only replay/Atlas path with a pure E03.5 evaluator. Reuse attested `ReplayTrace` scores, candidates and policy; never invoke training or select a new policy. A thin runner validates five-fold incumbent parity and E01 target identity, then atomically writes diagnostic-only outputs.

**Tech Stack:** Python 3.11, NumPy 2.4.6, existing pipeline/eval primitives, pytest, Git.

**Spec:** `docs/superpowers/specs/2026-09-21-amendment-a1-e03-5-c1-attrition-design.md`

## Global Constraints

- E04/E05 are paused; E03.5 is diagnostic only and cannot produce a new F1 headline or official registry performance field.
- Use sealed incumbent run `160afaf81debf1ee`, frozen 116D Context-v1, unchanged IoU 0.25 matcher, unchanged scores and policy.
- The three interventions are exactly disable NMS only, disable admission threshold only, and disable admission subject cap only.
- No `fit`, no additional `predict_proba` after sealed replay, no model/cache/release/dist writes.
- Oracle is `DIAGNOSTIC ONLY / USES GT`, operates independently per C1 truth, and cannot feed training or policy selection.
- New immutable output root: `outputs/challenger_70/e03_5_c1_attrition/`.

---

### Task 1: Pure layered attrition and counterfactual evaluator

**Files:**
- Modify: `src/pipeline/challenger/replay.py` (`ReplayTrace` gains the attested policy metadata; incumbent replay logic is unchanged)
- Create: `src/pipeline/challenger/c1_attrition.py`
- Test: `tests/pipeline/test_challenger_c1_attrition.py`

**Interfaces:**
- Consumes `ReplayTrace` including its exact attested `bundle.policy`, `classify_false_negatives`, `CandidateAdmissionConfig`, canonical candidate-control/event-policy primitives.
- Produces `audit_trace(trace: ReplayTrace) -> dict[str, object]` with truth/candidate rows, three ablations, event diffs and oracle rows; no I/O.

- [ ] Write failing synthetic-trace tests for: replay exposes attested policy without altering incumbent predictions; exactly one raw reachable C1 truth per NMS/threshold/cap branch; mixed reasons; frozen score identity; candidate-stage and official-matcher final FP accounting; oracle reset and downstream threshold/cap rejection. Use tiny `ReplayTrace` fixtures with explicit events/scores and no fitted models.
- [ ] Run `D:/Anaconda3/envs/bme/python.exe -m pytest -q -p no:cacheprovider tests/pipeline/test_challenger_c1_attrition.py` and confirm each new test fails on missing evaluator behavior.
- [ ] Implement pure helpers `_frozen_policy(trace)`, `_admitted_for_mode(trace, mode)`, `_final_for_indices(trace, indices)`, and `audit_trace(trace)`. Reconstruct the three ablations from frozen raw union and `blend_score`; call canonical NMS/admission and event-policy primitives with the incumbent constants. Validate baseline admitted/final indices against the trace before any ablation. Count newly admitted nonmatching candidates using IoU 0.25 to eligible truths. Use official matcher for newly matched truths and unmatched final-event differences, preserving duplicate coordinates.

```python
MODES = ("disable_nms_only", "disable_admission_threshold_only", "disable_subject_cap_only")
assert set(MODES) == {
    "disable_nms_only", "disable_admission_threshold_only", "disable_subject_cap_only"
}
```
- [ ] Implement per-truth oracle by adding one raw-reachable candidate to baseline admission, applying unchanged event policy, and official matching; process each truth independently and choose deterministic witness by `(-IoU, -frozen_score, raw_index)`.
- [ ] Re-run the targeted tests; commit only code/tests with `feat(challenger): add pure E03.5 attrition diagnostics`.

### Task 2: Sealed five-fold runner and immutable artifact contract

**Files:**
- Create: `scripts/run_challenger_e03_5.py`
- Create: `src/pipeline/challenger/c1_attrition_run.py`
- Test: `tests/pipeline/test_challenger_c1_attrition_run.py`

**Interfaces:**
- Consumes Task 1 `audit_trace`, `replay_outer_fold`, `protected_snapshot`, incumbent/E01 evidence.
- Produces `run_c1_attrition(root: Path, output: Path) -> Path`, writing only the six files in the spec under the exact output root.

- [ ] Write failing runner tests for: five-fold 114/153/197 parity; exact 11 short C1 keys from E01; no registry write; output-root/symlink/overwrite rejection; no `fit` or post-replay `predict_proba`; protected snapshot unchanged; atomic all-or-nothing installation.
- [ ] Run `D:/Anaconda3/envs/bme/python.exe -m pytest -q -p no:cacheprovider tests/pipeline/test_challenger_c1_attrition_run.py` and confirm RED.
- [ ] Implement runner with injected replay seam for tests but canonical CLI wired to `replay_outer_fold`. Validate E01 target truth keys, baseline, and protected snapshot before staging. Create `manifest.json`, `candidate_attrition.csv`, `truth_attrition.csv`, `counterfactual_events.csv`, `counterfactual_summary.json`, `oracle.csv`, `report.md`; the manifest must mark diagnostic-only/GT-aware artifacts. Atomically install once into the exact output root, without modifying `experiment_registry.jsonl`.

```python
OUTPUT_FILES = (
    "manifest.json", "candidate_attrition.csv", "truth_attrition.csv",
    "counterfactual_events.csv", "counterfactual_summary.json", "oracle.csv", "report.md",
)
assert len(OUTPUT_FILES) == 7
```
- [ ] Run targeted runner/evaluator tests; commit code/tests with `feat(challenger): add sealed E03.5 audit runner`.

### Task 3: Verification and one immutable E03.5 diagnostic

**Files:**
- Create only: `outputs/challenger_70/e03_5_c1_attrition/**` (seven files)

- [ ] Run `D:/Anaconda3/envs/bme/python.exe -m pytest -q -p no:cacheprovider tests/pipeline/test_challenger_c1_attrition.py tests/pipeline/test_challenger_c1_attrition_run.py tests/pipeline/test_challenger_atlas.py tests/pipeline/test_challenger_replay.py tests/pipeline/test_challenger_protection.py tests/parity/test_promoted_release_baseline.py`; require all pass and incumbent parity.
- [ ] Verify `outputs/challenger_70/e03_5_c1_attrition/` does not exist, then run `D:/Anaconda3/envs/bme/python.exe scripts/run_challenger_e03_5.py --root . --output outputs/challenger_70/e03_5_c1_attrition` exactly once.
- [ ] Independently recompute counts from CSV/JSON outputs; verify exactly 11 short C1 rows, baseline 114/153/197, no F1/`outer_f1` headline, no registry change, release/dist/model snapshots unchanged, and worktree contains only intended evidence.
- [ ] Commit immutable evidence with `data(challenger): record E03.5 C1 attrition audit`.
- [ ] Report the cause/FP-cost table without selecting a challenger. Stop for human review; E04/E05 remain paused.

## Plan self-review

Task 1 covers candidate/truth attrition, all three frozen-score ablations, FP costs and GT-aware oracle. Task 2 covers immutable output, provenance, no-registry and protected-boundary checks. Task 3 covers parity, one-shot execution, independent accounting and human review. No release or new model code is in scope.
