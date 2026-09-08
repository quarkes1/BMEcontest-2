# Subject-budget decoder implementation plan

> Date: 2026-09-08
> Baseline: locked nested CPU/no-TCN F1 0.416 (89/153 TP, 275 predictions)
> Coverage candidate ceiling: 132/153 matched candidates, but 324 predictions and F1 0.432
> Final project target: locked F1 >= 0.65

## Objective and rationale

The nested runner exposed a calibration/domain-shift failure: inner folds select roughly one
prediction per true event, while outer fold 2/3 emit 66/81 predictions for only 27/32 truths.
The dataset contains about four eligible meals per subject. A subject-level event budget learned
only from inner OOF predictions can stabilize this operating point without reading outer labels.

This stage is deliberately narrow. It tests whether the existing verifier ranking retains enough
true events after false-positive suppression. Coverage mode has 132 candidate-level matches; if a
roughly 150-180 prediction budget retains at least 100 TP, aggregate F1 can reach 0.65. If it does
not, the next stage must improve ranking with aggregated raw 62-dimensional window features rather
than tune density thresholds again.

## Task 1: Pure event-policy primitives

**Files:**
- Modify: `src/pipeline/event_stack.py`
- Modify: `tests/pipeline/test_event_stack.py`

1. Add a failing test that `apply_event_policy()` keeps at most K highest-scored events per
   subject, while preserving events from different subjects and applying the score threshold.
2. Add a failing test that `select_event_policy()` chooses only from the registered cap grid and
   breaks equal-F1 ties by PPV, smaller cap, then stricter threshold.
3. Implement immutable `EventSelectionPolicy(threshold, max_events_per_group, metrics)` plus pure
   `apply_event_policy()` and `select_event_policy()` functions.
4. Require event, score, and group arrays to align; reject non-finite scores and invalid caps.
5. Run focused and full tests.
6. Commit: `feat: add subject-budget event policy`.

## Task 2: Nested-runner integration

**Files:**
- Modify: `src/pipeline/runner.py`
- Modify: `tests/pipeline/test_runner.py`

1. Add `subject_cap_grid: tuple[int, ...] = ()` to `RunConfig` and
   `max_events_per_subject: int | None` to `FoldResult`; both enter JSON/cache identity.
2. Add a failing synthetic test proving the selected cap comes only from inner candidates and is
   applied to outer candidates by subject.
3. Select threshold and cap jointly on verifier OOF candidates using candidate subject IDs.
4. Apply the frozen policy to untouched outer candidates. Keep the existing threshold-only path
   byte-for-byte equivalent when the grid is empty.
5. Preserve both outer isolation guards and cache/result audit fields.
6. Run full tests and compile checks.
7. Commit: `feat: learn subject event budget from inner oof`.

## Task 3: CLI and registered ablation

**Files:**
- Modify: `scripts/crossfit_event_stack.py`
- Modify: `README.md`
- Modify: `docs/三阶段重构设计.md`

1. Add `--subject-cap-grid`, parsed as a comma-separated positive integer tuple. The registered
   experiment grid is `2,3,4,5,6`; no outer-derived value may be inserted after results are seen.
2. Run both locked configurations with CPU fold parallelism:

       D:/Anaconda3/envs/bme/python.exe scripts/crossfit_event_stack.py --fold all \
         --inner-splits 4 --no-tcn --workers 0 --subject-cap-grid 2,3,4,5,6

       D:/Anaconda3/envs/bme/python.exe scripts/crossfit_event_stack.py --fold all \
         --inner-splits 4 --no-tcn --workers 0 --coverage-fix \
         --subject-cap-grid 2,3,4,5,6

3. Adopt the budget decoder only if aggregate PPV rises, aggregate F1 improves by at least 0.02,
   and sensitivity falls by no more than 0.05 versus the matching uncapped configuration.
4. Record per-fold config hashes, selected caps, inner/outer metrics, candidate recall, slices,
   runtime, and remaining gap to F1 0.65.
5. Run full tests, CLI help, diff checks, and status hygiene.
6. Commit: `feat: evaluate nested subject budget`.

## Decision gate for the next stage

- If either locked aggregate reaches F1 >= 0.65, freeze the winning configuration and run a
  reproducibility pass before deployment work.
- If the best result is >= 0.55 but < 0.65, keep the budget decoder and add candidate-level raw
  feature aggregation under the same nested protocol.
- If the best result is < 0.55, treat verifier ranking—not threshold calibration—as the dominant
  bottleneck and proceed directly to raw-feature aggregation plus hard-negative modelling.
