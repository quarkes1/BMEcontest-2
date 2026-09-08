# Raw candidate verifier implementation plan

> Date: 2026-09-08
> Best locked diagnostic so far: coverage + subject budget F1 0.465
> Final project target: locked F1 >= 0.65

## Objective

The subject-budget experiment improved precision but removed too many true events, proving that
the current 37/42 verifier features cannot reliably rank true meals above same-subject hard
activities. The window HGB collapses 62 sensor-derived features into one probability, so the
verifier loses information that may separate these events. This stage exposes raw window-feature
summaries directly to the verifier while retaining the nested subject-disjoint protocol.

## Task 1: Pure aligned raw-feature aggregation

**Files:**
- Modify: `src/pipeline/event_stack.py`
- Modify: `tests/pipeline/test_event_stack.py`

1. Add failing tests for exact output dimensions, session isolation, event/context membership,
   and missing-context NaN handling.
2. Implement `aggregate_candidate_features(candidates, windows, window_features, context_ms)`.
3. For each of the 62 source columns append event mean, standard deviation, P10, P90, and event
   mean minus combined pre/post-context mean. Append event/context observed-window counts.
4. Keep source features aligned by `EventRef`; reject width/row mismatches and duplicate window
   references. Return 312 columns for 62 source features.
5. Run focused/full tests and commit `feat: aggregate raw candidate window features`.

## Task 2: Nested verifier integration and model selection

**Files:**
- Modify: `src/pipeline/runner.py`
- Modify: `tests/pipeline/test_runner.py`

1. Add `verifier_feature_mode: str = "probability"` and
   `verifier_c_grid: tuple[float, ...] = (0.1,)` to `RunConfig`; include both in cache identity.
2. Add `verifier_c: float` and `verifier_feature_count: int` to `FoldResult`/JSON.
3. In `raw_summary` mode concatenate the existing probability/coverage features with the aligned
   312-column aggregate for both OOF train candidates and untouched outer candidates.
4. Evaluate the pre-registered C grid `(0.001, 0.01, 0.1)` exclusively on verifier OOF event F1.
   Tie-break by PPV, then stronger regularization (smaller C). Fit the final verifier with the
   selected C and select threshold/cap using its OOF scores.
5. Add synthetic tests for feature count, selected C membership, outer isolation, and unchanged
   probability-only dimensions.
6. Run tests/compile checks and commit `feat: add nested raw candidate verifier`.

## Task 3: CLI and locked ablations

**Files:**
- Modify: `scripts/crossfit_event_stack.py`
- Modify: `README.md`
- Modify: `docs/三阶段重构设计.md`

1. Add `--verifier-features probability|raw_summary` and `--verifier-c-grid` with strict parsing.
2. Pre-register and run these configurations, without adding variants after outer results appear:

       # Raw summaries on legacy candidates
       ... --verifier-features raw_summary --verifier-c-grid 0.001,0.01,0.1

       # Raw summaries on coverage candidates
       ... --coverage-fix --verifier-features raw_summary \
           --verifier-c-grid 0.001,0.01,0.1

       # Raw summaries plus the already registered subject cap grid
       ... --coverage-fix --subject-cap-grid 2,3,4,5,6 \
           --verifier-features raw_summary --verifier-c-grid 0.001,0.01,0.1

3. Adopt raw summaries if aggregate F1 improves by at least 0.02 over the matching probability
   configuration and no critical slice (short or non-dominant recall) falls by more than 0.05.
4. Record hashes, selected C/K, metrics, slice recall, feature count, runtime, and the remaining
   gap to 0.65. Run full verification and commit `feat: evaluate raw candidate verifier`.

## Decision gate

- At F1 >= 0.65, freeze and reproduce the winning configuration before deployment changes.
- At 0.55-0.65, retain raw summaries and add leakage-safe frozen-FD or inner-trained TCN scores.
- Below 0.55, proceed to explicit hard-negative/event-sequence modelling; the raw-summary OOF
  artifacts will provide the training surface without adding standalone probe scripts.
