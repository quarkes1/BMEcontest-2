# Cross-Fitted Event Stack Design

## Purpose

Replace the current optimistic two-stage validation path with a subject-disjoint,
cross-fitted event stack.  The stack must generate verifier-training candidates
from out-of-fold window scores, select one deployable threshold without looking
at an outer validation subject, and report the error slices that drive the next
F1 improvements.

## Evidence and baseline

The current clean TCN CV result is 78 TP / 153 eligible meals / 152 predictions
(global F1 0.512), but its verifier threshold is selected independently on each
validation fold.  Re-scoring the saved OOF candidates with one pooled threshold
gives F1 0.484; fixed threshold 0.717 gives F1 0.478.  Strict CPU LOSO is 0.416.
The legacy 0.632 wbag result is invalid because its scoring models saw validation
subjects in other folds.

The final errors are 42 threshold rejects with an IoU-valid candidate, 28
candidate misses, 5 candidate geometry misses, and 74 false positives.  Short
meals and non-dominant-wrist meals are disproportionately affected.  In
particular, a 240-second window labelled positive only above 50% meal overlap
cannot label a meal of 120 seconds or less as positive.

## Goals

1. Reach aggregate event F1 >= 0.65 on subject-disjoint outer CV, with every
   outer-fold threshold frozen from inner OOF predictions.  This is the final
   research acceptance target; an independently held-out competition score is
   still the authority for submission performance.
2. Produce a deterministic, subject-disjoint outer-CV estimate with a single
   threshold learned only from inner OOF predictions.
3. Train the event verifier on candidates scored by a window model that did not
   train on that candidate subject.
4. Support a coverage-aware candidate path so the existing density coverage
   correction can be evaluated with a verifier trained on matching examples.
5. Make short-meal and non-dominant-wrist recall first-class reported metrics.
6. Keep runtime inference compatible with the existing CPU submission pipeline;
   TCN remains an optional score input rather than a deployment dependency for
   this phase.
7. Keep nested training practical through bounded CPU parallelism, batched
   scoring, and deterministic score caches; later neural stages must support
   CUDA without making GPU availability part of correctness.

## Non-goals

- Do not claim a new submission F1 from per-fold threshold optimisation.
- Do not reintroduce cross-fold model bags into validation scoring.
- Do not add raw PPG, retrain the TCN, or change the official IoU evaluator in
  this phase.
- Do not enable the density coverage correction in deployment unless it improves
  the nested outer-CV result at the frozen threshold.

## Architecture

### Data flow

For each outer GroupKFold fold, all subjects in the outer validation set remain
unseen until final scoring.

```
outer-train subjects
  -> inner GroupKFold window models
  -> inner OOF window probabilities for every outer-train subject
  -> density candidates (+ optional coverage features)
  -> event verifier training and inner-OOF verifier probabilities
  -> one frozen verifier threshold

outer-train subjects -> final outer window model -> outer validation windows
  -> density candidates using the same candidate configuration
  -> fitted event verifier + frozen threshold -> official event metrics
```

The verifier may use a final outer-train fit after its threshold is chosen from
inner OOF scores.  It must never be fit on outer-validation candidates.  The
window model used to score a verifier-training subject must exclude that subject
from its training rows.

### Code boundaries

- `src/pipeline/event_stack.py` — pure, testable utilities for candidate
  scoring, candidate labels, greedy event metrics aggregation, coverage metadata,
  and deterministic threshold selection.  It must not read files or inspect
  environment variables.
- `src/pipeline/crossfit.py` — subject-disjoint fold orchestration over supplied
  arrays and window identifiers.  It exposes explicit data classes for train,
  validation, OOF candidates, and evaluation results.
- `scripts/crossfit_event_stack.py` — the only CLI entry point.  It loads cached
  slide tables, calls the pipeline, writes one compact JSON result under
  `outputs/`, and accepts `--fold`, `--inner-splits`, `--coverage-fix`, and
  `--no-tcn`, plus `--workers` and `--device auto|cpu|cuda` for execution
  control.
- `tests/` — synthetic deterministic unit tests.  No test reads `Data/`,
  `cache/`, checkpoints, or large output files.
- `scripts/slide_verifier.py` — retain its current diagnostic command, but
  delegate shared candidate/label/threshold logic to `src/pipeline` after
  behavioural parity tests are green.

The new source package avoids adding another experimental top-level script with
duplicated event logic.  Large candidate dumps remain in `outputs/` and are not
committed.  The CLI result JSON is compact and reproducible.

### Execution efficiency

Array tables are loaded once per outer-fold worker, window probabilities are
computed in batches, and reusable inner-OOF scores are stored under
`cache/crossfit/` with a hash of fold IDs, features, model configuration, seed,
and candidate configuration.  Cache files are never written to `scripts/` or
`src/`.

`--workers 0` selects `min(physical_cpu_count, number_of_requested_outer_folds)`.
When fold processes run concurrently, each process limits native BLAS/OpenMP
thread pools to one thread; `--workers 1` may use the configured native thread
count.  This avoids nested parallel oversubscription.  All result files record
wall-clock seconds per stage and whether each score cache was hit.

The foundation HGB/LR stack is CPU-native.  Any later TCN or sequence verifier
uses batched tensors and `--device auto|cpu|cuda`; `auto` selects CUDA only when
available.  CPU and CUDA implementations must consume the same timestamps and
emit the same candidate-score schema.

### Coverage-aware candidates

The density routine accepts `coverage_fix: bool`.  With it enabled, a missing
segment outside the observed local time range does not count as a zero-coverage
window; explicit bridged gaps inside a segment still do.  Candidate metadata
must additionally carry:

- fraction of observed windows in the 600-second density neighbourhood;
- count and total duration of bridged gaps inside that neighbourhood;
- observed-window counts in the 20-minute pre- and post-context regions.

The verifier receives these values only when the matching candidate mode is
used in both inner training and outer validation.  This prevents the current
distribution mismatch where coverage-fixed candidates are judged by a verifier
trained on the old candidate population.

### Threshold policy

`select_event_threshold` evaluates every unique verifier score plus explicit
boundary values.  It uses the official greedy one-to-one IoU metric, not
candidate-row classification F1.  The selected threshold is a single scalar for
the whole outer fold and is derived only from the inner OOF candidate pool.
Tie-breaking is deterministic: highest event F1, then higher PPV, then higher
threshold.

The result JSON records the selected threshold, inner event metrics, outer event
metrics, candidate count, candidate-match recall, and per-slice metrics for
`dominant`, `nondominant`, `<10m`, `10-20m`, and `>=20m` eligible meals.

## Validation contract

Every change is accepted only when all of the following are true:

1. Synthetic tests prove subject exclusion, one global threshold selection,
   official one-to-one matching, and coverage-gap semantics.
2. The five outer fold result files are generated without loading a model trained
   on that fold's validation subject.
3. Reported aggregate F1 uses the frozen inner-derived thresholds.  The report
   preserves the diagnostic per-fold optimum separately and never labels it as
   deployment performance.
4. A candidate improvement must improve or preserve outer aggregate PPV while
   reducing the relevant candidate-miss slice.  A total-F1 change under 0.01 is
   recorded as inconclusive rather than adopted.
5. Strict CPU LOSO is re-run before any change is copied into `dist/`.
6. A one-fold smoke run reports stage timings and a second identical run hits the
   score cache without changing metrics or selected thresholds.

## Delivery sequence

1. Add the pure event-stack utilities and their failing-first tests.
2. Add inner/outer cross-fit orchestration and tests proving no subject overlap.
3. Add the CLI and compact JSON schema; reproduce the current no-coverage-fix
   baseline under frozen thresholds.
4. Enable and evaluate coverage-aware candidates with the new verifier features.
5. Only if step 4 is positive, build a separate short-meal specialist design;
   it requires its own test matrix and does not share threshold tuning with the
   main stack.

The full F1 >= 0.65 programme is deliberately split into sequential, measurable
milestones:

- Milestone A — trustworthy stack: nested subject OOF, frozen thresholds, and
  CPU deployment parity.  This milestone establishes the score to beat and does
  not claim an F1 increase by itself.
- Milestone B — recall recovery: coverage-aware candidates and a 60/120-second
  short-meal specialist.  Target candidate-match recall is at least 0.90 overall
  and at least 0.70 for meals shorter than 10 minutes, without reducing outer
  aggregate PPV.
- Milestone C — precision recovery: reviewed hard-negative mining and an
  event-level sequence verifier using the existing six-channel ACC+GYRO signal;
  PPG is evaluated only as a candidate-level addition.  Variants are retained
  only when locked outer-CV aggregate F1 improves by at least 0.01.

The programme is complete only when locked outer-CV aggregate F1 is at least
0.65 and the final `dist/` pipeline reproduces the selected feature set and
threshold policy.  If Milestones B and C exhaust their registered variants below
0.65, the result is reported as a data-information limit rather than presenting
an optimistically tuned score.

## Risks and mitigations

- The small number of eligible meals makes a single split noisy.  Retain all
  five outer folds, report aggregate counts, and do not select variants on an
  individual fold.
- High-scoring unlabelled candidates may be real meals omitted from annotation.
  Treat mined negatives as provisional until manually reviewed; do not blindly
  hard-label all of them as negative.
- Nested fitting is more expensive than current scripts.  Reuse cached window
  features and write only compact JSON by default; detailed candidate arrays are
  opt-in diagnostics under `outputs/`.
