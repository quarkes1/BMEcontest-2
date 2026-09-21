# Amendment A2 — E04 Sealed Verifier Failure & Separability Audit

## Authority and decision boundary

This amendment supplements `docs/challenge设计任务书.md` and follows the
completed A1/E03.5 diagnostic. It **reuses** E04 as an audit identifier; it
does not authorize the original E04 micro-v2 experiment. E05 remains paused.
The immediate question is why an IoU-reachable true meal candidate is assigned
a low score by the incumbent verifier. E04 ends with a human review of the
evidence, not with an automatically selected challenger.

The registered prohibitions are: **NO TRAINING, NO THRESHOLD TUNING, NO POLICY
CHANGE, NO NEW OUTER F1, NO DIST CHANGE, NO RELEASE CHANGE, NO MODEL
PROMOTION.** Do not append a performance record to
`outputs/challenger_70/experiment_registry.jsonl`. Outer-fold observations
are already-used development diagnostics, not an independent generalization
estimate or an inner-OOF selection surface.

## Inputs, population and immutable baseline

Use five `replay_outer_fold(root, fold)` traces for the attested incumbent run
`160afaf81debf1ee`. The replay must reproduce each fold's sealed TP/eligible/
prediction counts and pooled **114/153/197** with IoU **0.25**. The raw union,
candidate geometry, features, fitted verifier pipelines, scores, blend weight,
event threshold/cap and event matching stay frozen. The A1 output is supporting
provenance, not an input that changes selection. Validate the protected-root
snapshot before and after writing; fail closed on any mismatch.

The primary population is **every eligible truth that is unmatched in the
incumbent final predictions but has at least one raw-union candidate with
IoU ≥ 0.25**. Record its E01 FN stage and duration slice. This includes C1,
C2 and C3 if present, and any other raw-reachable FN; C0 is excluded by
definition. Report the total target count and its E01 stage split before any
interpretation. `reachable_fn_candidates.csv` contains **every** qualifying
truth/raw-candidate pair, not only a favorable witness. Mark each row
`DIAGNOSTIC ONLY / USES GT`.

For each target truth, retain two deterministic witnesses separately:

- `best_iou`: highest IoU, then highest frozen blend score, then lowest raw
  index;
- `best_score`: highest frozen blend score, then highest IoU, then lowest raw
  index.

This prevents a post-hoc choice of whichever witness supports a hypothesis.
Matched TP reference rows use the official one-to-one matched final event and
its raw index. Final FP reference rows use every officially unmatched final
event and its raw index. All three groups use the same 116D schema and score
fields, but the FN witness is GT-aware while TP/FP are actual final decisions.
Do not compare group counts as though the sampling units were identical:
candidate rows, truth witnesses and final events are distinct units.

## Step 1 — frozen score autopsy

Expose the already-computed raw **116D** verifier feature matrix from sealed
replay as an immutable audit view. Export base 56D as fixed `base_00` through
`base_55` columns and Context-v1 60D under its registered schema column
names, with NaN represented as an empty CSV cell. Export source support,
IoU for FN pairs, LR probability, LGBM
probability, incumbent blend score, frozen blend weight, event threshold,
probability margin `blend - threshold`, and the existing candidate-control /
event-policy terminal reason. The replay may expose these existing values but
must not refit or recompute probabilities.

`score_margin_analysis.json` reports fold-wise and pooled descriptive counts
for signed margins, plus both GT-aware FN witness definitions. The fixed
diagnostic bins are `margin < -0.10` (deep below),
`[-0.10, -0.05)`, `[-0.05, 0)` (near below), `[0, 0.05]` and `> 0.05`;
these are **not** decision thresholds and must not be optimized. Preserve
exact margins in CSV so the bins cannot hide the underlying distribution.
`lr_lgbm_disagreement.json` reports absolute probability differences and
fold-wise signed LR-minus-LGBM distributions for FN witnesses, TP and FP.
The fixed descriptive flag is `|p_LR - p_LGBM| ≥ 0.20`; also record which
model is higher. Never treat either individual probability as though the
incumbent blend threshold were its independently calibrated threshold.

## Step 2 — within-model Context contribution

The only allowed comparison is **within each already-fitted 116D model**.
This is not a 56D-versus-116D retraining ablation and cannot claim the causal
effect of adding Context-v1 to training. No synthetic Context replacement is
fed through `predict_proba`, and no new outer score is produced.

For the fitted LR pipeline, apply its frozen imputer and scaler to the same
116D replay features, then partition `coefficient × transformed_feature`
into base columns 0–55 and Context columns 56–115. Preserve the intercept;
assert `intercept + base_sum + context_sum` reconstructs the LR decision
function and stored probability within absolute tolerance `1e-6`.

For the fitted LightGBM pipeline, apply its frozen imputer and obtain native
`pred_contrib` values from the existing booster for the same rows. Map every
transformed column back to its original 116D index, including the possibility
that `SimpleImputer` dropped all-missing columns; do not assume transformed
width is 116. Partition mapped feature contributions into base/Context plus
the bias term; assert their sum reconstructs its raw margin within absolute
tolerance `1e-5` and stored probability within `1e-6`. Attribution calls
are allowed solely for explaining
the frozen score, not for candidate rescoring or policy decisions. If the
native contribution contract is unavailable or reconstruction fails, mark
the Context audit unavailable and **abort the final artifact** rather than
substitute a retrained 56D model or an unlabelled counterfactual.

`context_contribution.json` carries per-fold base/Context contribution
distributions for FN witnesses, matched TP and final FP, plus candidate-level
join keys into the CSVs. The sums are in each model's **log-odds/raw-margin
space**, not probability points. A negative Context sum is evidence about
that fitted model's decomposition, not proof that removing Context would
raise an event's deployed probability or final recall. Report LR and LGBM
separately before discussing agreement.

## Step 3 — fold-wise separability and mechanism labels

`separability_by_fold.json` compares rejected true candidates to final FP in
the existing feature and attribution spaces, **one fold at a time**. For each
feature, report group count, median, IQR, missingness and the sign of the
FN-minus-FP median difference for every fold. The pooled difference is
descriptive only. Mark a direction `stable_descriptive` only when at least
four of five folds have nonempty groups and the same nonzero sign; otherwise
mark `inconsistent_or_unsupported`. Report raw counts so a tiny fold cannot
silently receive the same evidential weight as a large fold. No classifier,
AUC optimization, feature selection, or outer-derived rule is fitted.

The report presents four **non-exclusive, descriptive** mechanism flags and
an `unresolved` option, applied to both FN witness definitions. A: blend
margin in `[-0.05, 0)`. B: for **both** fitted models, base contribution ≥
the same-fold matched-TP median, while Context contribution is negative and
≤ the same-fold matched-TP first quartile; if only one model meets this, mark
`B_model_disagreement` separately rather than B. C: `|p_LR - p_LGBM| ≥ 0.20`.
D: **both** model probabilities are below their same-fold matched-TP first
quartile and each lies within the same-fold final-FP interquartile range;
also report whether any individual existing feature has a stable descriptive
direction across folds. Undefined quantiles from empty reference groups give
`unsupported`, never a forced label. These labels are for
explanation, not a mutually exclusive prediction taxonomy or a recipe for
promotion. Cross-fold sign instability precludes a claim of a usable
separation direction. Do not infer a training branch automatically from the
largest descriptive category.

## Output and provenance

Create one new immutable root only:

```text
outputs/challenger_70/e04_verifier_audit/
├── manifest.json
├── report.md
├── reachable_fn_candidates.csv
├── tp_reference.csv
├── fp_reference.csv
├── score_margin_analysis.json
├── lr_lgbm_disagreement.json
├── context_contribution.json
├── separability_by_fold.json
└── candidate_examples/
```

`candidate_examples/` contains a small deterministic set of **data-only**
JSON examples keyed by fold/truth/raw index; no generated images, scripts or
model files. Per fold, select at most two `best_score` FN witnesses: the most
negative margin and the closest negative margin to zero, each tied by
`(truth_key, raw_index)`; deduplicate when the same witness wins both. Flag
every example GT-aware. The manifest records code SHA, run key, spec/E01/A1 hashes,
attested bundle/source hashes, frozen policy and score hashes, exact output
file list, protected snapshot, group definitions and prominent diagnostic-
only warnings. Existing output is never overwritten or resumed. A failed
attempt must leave no partial installed output or retained cache.

The report contains no new F1 headline and no recommended challenger
parameters. It may state evidence-backed hypotheses, limitations and
mechanism counts with fold-wise support. After human review, any future
challenger is separately preregistered and developed solely on
`outer-train → inner OOF`. An eventual outer gate is a new experiment, not
part of E04; the proposed engineering bar is inner aggregate F1 ≥ 0.78 from
the current ~0.7656, without a material TP loss or disproportionate FP cost,
and with most inner folds moving in the same direction.

## Correctness gates

Tests must establish: attested five-fold 114/153/197 parity; exact eligible
raw-reachable FN target set against E01; deterministic dual witnesses;
candidate/truth/event unit separation; complete 116D ordering and score-chain
identity; LR and LGBM contribution reconstruction; fold-wise sample/missing
counts and stable-direction logic; no `fit`, threshold search, policy mutation,
registry row, release/dist/model write or new F1 field; immutable atomic
output; protected snapshot parity; and readable but data-only examples.
Run the existing incumbent replay/parity tests before installing the output.
