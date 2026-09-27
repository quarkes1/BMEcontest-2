# A3-S3 — Policy-OOF score stability audit

ID `RV2-A3-S3-001`. Follow-up to S2 under autonomous exploration authorization. Diagnostic only: NO TRAINING, NO NEW POLICY, NO HEALTH THRESHOLD SWEEP, NO OUTER, NO DIST/RELEASE CHANGE, NO NEW F1.

## Frozen scope

Only read nine `fold{k}_policy{j}_scores.json` and three `fold{k}_policies.json` from committed S2 evidence (k,j=0,1,2). No health score/report files, raw IMU, source training or outer artifacts are inputs to the analysis. Each wrapper's16 policy subjects are reused across wrappers; report separately, do not present48 subject occurrences as48 independent people. Verify actual graph excludes each policy-validation partition, graph subject sets, array alignment, finite probabilities, fixed .75/.25 fusion arithmetic, and invalid-row equality to persisted readout prior. Hash every input before/after.

## Registered descriptive outputs

1. Per policy fold and concatenated wrapper, B/R/F fused scores plus standalone base/random/FD component profiles. Fixed quantiles .1/.5/.9, mean, prevalence, Brier, ten fixed equal-width reliability bins [0,1], ECE (descriptive only). Include valid/invalid and positive/negative groups; undefined empty groups null, never zero observations disguised as evidence.
2. Report subject-level profiles for actual observed candidates, plus candidate/valid counts. Scores are not guaranteed calibrated posterior probabilities; class weighting/sampling and correlated overlapping candidates affect interpretation. No significance tests on candidate rows treated as independent.
3. Invalid-motion effect: observed priors by policy fold, .25*(prior−base) score shift, positive/negative distributions. Scores, models, priors and policy remain frozen. Do not replace invalid values by zero or remove their candidates.
4. For each wrapper/arm, freeze its chosen S2 admission policy and evaluate event counts at the registered event threshold plus offsets −.05,−.025,0,+.025,+.05, confined to policy OOF candidates/truth. Report TP/FP/FN and event threshold only, no F1, no ranking, no preferred offset. Do not search admission, fusion or additional threshold values. This is sensitivity analysis on policy-selection data, not held-out performance or a challenger. Preserve >= ties and original canonical NMS/cap.
5. At frozen event threshold report counts within ±.025 separately by label and input validity. Distinguish score-scale variation across independent producer fits from a proven train→health distribution shift: health is not read, so the latter cannot be established here.

## Budget, implementation, verification

One CPU process, no GPU or fits, max15min wall, new output<=10MiB, no cache/model artifacts. New `stability_audit.py` and focused tests under existing representation_v2 research folders. Tests first for reliability bin endpoints/empty groups, corrupted fusion/prior/overlapping ancestry rejection, fixed-offset counts with missing truths retained, and unsafe input filename rejection. Input files must equal committed HEAD blobs; clean tree and protocol/code commit before run; derived JSON hashes and source identity after run. All original S2 files immutable.

Output `outputs/challenger_70/representation_v2/amendment_a3_stability/` with plan, summary/profile JSON and report. No formal performance field beyond null in registry. Report evidence and one narrow next hypothesis only; no automatic challenger run based on this descriptive audit.

Implementation ledger: implement and test parser/profiles/sensitivity, read-only terra/medium independent scale analysis, commit protocol/code, run audit once, update README/registry/report and commit. Reuse previous full-repository regression (646 passed) as historical evidence; rerun complete research tests plus relevant policy/provenance tests for this isolated read-only addition, explicitly not a new full-repository pass. No production code changes or disposable planning folders.
