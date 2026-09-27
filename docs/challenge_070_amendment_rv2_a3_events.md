# A3-S2 — Frozen FD event complementarity

## Purpose and authority

ID `RV2-A3-S2-001`. User authorized autonomous inner exploration and continuation after S1. Establish whether FD knowledge improves event detection beyond a controlled 116D verifier baseline, not just candidate AP. No outer access, dist/release writes, promotion, source retraining, hyperparameter sweep, or checkpoint copying. Existing research branch and workspace retained under user authorization.

## Registered design

Use the same original inner T24 subjects and three GroupKFold health partitions as S1. Original V7 and outer remain excluded. Repeated development population, not pristine generalization evidence.

For each 16-person T / 8-person H split:

1. Partition T into three policy folds. Independently generate candidates with `build_candidate_ledgers` on policy-train and policy-validation. Every macro/micro refit, OOF producer, micro threshold, imputer, scaler and verifier excludes the corresponding policy-validation subjects. No reuse of T-wide candidate features for this policy OOF.
2. In every policy split, fit baseline LR C=.1 and LGBM using existing registered estimator factories on the 116D train candidate ledger. Fixed blend .5 LR + .5 LGBM. This is a controlled baseline, not a replay of the deployed model's selected blend.
3. Independently fit the S1 subject-balanced C=.1 linear readouts for R and F on valid 133D motion features. Invalid candidates receive that partition's training-valid subject-balanced prior. Freeze both encoders, same unitfree/no-gravity-alignment inputs and checkpoint as S1.
4. Score three arms: B=baseline; R=.75B+.25random-readout; F=.75B+.25FD-readout. Fixed weights; no score calibration, feature concatenation, weight search or clipping except stable sigmoid of readout logits. Chosen before S2 outcomes to limit capacity and search surface; not claimed optimal.
5. Concatenate independently generated policy-validation candidates/scores. For each arm, select among the original admission grid nms=(.3,.5,.7), admission=(.2,.35,.5,.65), cap=(3,4,5,6,8). For each admission choose the event threshold using existing `select_event_threshold` on policy-OOF truth only. Rank by event F1, PPV, fewer admitted candidates, earlier admission grid order, higher event threshold. Prefer admission recall>=.8; if none feasible, first maximize admission recall. No post-admission subject cap. Exact >= ties and canonical production NMS maintained.
6. Freeze each policy. Refit producers and all verifiers on T, score H once, apply frozen policy, then consult H truth only for evaluation. All ancestors exclude H and original V; source ancestors use only FD train27 and exclude FD val7.

Three health folds x (three policy + one final split): 120 base candidate fits, 24 baseline-verifier fits, 24 motion-readout fits, 0 neural optimizer steps. Base candidates shared across arms within a split; no label-conditioned selection. Candidate/AP/BCE scores include invalid fallback rows. Event matching uses existing one-to-one IoU>=.25 metric, with all H truths including unreachable meals. Report pooled counts and F1, per-fold and per-subject counts/F1 and paired changes. Policy-fit F1 is selection evidence only, not headline validation.

## Gate and interpretation

`S2_SUPPORTS_WIDER_INNER_EVENT_VALIDATION` only if health pooled F1(F)-F1(B)>=.015, F1(F)>F1(R), at least 2/3 health folds improve over B, TP(F)>=TP(B), and either FP(F)<FP(B) or TP gain>FP increase. All integrity/resource checks must pass. Otherwise `S2_NO_SUPPORTED_FIXED_FUSION`; no post-hoc weight tuning. Three folds cannot authorize outer: next broad inner gate additionally needs approximately .78 aggregate F1, >=.015 gain over matched baseline and majority direction before any outer request. No automatic outer even if S2 exceeds .78.

Negative S2 only rejects this fixed score fusion, not every possible feature-based complementarity. A positive result supports a next preregistered wider development evaluation, not a new official score.

## Resources and artifacts

Max3 wallh, 1h encoder inference, free disk>=20GiB, GPU memory reserve>=1280MiB and peak<=80% starting free memory; original total24GPUh/48wallh limits remain. Use CUDA if available and CPU threads4. No persistent token cache or new neural checkpoint; output<=2GiB. Abort with partial evidence on failure, no overwrite/resume under same ID. Save code/spec/source hashes, config, policies, OOF and health score/event ledgers, subject lists and ancestry graphs. Output `outputs/challenger_70/representation_v2/amendment_a3_events/`; source SHA pinned to S1's `c6276837b1366117075e7e8a7504e084926cbfc5d80b98b863e442255995efc4`. Before/after protected snapshot must match.

## Implementation plan and ledger

Use executing-plans and TDD inline. User's autonomy authorization replaces repeated approval pauses; evidence stays in this single task document and output report, not disposable duplicate planning folders.

- [x] Add `scripts/research/representation_v2/event_probe.py` and `tests/research/test_representation_v2_event_probe.py`. Tests first: fixed fusion arithmetic, invalid fallback retained, policy roundtrip and empty predictions, gate TP/FP/majority conditions, forbidden ancestor, real-estimator finite predictions with missing116D features. Existing source loader tested separately.
- [x] Implement explicit nested split orchestration using existing crossfit, unitfree, readout, candidate-control and event-matching functions. Health truth never passed into policy selection. Persist complete policy-selection OOF evidence and source/target ancestor graph; final scoring receives frozen policies only.
- [ ] Run research regression and full pytest suite; independent terra/medium read-only review before experiment. Commit code and protocol with clean tree before running.
- [ ] Run once, replay saved gate/counts, inspect source/protected identity and disk budget; update README/registry/report and commit. Preserve negative evidence and leave no temporary scripts/token caches.

Review focus: overlapping session/candidate duplicates; all-invalid motion candidates; empty admitted set; labels from H leaking into policy choice; actual fit ancestry not merely declared partition sets. Tests use synthetic real estimator inputs and real candidate policy primitives. Runtime graphs are checked for each split and final policy ancestry.

Ruling: fixed late fusion instead of high-dimensional concatenation — reduces overfitting and avoids learning fusion on insufficient strict OOF rows; cost is that a negative cannot rule out representation-level fusion. No outcomes from S2 informed this choice.

Preflight: six new tests RED (module missing) -> GREEN; research suite 101 passed. Full pytest started and remains pending at code registration. Independent terra/medium read-only review found no blocking defects. Minor deferred: an explicit empty-input `choose_policy` test is absent; empty decoded output is tested and selector handles empty arrays. Ruling: full integration suite can finish alongside the inner probe after research regression and independent review pass; no completion claim before its final result, and any failure will be recorded, not hidden.
