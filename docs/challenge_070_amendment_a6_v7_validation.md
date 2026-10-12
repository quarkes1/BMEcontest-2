# A6 V7 — frozen primary TCN validation

2026-10-12. User approved the bounded continuation: incumbent baseline, TCN primary, Transformer secondary; no outer or dist change.

## Registered protocol

ID `RV2-A6-V7-001`; outputs `outputs/challenger_70/representation_v2/amendment_a6_transformer/v7`.

Use exactly the T24/V7/outer/source identities of committed T1 manifest at `5e0b505`. TCN is selected as primary from T24 evidence; Transformer is a predeclared secondary comparison, never a replacement selected using V7. Report all three arms even when primary fails. No pooling rerun, fusion search, source training, new proposal, FD expansion or localization change.

T24 policy input consists of the three committed `wrapper{k}_health_scores.json` from T1. They cover 24 distinct held-out subjects; validate their learned ancestry, source identity, candidate ownership, row alignment, invalid fallback and fixed .75/.25 fusion. Reusing these already-examined scores is policy selection on development data, not new validation. Select one policy per b/c/t using unchanged `choose_policy` and all T24 GT. Hash inputs and policies before V7 inference.

Final refit uses T24 only: four-fold macro/micro candidate OOF (8 fits), micro threshold from T24 OOF, 116D Context-v1, LR/LGBM equal blend (2 fits), c/t frozen-FD readouts (1000 steps each, effective batch 16, AdamW .001/.01, clipping 1, FP32). The paired refit seed is `20261024`; source checkpoint and input/architecture are exactly T1. Fit two final macro/micro producer models (2 fits), persist those models, base verifiers, readouts, train-only moments and policy. Thus total new fits = **10 producer, 2 base, 2 neural, 2000 steps**. No early stopping or checkpoint selection.

The final models, policy, training subject sets and registered hashes must be committed before V7 score/data evaluation. V7 predictions use only those persisted models; no `.fit` in inference. Freeze score file before using GT for metrics. Stop on existing output, tampering, nonfinite results, provenance mismatch or incomplete seven-subject coverage; no overwriting or retries.

## Historical exposure and loader boundary

V7 is held out from recent representation fitting, not a pristine independent cohort: the competition population was previously included in baseline/challenge CV. Also the inherited `load_outer_train_only` computes eligible truths using session timestamps and validity masks before callers slice to T24. Consequently previous paths could read V7 timestamp/mask arrays and build its truths in memory; this was not V7 embedding/model training or V7-based policy selection. Do not describe V7 as entirely unread.

The new loader filters all macro/micro batches by allowed sid **before** inherited eligibility/raw-session processing. Source metadata maps may include other subjects, but returned batches/truths and session accesses must be limited to the authorized subset. Existing shared NPZ feature tables and canonical metadata tables are physically loaded wholesale and filtered; disclose this, do not claim byte isolation of those shared tables. V7 session access is permitted only after the committed final-model freeze marker passes hash checks.

## Gates and interpretation

Primary gate: V7 aggregate event F1(c) - F1(b) >= .01. Report t-b and t-c as secondary; secondary success cannot override primary failure. Passing only supports requesting a separately approved complete outer evaluation, not release or proof of .70 formal F1. V7 has seven people and potentially few events; report exact counts without significance claims or adding hidden gates.

Mandatory diagnostics: TP/FP/FN, PPV/recall, canonical <10min recall, candidate reachability, all seven per-subject counts, GT-subject median/IQR/P10 and improved/tied/worsened counts; no-GT subjects and their FP remain included. Preserve predictions and score provenance.

## Efficiency, budgets, files

Do not increase effective batch or change precision during this validation. Reusing existing T24 OOF evidence is the principal safe speedup. T1 packing implementation stays unchanged. Profiling of packing/transfer/compute is a separate follow-up, not an unverified speed claim.

CUDA when available; 4 CPU threads. New run wall <=4h, neural <=2h, encoding <=1h, CPU <=1h, per-head <=20min; cache <=2GiB, evidence <=256MiB, free disk >=20GiB and CUDA allocation <=80% starting free. Check active work in limits. Runtime violations produce INCOMPLETE, no gate. T1 cache ~11.4MiB may remain but is not silently trusted without matching hashes; no source raw copies or root-level scripts.

New modules under `scripts/research/representation_v2/transformer/`: `v7_contract.py` (seal/OOF validation), `v7_models.py` (authorized loader and persisted producer/base pipeline), `v7_probe.py` (stages/refit/report). Tests in `tests/research/test_representation_v2_transformer_v7.py`.

Write tests first: pre-freeze blocking; model tamper/partial freeze; OOF duplicate/leaked ownership; subset filtering before eligible truths; inference independent of GT; subject completeness and no-GT FP; secondary cannot pass primary gate. Run RED then GREEN, existing representation tests, whole-repository regression before real training. Register committed code/spec/source/input hashes, run train, commit freeze, run evaluate, verify counts/protected artifacts, update README, commit evidence. Retain necessary models/scores, bounded reusable cache only. A5 remains unstarted.
