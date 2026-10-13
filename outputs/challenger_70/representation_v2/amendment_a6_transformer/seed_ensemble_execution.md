# A6 fixed three-seed execution record

2026-10-13. User approved bounded design and real experiment. Binding protocol: `docs/challenge_070_amendment_a6_seed_ensemble.md`. Formal release/dist unchanged; no V7/outer evaluation.

- Implementation `2ad412c`; review fixes `034561b`. Original8 tests observed RED→GREEN. Three review regression tests observed RED→GREEN; targeted total11 passed. Representation research suite309 passed,20 known/expected warnings.
- Luna/medium read-only review found source token cache needed pre-registration hash anchoring, explicit policy OOF coverage, and Transformer paired wrapper diagnostics. All addressed before registration/training. Minor diagnostic rollup also included. Reviewer did not run training or judge quality/generalization.
- Earlier full-suite session40041 was intentionally stopped before applying review fixes (exit1), not a passing run. Final-code full regression session66265 completed exit0: **868 passed,5 skipped,31 warnings,1043.60s**. Actual complete output inspected; log `seed_ensemble_full_regression.log` is ignored. Registration/training gate now passed.
- Read-only runtime preflight session6635 passed:24 T24 subjects,696 canonical sessions,97 eligible GT,227 existing source-only cached sessions. Source/header identity checks pass. Shared feature tables physically loaded then filtered before raw eligibility processing. No V7/outer returned subjects, no training fits.
- Planned new work:120 producer/24 base/48 neural fits,48000 steps. Original-seed checkpoints and score evidence reused only inside their own registered partitions. Source token header+NPZ hashes/inventory anchored in registry, checked before/after phases; no new token cache.
- Registration completed exit0 in session25026, committed `56b225a` after full regression. Source token header+NPZ inventory/hash and all code/spec/evidence inputs sealed. Worker session80370 started policy then health sequentially, with explicit successful phase commits between them; any failed subprocess stops the chain. Log `seed_ensemble_train.log` is ignored.
- First policy partition candidate/base/normalization/original-checkpoint reproduction passed before new training; first c fit reached step900. This is liveness evidence, not model performance. State POLICIES_FROZEN_RUNNING means policy training in progress, not frozen complete.
- Remaining: policy phase/commit frozen policies → health phase/report → integrity/budget/count verification, README closeout and commit. Stop on INCOMPLETE; do not overwrite/retry an attempt or change scientific settings.

Entrypoint: `D:/Anaconda3/envs/bme/python.exe -m scripts.research.representation_v2.transformer.seed_ensemble --root D:/BMEtest --stage register|policy|health`.

## Completed closeout (2026-10-13)

The preceding bullets retain chronological startup evidence. Training is now complete: policy evidence committed as `e5a8226`, health evidence as `294b59a`; worker log ends with both stages successful. State is COMPLETE, outer_calls=0.

- Fresh read-only `registered(root)` verification passed after completion: registered code/spec/input hashes, exact token inventory and protected release snapshot unchanged. Exact fit/step totals independently asserted:120 producer,24 base,48 neural,48000 steps.
- CUDA neural fitting cumulative1013.824s (16.90min); CPU fitting125.254s; sequence organization33.351s. These are component times, not total experiment wall time. Peak recorded CUDA allocation322216448 bytes. Output evidence/models22.45MiB; no new token cache. Retain registered checkpoints and evidence for reproducibility; no disposable new dataset/cache to remove.
- T24 incumbent F1 .721311 (66/20/31); original single TCN .763819 (76/26/21), ensemble TCN .760870 (70/17/27); original single Transformer .760417 (73/22/24), ensemble Transformer .752688 (70/19/27).
- Primary gate FAILED: ensemble TCN exceeds incumbent by .039558 but trails original single by .002950, with only1/3 wrappers non-worse (requires2/3). Secondary Transformer also trails original single by .007728 and is non-worse in only1/3 wrappers. No substitution of secondary for primary, no V7/outer evaluation, no promotion.
- Interpretation: probability averaging reduces false positives but sacrifices true events; it did not retain the original single-model event gain. This is evidence against adopting this fixed ensemble as the repair, not proof that seed variation is irrelevant or that the V7 cause is isolated. Training size/composition, sample exposure under fixed steps, and probability/policy transfer remain confounded. Any follow-up should be separately registered and confined to training-side evidence; no post-hoc policy tuning in this attempt.
- Formal release F1 remains .6514285714; dist/release unchanged. README updated; original registered protocol intentionally preserved as the historical contract.
