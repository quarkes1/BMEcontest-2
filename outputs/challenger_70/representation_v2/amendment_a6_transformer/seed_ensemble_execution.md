# A6 fixed three-seed execution record

2026-10-13. User approved bounded design and real experiment. Binding protocol: `docs/challenge_070_amendment_a6_seed_ensemble.md`. Formal release/dist unchanged; no V7/outer evaluation.

- Implementation `2ad412c`; review fixes `034561b`. Original8 tests observed RED→GREEN. Three review regression tests observed RED→GREEN; targeted total11 passed. Representation research suite309 passed,20 known/expected warnings.
- Luna/medium read-only review found source token cache needed pre-registration hash anchoring, explicit policy OOF coverage, and Transformer paired wrapper diagnostics. All addressed before registration/training. Minor diagnostic rollup also included. Reviewer did not run training or judge quality/generalization.
- Earlier full-suite session40041 was intentionally stopped before applying review fixes (exit1), not a passing run. Final-code full regression session66265 completed exit0: **868 passed,5 skipped,31 warnings,1043.60s**. Actual complete output inspected; log `seed_ensemble_full_regression.log` is ignored. Registration/training gate now passed.
- Read-only runtime preflight session6635 passed:24 T24 subjects,696 canonical sessions,97 eligible GT,227 existing source-only cached sessions. Source/header identity checks pass. Shared feature tables physically loaded then filtered before raw eligibility processing. No V7/outer returned subjects, no training fits.
- Planned new work:120 producer/24 base/48 neural fits,48000 steps. Original-seed checkpoints and score evidence reused only inside their own registered partitions. Source token header+NPZ hashes/inventory anchored in registry, checked before/after phases; no new token cache.
- Remaining: full regression → register/commit → policy phase/commit frozen policies → health phase/report → integrity/budget/count verification, README closeout and commit. Stop on INCOMPLETE; do not overwrite/retry an attempt or change scientific settings.

Entrypoint: `D:/Anaconda3/envs/bme/python.exe -m scripts.research.representation_v2.transformer.seed_ensemble --root D:/BMEtest --stage register|policy|health`.

Training is not yet started. Execution state after registration is authoritative in `seed_ensemble/state.json`; completion cannot be inferred from training loss or partial step logs.
