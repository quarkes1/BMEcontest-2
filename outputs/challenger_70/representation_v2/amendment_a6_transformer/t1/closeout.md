# A6 T0/T1 closeout — 2026-10-12

Status: COMPLETE. Results committed in `5e0b505`; all policies were committed in `3f15d8b` before health evaluation. The registered design and plan retain their historical status text to preserve the exact attested bytes; current status is in state.json, report.md and the repository README.

## Verified evidence

- 3 complete wrappers, 24 unique health subjects, 97 GT; exact 120 producer / 24 base / 36 neural fits and 36000 optimizer steps.
- Recomputed summary equals the committed summary. Source/spec/plan hashes, protected release digest and frozen policy hashes passed revalidation after completion.
- 39 targeted tests passed again at closeout (6.18s). Full implementation regression before real training: 838 passed / 5 skipped / 31 warnings, 1033.50s; all representation tests: 279 passed.
- Inner F1 b/p/c/t = .721311 / .739583 / .763819 / .760417. All new arms improve or tie on 3/3 wrappers and pass the registered wider-validation gate. These are development results, not a new outer headline or release qualification.
- Short-event recall b/p/c/t = 9/22, 10/22, 13/22, 12/22. TCN subject changes: 9 improved, 2 worsened, 13 tied; Transformer: 10 improved, 4 worsened, 10 tied.
- Measured neural elapsed 3413.24s (includes orchestration/prediction in that phase), encoding/cache elapsed 877.29s, producer/base CPU phase elapsed 125.88s. These are phase wall durations, not measured GPU-active utilization or a bottleneck profile.
- The resource ledger's reported CUDA peak is 322138624 bytes; it resets when the health process starts, so this value is not asserted to be the exact global policy-plus-health maximum.
- The small frozen-token cache (~11.4 MiB) is retained, ignored by Git and below the 2 GiB quota, for possible reuse under identical source/preprocessing identities. It contains no new raw-data copy. Necessary checkpoints/scores/provenance remain committed; temporary implementation/test/review logs are disposable.
- No outer execution, dist update, source-model training or release promotion occurred. A5 remains unstarted.

## Execution rulings (in order)

1. Work in the existing research branch as approved, avoiding large worktree copies. Cost: no isolation from concurrent edits; status checks were used before commits.
2. Disable pytest cacheprovider because the existing root cache has access restrictions. Cost: loss of pytest cache convenience, not test coverage/model behavior.
3. Allow baseline tests to continue after collection alongside additive isolated-module implementation. Cost if a baseline failed: attribution would require reproduction at the base commit. The final full suite passed.
4. Initially permit T24 raw checksums in T0 while measuring coverage at authorized encoding time. This ruling was superseded after independent review: T0 reads stat metadata only; raw checksums are computed lazily after the split authorization gate. Cost: malformed raw data may first fail a later split instead of T0. No health raw is read in its policy training partitions.

Independent Luna/medium review: two Important findings fixed with RED-to-GREEN tests (early raw-byte audit; active elapsed omitted from cumulative resource checks). No deferred minors or declined-to-judge behaviors remain.

## Proposed continuation, not executed

Prefer a fixed TCN primary comparison against incumbent on V7, with any Transformer secondary comparison declared before opening V7 outputs. Train/select policies on T24 only; freeze all policies and checkpoints before V7 inference. Do not select a winner, threshold or fusion weight on V7. Audit historical V7 exposure before describing its evidential independence. Register an explicit pass/fail gate and fit/time/cache budget before execution; outer and release still require their own approval.

Efficiency work should first profile CPU packing, transfers, forward/backward and synchronization without target validation labels. Prioritize reuse of train-only normalized tensors and batching by length while preserving candidate ownership, paired sampling, effective batch=16 and optimizer updates. Increasing effective batch, mixed precision or architectural changes are separate experiments, not invisible execution repairs.
