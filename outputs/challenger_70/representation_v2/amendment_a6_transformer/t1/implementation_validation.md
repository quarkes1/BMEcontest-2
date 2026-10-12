# A6 T0/T1 implementation validation

Before real training (2026-10-12):

- Baseline whole-repository tests: 799 passed, 5 skipped, 31 warnings.
- Final whole-repository tests: 838 passed, 5 skipped, 31 warnings (1033.50 seconds, exit 0).
- New Transformer contract/timeline/cache/head/training/probe/report tests: 39 passed.
- All representation research tests: 279 passed.
- Independent Luna review completed; two important findings fixed with RED-to-GREEN tests: T0 reads metadata only and raw checksums are lazy after subject authorization; active encoding/training elapsed time counts toward budgets.
- User confirmed FD-I and competition cohorts are independent. Anonymous dataset IDs cannot independently verify this assertion.
- No new outer evaluation, release change, or dist change is authorized by this experiment.

Execution rulings: existing research branch retained as approved; pytest cacheprovider disabled because the existing root cache is inaccessible; baseline collection preceded additive implementation and the final full suite verifies the complete implementation. T0 records file metadata, while raw coverage/gaps/checksums are checked only when authorized sessions are encoded. Consequently malformed raw data can stop a later split rather than the initial audit. No deferred review findings remain.
