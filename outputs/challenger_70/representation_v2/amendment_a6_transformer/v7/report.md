# A6 frozen V7 validation

TCN primary; Transformer secondary. Same historical competition cohort, not pristine independent evidence. No outer/dist/release change.

| Arm | F1 | TP | FP | FN |
| --- | --- | --- | --- | --- |
| b | 0.745763 | 22 | 4 | 11 |
| c | 0.689655 | 20 | 5 | 13 |
| t | 0.724138 | 21 | 4 | 12 |

Primary TCN delta: -0.056108; gate passed: False.
Secondary success cannot override a failed primary gate. A pass only supports requesting separately approved outer evaluation.

Per-subject, short-event and candidate counts are in summary.json; final models/policies were committed before V7 inference.

## Closeout interpretation

All seven V7 subjects and 33 GT are included. TCN changes versus baseline: 0 subjects improved, 2 worsened, 5 tied. Transformer: 0 improved, 1 worsened, 6 tied. All three arms detect 4/10 short GT; candidate reachability is 32/33. TCN loses two net TP and adds one FP; Transformer loses one net TP with unchanged FP. The T24 gain did not reproduce on this validation subset. Seven subjects are limited evidence; this is not proof that temporal representation learning generally cannot help.

Final-model/policy freeze commit `2b4733f` precedes result commit `f0588cf`. Registered code/spec/input hashes, protected release, frozen model/policy hashes, score checksum, canonical event counts, seven-subject coverage and exact fit counts passed closeout verification. Full regression: 851 passed, 5 skipped, 31 warnings; targeted tests 13 passed, representation tests 292 passed. No outer evaluation or dist update.

Candidate coverage is high, so do not interpret this result as evidence that merely adding proposals would fix these losses. A useful next diagnostic would trace the lost true candidates through frozen base/motion/fused scores, admission/NMS and event threshold, and distinguish representation failure from score/policy transfer. This is a hypothesis for a separately scoped audit, not permission to tune against V7 or rerun this validation with newly chosen rules.
