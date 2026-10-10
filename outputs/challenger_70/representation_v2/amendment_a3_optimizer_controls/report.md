# S10 — train-only optimizer controls

Status: `S10_OPTIMIZER_SELECTED`; chosen: **larger_batch**.
This is training optimization evidence, not event F1 or generalization evidence.
No health scoring, outer evaluation, dist change, release change or promotion.

Code: `973698acff35167c29d2864839818dd50baa94ac`.
Same S9 eleven-subject partition: 272 valid candidates (41 positive), 72 invalid.
Eight OOF producer fits, two initial LR fits, zero event verifier fits.
Both arms completed the registered 1000 steps without early stopping.

| Configuration | Fixed subject-weighted train BCE | AP | Subjects non-worse | Fixed source BCE | Target examples | Adaptation seconds |
|---|---:|---:|---:|---:|---:|---:|
| Initial encoder/head | .221838619 | .764366346 | — | .223644453 | — | — |
| No clipping, batch4 | .213448633 | .773029713 | 8/11 | .221256667 | 4000 | 35.03 |
| Clip norm1, batch32 | .202209279 | .782963047 | 9/11 | .219083320 | 32000 | 216.84 |

No clipping improves BCE by .008389985, below the fixed .01 gate, so it is rejected.
Batch32 improves BCE by .019629339, improves source BCE by .004561132,
and meets the majority-subject requirement. It is selected by the preregistered
first-eligible rule, not a search for the lowest training loss.

Interpretation: the original adaptation's poor optimization is not inevitable;
batch32 produces a better fixed training objective. This does not establish that
gradient noise alone caused the problem: candidate exposure increased eightfold.
It also does not establish transferable representation or higher event F1.

Next: S11 evaluates **only** batch32/norm1 through the unchanged nested event
protocol with frozen-FD and incumbent controls. All policies must be frozen before
health scoring; failure closes this recipe rather than adding health-selected
optimizer variants.

Wall time 343.63s; peak allocated CUDA memory 1488131584 bytes (~1.39GiB),
recorded for the whole pilot, not each arm. Persistent token cache: zero.
Source identity, initial readout replay and protected release checks passed.
Selection binds plan, comparison and both adaptation files by SHA-256.
Full repository regression is still running; 145 representation tests passed.
