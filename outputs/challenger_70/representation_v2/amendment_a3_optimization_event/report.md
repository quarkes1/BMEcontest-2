# S11 — selected batch32 adaptation: closed

Status: `S11_NO_SUPPORTED_OPTIMIZATION_EVENT_GAIN`.
This is repeated-T24 inner development evidence, **not formal outer/release F1**.
No outer, release, dist or promotion changes.

S10 fixed training gate selected only batch32/norm1. Twelve independent nested
partitions completed 1000 steps each, with all policies frozen before health.
Code `20f4bc220fcf1354e652a91b0b367e295e20019f`; source/selection identities saved.

| Arm | TP | FP | FN | Pooled inner F1 | PPV | Recall |
|---|---:|---:|---:|---:|---:|---:|
| B incumbent116D | 66 | 20 | 31 | .7213114754 | .7674418605 | .6804123711 |
| R frozen FD251D (not random) | 69 | 20 | 28 | .7419354839 | .7752808989 | .7113402062 |
| F batch32 adapted FD251D | 66 | 16 | 31 | .7374301676 | .8048780488 | .6804123711 |

F improves B by .0161186922 with four fewer FP and no aggregate TP loss;
2/3 folds improve over B. But F remains below the registered frozen-FD control
(.7419354839), so the **whole gate fails**, despite meeting several subconditions.
No post-hoc relaxation or switching the promoted arm to R.

| Fold | B F1 | R F1 | F F1 |
|---|---:|---:|---:|
| 0 | .805555556 | .793650794 | .761904762 |
| 1 | .736842105 | .730158730 | .750000000 |
| 2 | .592592593 | .700000000 | .700000000 |

24-subject diagnostics: F vs B improves7, worsens7, ties10.
Median B/R/F .666667/.696970/.666667;
IQR intervals B [.553571,.857143], R [.520202,.888889], F [.564935,.888889];
10th percentile B .353333, R/F .12. Lower-tail robustness still does not improve.
Macro candidate AP B/R/F .787110/.804356/.807494 and BCE .225848/.210231/.208769
show ranking gains that are insufficient for the fixed event gate.

The training repair therefore helped optimization and some FP suppression but
does not establish better event representation than the frozen reference.
Close this adaptation recipe. Start the preregistered S12 frozen source-head
probability morphology test, rather than another health-selected optimizer setting.

120 producer fits, 72 event verifier fits, 12 initial readout fits, 12000 steps.
Wall3360.58s (~56min), adaptation/encoding2544.10s, all12 splits completed.
Output17762885 bytes before this report (~16.94MiB); zero persistent token cache,
zero copied checkpoint. Protected release unchanged and all resource checks passed.
All baseline ledgers, scores, policies and health event counts replayed sealed S2.
Two existing sklearn sqrt warnings did not yield nonfinite predictions.

Full repository regression after S9/S10 code freeze: **704 passed, 5 skipped,
15 existing warnings**, 1234.55s. This proof precedes any S12 implementation.
Current formal release F1 remains .6514285714.
