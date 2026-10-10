# S12 — frozen gesture probability morphology: closed

Status: `S12_NO_SUPPORTED_GESTURE_MORPHOLOGY`; all12 partitions completed.
Repeated-T24 inner development evidence only, **not formal outer/release F1**.
No outer, dist, release or promotion change. Code `32a886f364712feb397e416a07cba308536b8d22`.

| Arm | TP | FP | FN | Pooled inner F1 | PPV | Recall |
|---|---:|---:|---:|---:|---:|---:|
| B incumbent116D | 66 | 20 | 31 | .7213114754 | .7674418605 | .6804123711 |
| R frozen FD251D, NOT random | 69 | 20 | 28 | .7419354839 | .7752808989 | .7113402062 |
| F frozen source-head morphology140D | 68 | 23 | 29 | .7234042553 | .7472527473 | .7010309278 |

F-B is only +.0020927799, below the registered +.015. Two additional TP cost
three additional FP; F also remains below R. Two of three fold wins over B do
not override these failures. No post-hoc replacement by R or relaxed gate.

| Fold | B F1 | R F1 | F F1 |
|---|---:|---:|---:|
| 0 | .805555556 | .793650794 | .788732394 |
| 1 | .736842105 | .730158730 | .745762712 |
| 2 | .592592593 | .700000000 | .620689655 |

24-subject diagnostics: F vs B improves6, worsens5, ties13;
R vs B improves8, worsens7, ties9. Median B/R/F .666667/.696970/.708333;
IQR intervals B [.553571,.857143], R [.520202,.888889], F [.486111,.865079];
p10 B .353333, R/F .12. A higher median does not establish lower-tail robustness.
Macro candidate AP B/R/F .787110/.804356/.806585 and BCE .225848/.210231/.215156
again show that ranking improvement alone does not establish event improvement.

All nine policy partitions finished before any health scoring; frozen marker
SHA `52b99decdf6775a23815bb092f4c60debab1847cdad0e69e1db11897e3b7d434`.
B candidate/score/policy/event ledgers replay sealed S2. R candidate/score ledgers
replay sealed S8 with registered numerical tolerances. Source checkpoint and
subject ancestry retained; all health integrity/resource checks passed.

120 producer fits,72 event verifier fits,0 neural optimizer steps. CUDA used
for frozen encoding. Wall920.13s (~15.34min); encoding/scoring19.55s, which does
not include candidate production and raw preparation. Output12252197 bytes
before this report (~11.68MiB), persistent token cache0, copied checkpoint0.
Protected release unchanged. This was a complete negative experiment, not an
incomplete engineering failure. Full repository regression completed:
**721 passed,5 skipped,31 existing/expected warnings**,1207.41s, no failures.
Independent Luna/medium review recomputed counts/gate/subject diagnostics and
verified all12 B/R replay identities, policy hashes and subject ancestry; no
result-integrity blocker. Temporary/rebuildable research pyc files are cleaned;
the saved ledgers and source checkpoint remain available for reproducibility.

Close the fixed24D schema. Next design should investigate a stronger complete
5s motion representation (longer receptive field / larger encoder), rather than
retuning these probability thresholds on repeated health. This is a hypothesis,
not proof of a capacity bottleneck. External pretrained alternatives are reviewed
in `docs/challenge_070_external_pretrained_review.md`; physical-input and license
gates remain unresolved and must not be silently bypassed.

Formal release remains run160afaf81debf1ee, F1 .6514285714.
