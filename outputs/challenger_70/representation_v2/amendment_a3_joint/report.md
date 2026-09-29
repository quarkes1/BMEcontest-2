# S5 — Frozen representation joint verifier: rejected

Status `S5_NO_SUPPORTED_JOINT_VERIFIER`. Protocol `RV2-A3-S5-001`; preregistration code `8756209`, lossless replay correction `923ef1d`. This is repeated T24 inner-development health evidence, **not outer performance**. Formal release F1 remains **0.6514285714**, release/dist unchanged.

## Event result

| Arm | TP | FP | FN | Pooled health F1 |
| --- | ---: | ---: | ---: | ---: |
| B: original 116D | 66 | 20 | 31 | 0.7213114754 |
| R: 154D with frozen random encoder | 70 | 27 | 27 | 0.7216494845 |
| F: 154D with frozen FD encoder | 59 | 13 | 38 | 0.6982248521 |

F loses 7 TP while removing 7 FP; F1 delta versus B is -0.0230866233. It is below R, TP decreases, and only 1/3 folds improves. The preregistered gate fails; no promotion, outer call, post-hoc threshold change, or retry with different pooling.

| Health fold | B TP/FP | R TP/FP | F TP/FP | B F1 | F F1 |
| --- | --- | --- | --- | ---: | ---: |
| 0 | 29/11 | 31/17 | 23/6 | .80556 | .75410 |
| 1 | 21/5 | 23/7 | 21/4 | .73684 | .75000 |
| 2 | 16/4 | 16/3 | 15/3 | .59259 | .57692 |

Per-subject F1 versus B: 5 improved, 7 worsened, 12 tied. All 97 GT retained in event denominators, including unreachable events.

## Representation versus event outcome

| Candidate metric (macro over 3 health folds) | B | R | F |
| --- | ---: | ---: | ---: |
| AP | .78711020 | .80618729 | .80847599 |
| BCE | .22584805 | .22366989 | .21627322 |

F candidate AP exceeds B in 2/3 folds, but exceeds R in only 1/3 folds. Its macro AP advantage over R is just .00229. Better candidate ranking/BCE does not imply a better selected event operating point. The previous S1 stand-alone FD representation benefit does not establish complementarity with the 116D baseline.

This experiment changes both pooling (global 32D rather than ordered 128D) and integration. It therefore rejects this **specific joint scheme**, not all feature-level integration or transfer learning. It cannot isolate whether temporal information loss, redundant representation, verifier fitting, or policy selection causes the event loss.

## Execution and integrity

- Source checkpoint remains the pinned S0 artifact; frozen random and FD encoder; zero neural optimizer steps.
- Strict nested producers: 9 policy splits first, then 3 final health refits; 120 producer and 72 verifier fits in the successful attempt.
- All three policy files written and hashed before any health scoring/reference read; all files and marker reloaded/verified before each health fold.
- B reproduces sealed S2 candidate identities, labels, groups, scores (absolute tolerance 1e-12), exact policies, event predictions/counts, candidate metrics and subject metrics.
- Successful run wall 793.53s, encoding 24.83s; 12 completed splits. CPU4 threads, frozen encoding CUDA. Output before this report: 12,245,537 bytes (~11.68MiB). No token cache, copied checkpoint or new model retained.
- Protected release snapshot unchanged; no outer calls. Budget limits passed. These times do not include test execution and the earlier failed start.
- First start stopped at first policy replay because in-memory probability tuples were compared directly with JSON lists. A real-CandidateEvent regression reproduced it; lossless JSON canonicalization fixes comparison without loosening values. Failure plan/state retained in sibling `amendment_a3_joint_failed_serialization/`. No health was read. Extra first-partition work: 10 producer fits and 6 verifier fits; combined total 130/78 fits. Its encoder timing was not persisted, so no invented cumulative timing is reported.
- 184 relevant research/candidate/event tests passed, 4 expected all-missing-column imputer warnings; not a new full-suite run. terra/medium independent review found no important blocker. Optional per-health marker SHA duplication deferred; frozen manifest and summary SHA are retained.

## Decision and next hypothesis

Do not expand this 154D joint branch or tune it on these health folds. S2/S4/S5 together do not support additional ad-hoc averaging/fallback/concatenation variants. Before spending another training attempt, the useful question is whether frozen FD evidence adds **conditional event discrimination beyond the original verifier**, rather than stand-alone candidate AP. Any follow-up should use policy-training OOF only to examine paired ranking and event selection stability, then pre-register one narrow change if justified. Do not reuse health errors to invent rescue rules. A larger encoder or Transformer is not warranted by this result alone.
