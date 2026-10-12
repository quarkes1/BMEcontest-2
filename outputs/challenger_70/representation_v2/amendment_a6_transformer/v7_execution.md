# V7 execution record

2026-10-12, bounded extension approved by user. Protocol: `docs/challenge_070_amendment_a6_v7_validation.md`.

- Implementation commits: `26dedb5`, review clarification/tests `6b76185`, canonical coverage fix `e670254`.
- First eight feature tests observed RED before implementation. Coverage-fix test also observed RED, then GREEN.
- Latest targeted result: 13 passed. Latest full representation research suite: 292 passed, 20 existing/expected warnings, 15.26s.
- Independent Luna/medium review: no concrete OOF/refit/inference/metric bug; Important wording issue about V7 stat access resolved by disclosing allowed metadata-only checks versus prohibited pre-freeze raw-array reads. Added partial/tampered-seal guards before deserialization/raw runtime. Reviewer did not judge unrun training quality/runtime feasibility; these remain empirical checks, not waived requirements.
- Read-only real-input checks: committed T24 OOF ancestry/fusion/coverage passes; T1 source/spec/plan/release attestation remains unchanged. Sensor index has 908 sessions for these 31 people; canonical outer-train feature roles cover 890, all cached. Registration uses canonical roles, not the entire index.
- First whole-repository regression was interrupted around 42% for the registration coverage fix. It is not a passing run. Its log is `v7_full_regression_interrupted.log`.
- Final-code whole-repository regression completed: **851 passed, 5 skipped, 31 warnings, 1145.64s**, session 26574. Log: `v7_full_regression.log`.
- Follow-through session 54024 stopped before registration because its summary regex did not accept the extra elapsed-time suffix `(0:19:05)`. The full passing summary was inspected, no repeat test was claimed, and registration explicitly continued in session 2538, exit 0.
- Registration/audit committed in `20ca75c`; final models and policy frozen/committed in `2b4733f` before V7 inference; results committed in `f0588cf`. Exec session **63692** completed exit 0. No worker remains to advance this attempt.
- Final state COMPLETE: exactly 10 producer / 2 base / 2 neural fits, 2000 optimizer steps. Neural/refit+inference phase 231.66s, encoding/cache phase 375.21s, producer/base phase 17.46s; phase elapsed times are not measured GPU-active time. Token cache ~14.26MiB remains ignored and bounded, necessary local model/score evidence retained.
- V7 covers all 7 subjects and 33 GT. b/c/t F1=.745763/.689655/.724138; TP/FP/FN=22/4/11,20/5/13,21/4/12. Primary delta=-.056108, gate FAILED; secondary delta=-.021625. Do not reselect model/policy on V7 or automatically proceed to outer.
- Canonical short-event recall is 4/10 for all arms; candidates reach 32/33 GT. TCN:0 improved,2 worsened,5 tied; Transformer:0 improved,1 worsened,6 tied. T24 gains were not reproduced here; this does not establish that every temporal model or transfer method is ineffective.
- Closeout verification passed: registered code/spec/input hashes and protected release unchanged; frozen model/policy hashes verified; persisted score checksum verified; canonical event counts recomputed from predictions and truths match summary; seven-subject coverage and exact fit counts checked.
- README updated. No outer, A5 restart, dist or release change. Do not call this development validation a formal .70 result.
