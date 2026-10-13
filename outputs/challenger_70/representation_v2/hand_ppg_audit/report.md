# Wearing-hand and PPG feasibility probe

2026-10-13. User-approved feasibility investigation. NO TRAINING, NO NEW OUTER F1, NO POLICY TUNING, NO DIST/RELEASE CHANGE.

## Scope and reproducibility

Read-only inline probes; no reusable model or parser implementation added. Source revision:78e4271. Inputs: `src.data.manifests.MEALS_CSV`, sensor index, A6 `t1/manifest.json`, `t1/wrapper{0,1,2}_health.json`, corresponding health_scores, and frozen `v7/summary.json` / `v7/v7_scores.json`. V7 uses already stored predictions only: no model inference or raw reads. Outer evidence is not evaluated. Shared metadata tables are read, but raw sampling is restricted to T24.

Join each stored eligible truth to the unique meal with the same indexed subject and a meal interval containing the truth interval. Normalize wearHand by removing 佩戴 and compare with dietaryHand. All97 T24 and33 V7 truths joined uniquely. These fields describe reported wearing/eating hand, not independently verified lifelong handedness. Unknown values must not be silently classified as nondominant in any future implementation.

Reuse `validation_metrics.matched_indices` on the original complete predictions/truths before grouping. This function checks against canonical TP totals; independent assertions checked each stored arm TP. Do not rematch separately by scene. Candidate reachability below means any same-session candidate with IoU>=.25, not a one-to-one candidate assignment. This is historical evaluator replay; it does not change the official >.25 contract. No per-scene FP, PPV or F1 is reported because meal metadata cannot assign scene to all negative time.

## Frozen event attribution

| Cohort | Scene | Eligible GT | Any candidate reachable | incumbent TP | single TCN TP | single Transformer TP |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| T24 | dominant | 41 | 41 | 31 | 34 | 33 |
| T24 | nondominant | 56 | 52 | 35 | 42 | 40 |
| V7 | dominant | 15 | 15 | 12 | 12 | 12 |
| V7 | nondominant | 18 | 17 | 10 | 8 | 9 |

T24 incumbent recall:dominant75.61%, nondominant62.50%. T24 TCN improves both groups (82.93% /75.00%); thus IMU is not universally ineffective on nondominant meals. V7 incumbent recall80.00% /55.56%, TCN80.00% /44.44%, Transformer80.00% /50.00%. Net TCN/Transformer TP losses relative to incumbent occur in the nondominant group; equal group TP does not establish identical matched meals. All four T24 unreachable truths and the one V7 unreachable truth are nondominant.

Nondominant share is56/97=57.73% in T24,18/33=54.55% in V7. V7 is not simply enriched for nondominant meals. Subject composition, duration, utensils, and training/refit differences remain confounded. These descriptive groups are small; no causal or significance claim.

## PPG packet probe

Selection fixed independently of signal/labels: restrict sensor index to manifest T24; sort by externalid/session_id; choose first indexed session per subject, then first lexicographically sorted collect_data file. Read first20000 data lines per file,24 subjects /480000 lines total. A valid row has PPG_TIME>0 and at least one nonzero value in columns PPG1..PPG44. Group by PPG_TIME; inspect packet size, unique44-value vectors and varying columns. No dataset cache created.

Results across all24 excerpts:

- Positive timestamp increments have median602ms in every excerpt; normal observed increments599–605ms. Some excerpts contain long gaps, notably HNU21017 (~5560s) and HNU21032 (~135s); never interpolate through these gaps.
- Each excerpt contains289–298 distinct positive timestamps and4332–4470 valid rows. Complete groups have15 rows; every complete group has15 DISTINCT44-value vectors.
- OnlyPPG1..PPG20 are active and vary within complete packets in these excerpts; PPG21..PPG44 are all zero. This is not a whole-dataset claim.
- Non-15 groups occurred for HNU21005(9), HNU21008(2), HNU21015(14), HNU21016(12), HNU21017(5,7). Fixed excerpt boundaries and acquisition gaps must be distinguished in a full audit.
- Packet arrival rate is approximately1/.602=1.66Hz. If rows are successive samples,15/.602=24.92 samples/s PER COLUMN is plausible. Neither multiplying by44 to claim~1kHz nor treating packet timestamps as~2Hz waveform sampling is justified.

These observations support a packetized waveform hypothesis, but do not prove channel/wavelength semantics, sample ordering, timestamp anchor (start/end), or reliable pulse/HRV recovery. Columns have different amplitude scales; do not flatten44 columns into one waveform. No HR/HRV, SpO2, physiological quality score or classifier was computed.

## Decision and remaining gate

Proceed with PPG feasibility work; do not start multimodal training from the old2Hz assumption. First obtain/locate the device export definition for PPG1..44,15-row packet ordering, nominal rate and timestamp anchor. If unavailable, document uncertainty and test continuity/packet-boundary sensitivity on T24 without selecting interpretations for event F1. Then audit quality and coverage over complete T24 sessions and compare reliable local PPG trends versus quality/missingness-only controls using subject-contained OOF.

First modeling scope, if quality supports it: optional quality-gated PPG candidate evidence with exact IMU fallback, not a wholesale Transformer redesign. Independently evaluate whether a PPG candidate proposal is needed for the uncovered nondominant meals; verifier-only fusion cannot recover absent candidates. Meal-side labels remain diagnostic only unless independently available at inference. No use of V7 groups to tune parameters or select models.

This completes the initial feasibility probe, NOT the full signal-quality audit or multimodal experiment. Formal F1 remains.6514285714. No new models, no raw copies, no persistent cache. Preserve old registered contracts and code unchanged; README/data notes now qualify the stale sampling statements.
