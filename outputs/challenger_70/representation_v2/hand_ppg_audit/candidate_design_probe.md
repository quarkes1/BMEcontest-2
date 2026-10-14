# Candidate coverage and temporal control follow-up

2026-10-14. Read-only exploratory analyses on existing JSON evidence; no raw reads, new model, policy change or F1. These are follow-up design diagnostics, NOT retrospectively added preregistered waveform endpoints.

## Selection limitation

Intersect the24 waveform panel sessions/windows with original T1 health truths and candidate intervals across three wrappers. Sum disjoint window overlap with each interval; clip the fraction to[0,1]. Outcomes:

| Existing health evidence | Total | Any waveform overlap | At least50% interval overlap | At least50% with pair-pass windows |
| --- | ---: | ---: | ---: | ---: |
| Eligible GT | 97 | 1 | 1 | 1 |
| Candidates | 794 | 24 | 24 | 23 |

This is coverage of the previously selected24-file panel, not the availability of PPG in the remaining raw data. It proves the waveform panel cannot answer meal-discrimination questions. Do not evaluate classification performance on this selected panel or extrapolate its quality fraction to all meals.

## Fixed temporal mismatch comparison

For each subject, use original window i and window i+20, retaining pairs separated by8–12min in their stored start times. Compare same-time PPG3/PPG5 agreement against PPG3(i)/PPG5(i+20), with the identical frozen concentration and frequency-difference rule. No lag search, wrapping across session ends, or labels. Same target windows in both arms.

- Eligible pairs2742; contemporaneous pass2126/2742=77.53%; shifted pass1349/2742=49.20%.
- All24 subjects have higher same-time pass rates.
- This mismatch comparison is not an independent randomization test and does not prove a physiological source. Subject frequency baselines, autocorrelation and simultaneous movement remain possible contributors. No p-value or significance claim.

## Metadata-only candidate inventory

The12 original T1 score artifacts contain2411 candidate occurrences,2034 unique(sid,start_ms,end_ms) intervals across227 sessions and24 T24 subjects. All227 sessions resolve to one collect_data file each, total18.4202247GiB by file metadata, none missing. Health artifacts alone contain794 candidates. The full T24 manifest has696 sessions.

Only metadata paths/sizes were inspected for this inventory. All candidate subject ownership was checked against T24 before resolving raw paths. Same deterministic per-session extraction may be reused for identical raw data across partitions, but outcomes and learned transforms must remain partition-contained. Never stack all wrapper OOF files and call a randomly split meta-model strict OOF.

Recommendation: a bounded candidate-aligned PPG coverage/conditional-evidence audit precedes any classifier training. Proposed design: `docs/challenge_070_amendment_a7_ppg.md`.
