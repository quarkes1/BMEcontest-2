# T24 PPG waveform feasibility results

Completed and verified2026-10-14. Protocol frozen before analysis in commit28c34b6. This is a waveform feasibility study, not an event-classification experiment. No training, no labels, no V7/outer raw, no event F1, no release change.

## Main result

The observed PPG3/PPG5 streams are consistent with temporally ordered packetized signals near25 samples/s per column. Pairwise pulse-band spectral agreement is common and higher during lower ACC activity. This supports further quality-gated research; it does not establish wavelength identity, heart-rate accuracy or incremental eating-detection benefit.

| Quantity | Observed value |
| --- | ---: |
| Subjects / full files | 24 /24 |
| Raw rows read | 11,769,328 |
| Bytes read | 1,997,645,810 (1905.10MiB) |
| Valid PPG pair rows | 2,574,480 |
| Rows in analyzed complete windows | 2,553,750 (99.19% of valid pair rows) |
| Nonoverlapping50-packet windows | 3,405 |
| Nominal analyzed waveform duration | 28.375 hours at assumed25Hz |
| Pair spectral agreement flag | 2,635/3,405 =77.39% |
| Lower-motion half, per-subject median split | 1,436/1,708 =84.07% |
| Higher-motion half | 1,199/1,697 =70.65% |
| Subjects with higher low-motion agreement | 23/24 |
| Subject agreement fraction P10/median/P90 | 62.99% /78.33% /92.28% |
| Read/compute elapsed time | 169.90 seconds, four workers |

These files were selected by index order, not by meal content or signal quality. They are not the full696-session T24 dataset. The99.19% refers to valid PPG pair rows in this panel, NOT all-day wear-time coverage. Entire missing periods and unobserved meals are not represented by that fraction. A spectral flag is NOT a clinical quality certificate.

## Packet order and sampling assumption

Across subjects, the median boundary-step/within-packet-step ratio was0.9964 forPPG3 and1.0034 forPPG5. Subject ratios ranged0.9543–1.0257 and0.9730–1.0353. Reversing samples inside each packet raised subject-median ratios to13.4688 and12.8709 (subject ranges8.0498–19.6610 and8.3803–17.6381). This is strong descriptive evidence for retaining original row order rather than reversing it. It is not a proof of absolute timestamp anchor or of every sample's timing.

Local inferred rate15/median_packet_interval_seconds had P10/median/P90 all24.91694Hz. Scaling spectral peak frequencies from assumed25Hz to that rate gave peak-frequency-change P10/median/P90 of0.00292/0.00389/0.00503Hz. This is a frequency-axis sensitivity calculation, NOT a second complete filtering/reconstruction experiment. It does not resolve first-sample vs last-sample vs callback timestamp alignment.

All selected files had zero parsed malformed rows and no backward packet timestamp transitions. Gaps and incomplete packets were split, not interpolated. First/last partial packets and partial50-packet tails were excluded under the frozen protocol.

## Spectral and motion interpretation

Each channel was filtered independently; no optical channels were concatenated. The flag requires both spectral concentrations>=.15 and peak-frequency difference<=1/6Hz in a fixed .7–3Hz search band. Median bandpassed interchannel correlation was.8518 (P10=.2945, P90=.9868). Shared motion artifacts can also produce agreement and correlation; no ECG or reference pulse labels were available.

ACC activity is raw-unit standard deviation of vector norm over an approximately aligned window. Low/high grouping is descriptive within each subject and is not a deployable eating-hand classifier or a learned threshold. HNU21038 was the one reverse-direction subject (low-motion agreement64.38%, high75.34%); retain this exception, do not tune it away.

## Verification and evidence

- `waveform_results.json` retains input relative paths/SHA256, packet-size and distinct-vector counts, subject summaries and all3405 window records. Size about2.67MiB; no raw copies or new waveform/token caches.
- Input ownership was restricted to manifest T24 before raw reads. Completion checks independently asserted exactly24 unique subjects, protocol hash, raw packet row accounting, window accounting, and the fixed agreement flag on every record.
- Separate synthetic probes recovered a1.2Hz sinusoid within one Welch bin, rejected a1.2/2Hz pair as disagreement, and verified ordered/reversed packet-boundary ratios of1/29 on a monotonic synthetic sequence.
- The exploratory calculation ran inline; no new production parser, training code, external dependency or retained scratch script. This report and the frozen protocol specify the analysis; the exploratory command remains in conversation execution history. Production implementation will require a versioned, tested parser and reproducible entrypoint before any classifier experiment.
- Existing src/scripts/dist were not edited. No production test-suite result is claimed for this documentation/data-only probe.
- Fresh A6 `seed_ensemble.registered(root)` verification passed at closeout: existing registered code/input hashes, token inventory and protected release snapshot unchanged.

## Next research step

Proceed to a separately specified T24 candidate-level PPG audit, before expanding neural models. Use original training-side candidate intervals, not GT meal anchors, to sample local PPG evidence. Build a minimal tested parser that preserves within-packet order, breaks gaps, exposes missingness and retains per-channel signals. Treat25Hz and the packet timestamp anchor as explicit assumptions; assess alignment sensitivity without choosing it by event F1.

The next comparison must distinguish (1) frozen IMU evidence, (2) IMU plus PPG availability/quality only, and (3) IMU plus quality-gated PPG temporal evidence. Otherwise a benefit may just encode collection schedule or motion rather than added physiology. Avoid waveform HRV initially; use per-channel spectral/relative-change proxies, with unavailable PPG falling back exactly to IMU. Assess complete candidate/negative-time coverage, not just selected high-quality meals.

If later training is justified, all imputation, normalization, feature selection, fusion and thresholds remain training-subject-contained OOF. Meal hand labels are diagnostic only. No automatic V7 reopening, no new outer call, no dist promotion. This phase establishes feasibility only; formal release F1 remains.6514285714.
