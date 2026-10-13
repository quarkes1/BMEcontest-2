# T24 PPG waveform feasibility protocol

2026-10-13. Follow-up approved by user despite unavailable official protocol. A read-only research probe, not production parser development or a classifier experiment. Freeze this protocol before signal analysis.

## Inputs and isolation

Use A6 t1/manifest.json T24 IDs only. Sensor index metadata may be read in full; filter to T24 before resolving or reading raw files. Sort externalid/session_id; choose first indexed session per subject, then first lexicographic collect_data file. Read each chosen file completely, not just the original20000-line excerpt. Exactly24 subjects, total file size1905.10MiB at planning. This deterministic panel is not all696 T24 sessions and does not estimate full-day quality coverage. No V7/outer raw, no labels, no event predictions, no refits or F1.

## Fixed analysis

- Read by column names:PPG_TIME,PPG3,PPG5,ACC_TIME,ACC_X/Y/Z. Preserve file order. PPG row requires positive timestamp, finite pair and at least one nonzero pair value. Do not deduplicate quantized rows or sort away time reversals.
- Group contiguous equal positive PPG timestamps. Complete packet=15 rows. Report packet-size distribution and timestamp increments, backwards jumps, within-packet distinct pair counts.
- A continuous segment consists of complete packets separated by450..750ms positive increments. Split on any incomplete packet, reversal or larger gap; do not bridge/interpolate gaps. This conservative fixed range is a diagnostic convention, not learned policy.
- Partition each segment into nonoverlapping50-packet windows (750 samples, approximately30s), discard partial tails and report how many rows are used. Compare assumed25Hz with local implied rate15/median_packet_interval_s; report local rate and spectral peak sensitivity to rate assumption.
- For each channel, compare median absolute boundary step against median within-packet step in file order, and after reversing samples within each packet. Report descriptive ratios; do not select reconstruction based on event labels. Zero/degenerate denominator is undefined, not zero. Compute each channel separately, never flatten columns.
- On finite, nonzero, nonconstant windows, demean and apply third-order Butterworth .7..3.3Hz bandpass (zero phase) at assumed25Hz, Welch nperseg512, peak search .7..3.0Hz. Report peak frequency, fraction of band power within±.12Hz of peak, absolute pair frequency difference, and sample correlation. Call these spectral proxies, NOT calibrated heart rate or HRV.
- Fixed pair-pass diagnostic: both peak concentrations>=.15 and pair frequency difference<=1/6Hz (10 cycles/min). This measures agreement, not physiological validity. Do not optimize thresholds.
- ACC motion proxy=std of ACC vector norm in raw rows with ACC_TIME in the PPG window support [first_packet,last_packet+600ms), finite nonzero vectors, at least100 rows. This is an approximate packet-anchor alignment and raw-unit activity measure, not reconstructed ACC spectrum. Per subject split motion by its median, report low/high agreement rates, without training or choosing thresholds from meal labels. Report missing motion.

## Resources and evidence

Four thread workers, numerical libraries one thread. At most24 full raw reads, no disk data caches or copies, no downloads. Retain small protocol/results/report only. Record input SHA256 by streaming during the same raw read, source paths relative to repo, rows, data errors, per-subject summaries. Stop on parsing/ownership errors rather than silently replacing selected sessions. No dist/release change. Compare protected git paths on closeout.

Interpretation must distinguish observational continuity, plausible pulse-band periodicity and unknown clinical validity. This phase cannot establish incremental event F1; favorable findings only justify a subsequent training-side quality/coverage and candidate-level experiment.
