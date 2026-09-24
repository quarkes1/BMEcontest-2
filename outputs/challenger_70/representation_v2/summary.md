# Representation v2 status

P0 status: `P0_READY`. No model has been fitted and no outer evaluation has been run. The old FD checkpoint is excluded; FD-I source data is present. Five outer-train wrappers contain 32/33/33/33/33 subjects. The full topology would use at most 450,000 optimization steps if all gates passed. A 480-token legacy-encoder-only forward pass averaged 1.28 ms on CUDA; this is a limited feasibility check, not a full training estimate. P1 must measure its real per-step and input-processing costs before P2.

P1 sampled 20 fold-0 training subjects in 5.38 s. The target cache uses packet timestamps, valid spans and raw sensor counts; FD-I has physical-scale acceleration. Target-only training is input-ready. FD unit alignment still requires review before a transfer arm.

Of the 20 first-session 5 s smoke slices, 17 passed the fixed 80% coverage rule. The other three stay in the candidate ledger and use the predeclared control-score fallback. Target median acceleration magnitude across sampled sessions is about 3,945–4,300 raw counts; the sampled FD-I participant was about 9.8 in its acceleration columns. No target gyro physical conversion has been validated.

P2 preparation: the FD-I source adapter is implemented with participant-disjoint loading and explicit `undocumented_raw` units. Transfer arms Q2/Q3 remain disallowed pending physical-unit verification. The dry-run identifies 20 T/V partitions, 200 shared macro/micro candidate-producer fits, and 40 Q0/Q1 branch fits (80,000 optimizer steps); potential FD arms would add 40 branch fits and one 10,000-step source fit. These are fit counts, **not** a validated wall-clock estimate. No P2 fit, probe score, P3 event score, outer score, or release change has occurred. See `manifests/p2_cost_dry_run.json`.
