# Representation v2 status

P0 status: `P0_READY`. No model has been fitted and no outer evaluation has been run. The old FD checkpoint is excluded; FD-I source data is present. Five outer-train wrappers contain 32/33/33/33/33 subjects. The full topology would use at most 450,000 optimization steps if all gates passed. A 480-token legacy-encoder-only forward pass averaged 1.28 ms on CUDA; this is a limited feasibility check, not a full training estimate. P1 must measure its real per-step and input-processing costs before P2.
