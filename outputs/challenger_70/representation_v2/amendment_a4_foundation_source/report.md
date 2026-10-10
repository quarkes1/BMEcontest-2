# S14 — unit-free foundation source probe: complete negative

Status `S14_NO_SUPPORTED_FOUNDATION_TRANSFER`. **Source AP is not event F1.**
Code `12c33d03626652dc082e75719c22c3212264c144`; spec
`docs/challenge_070_amendment_rv2_a4_foundation.md`.
No new target raw, outer evaluation, dist/release or promotion changes.
Formal event F1 remains .6514285714 (run160afaf81debf1ee).

## Fixed comparison

| Source validation metric | R random LIMU | F pretrained LIMU | D frozen FD |
|---|---:|---:|---:|
| AP | .7642776704 | .6974671879 | .8514425742 |
| Raw-logit BCE | .2721772811 | .2939240972 | .2267743373 |
| ROC AUC | .9288550774 | .9175496443 | .9531782565 |
| Brier | .0857130908 | .0928954264 | .0685187142 |

F-R AP=-.0668104825, F-D AP=-.1539753863. F raw BCE is worse than both.
Registered AP gate is max(R+.01,D+.01,S0+.01,prevalence+.10)=.8614425742;
BCE ceiling is min(R+.02,D+.02,S0+.02)=.2467743373. Both fail.
Every integrity check passed, so this is a complete negative, not incomplete.

The 44352-parameter LIMU encoder is frozen in both R and F. Its four shared
attention/FF passes and unusual residual graph reproduce the pinned upstream
calculation. Five-second unit-free tokens are resampled to20Hz and split into
five1-second sequences, not squeezed into20 points. Per-subwindow72D means
undergo train-only PCA→32D before fixed mean/std/max→96D pooling and LR C=.1.
D uses the unchanged S0 encoder's16D output with the same LR family.
D differs in architecture/width; R is the matched pretrained-weight control.

This unit-free input removes per-token triad means/scales, unlike the upstream
physical preprocessing. The official industrial pretraining rate/participant
identities are not established by the public checkpoint. Phone/wrist domains
also differ. Negative transfer under **this** bridge is observed; which domain
or preprocessing mismatch causes it is not identified. Do not conclude that
all foundation models or a correctly calibrated HARNET model cannot help.
Do not retune rate/pooling/PCA/C or fine-tune this model against these scores.

## Participant diagnostics

AP defined for6 participants; participant23 has no positives and retains null
AP/AUC. BCE defined for all7. F vsR: AP improves1/worsens5; BCE improves2/worsens5.
F vsD: AP improves1/worsens5; BCE improves1/worsens6.

| Arm/metric | Median | IQR | 10th percentile |
|---|---:|---|---:|
| R AP | .784594 | [.771117,.830543] | .742299 |
| F AP | .772689 | [.757550,.786020] | .681311 |
| D AP | .888587 | [.835117,.929665] | .778516 |
| R BCE | .284142 | [.216970,.315983] | .198851 |
| F BCE | .284263 | [.263752,.314770] | .234935 |
| D BCE | .218764 | [.158627,.245045] | .156111 |

These are mandatory diagnostics, not new promotion criteria. The same seven
source-validation participants have been reused: development evidence only.

## Integrity / resource / verification

FD-I right wrist27/7 subjects;8847 train/2274 validation tokens,482 positives.
All source fingerprints, selected population and validation token identities
match sealed S0. S0 logits replay within registered1e-5 tolerance (maximum
absolute difference1.9073486328125e-6); batched evaluation is not bit-identical.
No validation signal fits PCA/scaler/LR, no encoder optimizer runs. All encoder
state digests remain identical before/after. Three LR fits,0 neural steps.

Pinned upstream commit bdfc3d0f734befaeb9375f5f7b8a7342f0ddfdcb; checkpoint
SHA25665697e7017020f36429a221382253f1fc370fdfd493cfb71667f95846c18516c.
MIT notice and exact reference files retained under upstream; code is reference
text only, never executed. Checkpoint loading uses weights_only=True and strict
keys/shapes/dtypes/finite values. No full clone or extra framework installation.

Whole experiment wall29.26s; peak CUDA37880832B (~36.1MiB); output869360B
before this report (<1MiB); persistent token/embedding cache0. Per-arm timings
include PCA/LR for R/F; D timing excludes prior shared FD encoding, so do not
interpret them as a fair encoder speed benchmark. Wall includes preparation.

27 new tests and211 related representation tests pass. Independent Luna/medium
review found one Important preflight-status issue, fixed via2 RED→GREEN tests;
no remaining Critical/Important/Minor findings. Post-fix full regression:
**770 passed,5 skipped,31 existing/expected warnings**,1029.36s,exit0
(torch CPU2; pytest -q -p no:cacheprovider). The pre-fix interrupted run does not
count as verification. Rebuildable research bytecode and all three own32MiB
test placeholders removed; scientific evidence retained. Stage is sealed.

Validation-row metrics, checkpoint identity and fixed gate were independently
recomputed after execution; all agree. No new source/target result was used to
change the registered experiment. No temporary checkpoint or persistent bank.
An independent no-fit replay reconstructed PCA/scaler/LR predictions from saved
JSON on all2274 held-out source tokens. R/F/D maximum absolute logit errors are
6.7862e-6/3.0640e-6/1.8193e-6, within the replay's fixed atol/rtol1e-4. All seven
upstream files match their manifest SHA/size and committed blobs byte-for-byte;
path-scoped Git attributes preserve original formatting, not reformatting it.

## Next direction, not another registered experiment

Stop this unit-free frozen LIMU bridge. Prioritize domain/input fidelity and
training-data coverage over another arbitrary backbone swap. A read-only audit
of existing S0 training population finds144975 eligible negative tokens but only
6912 selected (4.7677%), while all1935 positives are retained. This supports
investigating more diverse/hard source negatives as a hypothesis, not a proven
cause or performance gain. Any such study needs a separate preregistered matched
sampling control and fixed budget; no source-val mining. Target event validation
remains necessary before claiming an advance toward .70.
