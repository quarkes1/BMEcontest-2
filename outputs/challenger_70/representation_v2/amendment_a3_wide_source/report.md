# S13 — full-window source encoder: closed

Status `S13_NO_SUPPORTED_WIDE_SOURCE`, complete5000 steps, fixed source gate failed.
**Source-validation development evidence only. No target event F1 was measured.**
No target signal access, outer evaluation, dist, release or promotion changes.
Implementation code `d9ea415`; checkpoint SHA
`ee62bbfb34b730125a4f0447748e9ea4480348298635f2f900af079e8c2ab4ae`.

## Comparison on the identical source population

| Metric | S0 original | S13 wider/longer-RF |
|---|---:|---:|
| Encoder parameters | 11656 | 126544 |
| Local receptive field, samples | 33 | 129 |
| Validation AP | .8440475047 | .8430429427 |
| Raw-logit BCE | .2570439361 | .5041640489 |
| ROC AUC | .9509310629 | .9472899378 |
| Brier | .0775582818 | .0899436198 |
| Positive with p<.01 | 3 | 27 |
| Negative with p>.99 | 0 | 53 |

Registered AP threshold .8540475047 is not met. BCE threshold .2770439361
is not met either. The source sampler-prior BCE .6182658045 is still beaten,
but passing that subcondition does not override the full gate.
S13 AP-S0 = -.0010045620; raw BCE-S0 = +.2471201129.

Training final100-step mean loss .0466568898, while source-validation BCE is
.5041640489. Logits span[-76.11805,38.97601], versus S0[-15.97675,6.18279].
This pattern is consistent with overfitting/overconfidence under this fixed
architecture/data/budget, not evidence that capacity alone improves transfer.
Clipped log-loss .4791963903 understates the raw-logit BCE; use the raw metric.
No early checkpoint, source-val temperature or alternative seed was selected.

Seven validation participants: AP defined for6, improves3/worsens3;
BCE worsens for all7. AP median S0/S13 .884365/.852880,
IQR S0[.825280,.928556], S13[.828954,.894303], p10 .760505/.819128.
BCE median S0/S13 .243260/.471275,
IQR S0[.225012,.287018], S13[.430711,.626597], p10 .199818/.343881.
The single-class participant has undefined AP/AUC, retained as null, not zero.
Distribution statistics are diagnostics, not post-hoc promotion criteria.

## Integrity and resources

S0 plan/population/validation/summary and preprocessing helpers bound to29a625e.
Source file fingerprints unchanged; source27/7 partition, population and all
2274 held-out participant/start/label identities replay exactly. S0 frozen
validation logits replay with **maximum absolute difference0**. Training uses
8847 tokens; class/subject draws equal the original S0 draws. Validation raw
signals never enter the optimizer or training priors. No new target data read.

All finite/isolation/optimizer/resource/protection checks passed. A complete
negative result, not incomplete. Training65.88s; wall92.44s, including preparation
and integrity checks; CUDA peak105892352 bytes (~101MiB),5000 steps,batch128.
Output1301542 bytes before this report (~1.24MiB); token bank only in memory.
One small final artifact retains initial random and final trained weights;
no duplicate raw source, token cache or temporary checkpoint remains.

22 new tests and184 related representation tests passed before training.
Full repository regression is running; its final result will be appended to
the spec/README before this stage is declared sealed.

## Decision

Close this exact architecture/budget. Do not start its target event probe,
do not retune source-val steps/seed/temperature to bypass the registered gate.
The result does not disprove all longer-receptive-field or larger models.
Next design should prioritize input-contract resolution and broader pretrained
representations, or a separately registered multi-scale/regularized route;
do not reactivate S12 morphology-threshold search.

Formal release remains run160afaf81debf1ee, event F1 .6514285714.
