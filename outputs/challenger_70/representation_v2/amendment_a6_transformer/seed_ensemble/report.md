# A6 fixed three-seed ensemble

T24 nested development evidence only. No V7/outer/dist/release change.

| Arm | F1 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: |
| b | 0.721311 | 66 | 20 | 31 |
| p | 0.739583 | 71 | 24 | 26 |
| c | 0.760870 | 70 | 17 | 27 |
| t | 0.752688 | 70 | 19 | 27 |
| single_c | 0.763819 | 76 | 26 | 21 |
| single_t | 0.760417 | 73 | 22 | 24 |

Primary TCN gate: {'primary': 'c', 'passed': False, 'delta_vs_incumbent': 0.03955808980755526, 'delta_vs_single': -0.0029495302599955586, 'nonworse_wrappers_vs_single': 1, 'outer_authorized': False}

Secondary Transformer: {'arm': 't', 'nonworse_wrappers_vs_single': 1, 'delta_vs_single': -0.007728494623655879, 'delta_vs_incumbent': 0.03137669663317466}
Wrapper directions versus single: {'c': [True, False, False], 't': [True, False, False]}

b/p are unchanged original controls; c/t are within-partition three-seed probability means.
This does not isolate train-size/composition effects or establish pristine generalization.

## Health candidate / seed diagnostics

| Wrapper | Arm | AP | BCE | Valid mean seed std | True motion mean | Negative motion mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | c | 0.916770 | 0.181987 | 0.038511 | 0.762023 | 0.099676 |
| 0 | t | 0.920047 | 0.174203 | 0.044969 | 0.733810 | 0.067878 |
| 1 | c | 0.797419 | 0.203083 | 0.025428 | 0.464927 | 0.051094 |
| 1 | t | 0.797426 | 0.207340 | 0.075335 | 0.540581 | 0.081441 |
| 2 | c | 0.701495 | 0.231771 | 0.033398 | 0.539046 | 0.064455 |
| 2 | t | 0.714814 | 0.228731 | 0.043828 | 0.499781 | 0.035324 |
