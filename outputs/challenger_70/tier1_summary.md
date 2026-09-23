# E05/E06 inner-only challenge summary

Final state: `NO_TIER1_CHALLENGER_PASSED`

No outer evaluation, promotion, release update, model update, or dist rebuild was performed.

| Experiment | Control F1 | Challenger F1 | TP Δ | FP Δ | Fold stability | Mechanism | Gate |
|---|---:|---:|---:|---:|---|---|---|
| E05-GYRO5 | 0.7655986509 | 0.7651006711 | +2 | +4 | 2/5 improved | 6-feature centred GYRO5 block | `E05_INNER_GATE_FAILED` |
| E06-FUSION | 0.7655986509 | 0.7567114094 | -3 | +9 | 1/5 improved | 8-feature OOF logistic meta-fusion | `ADAPTIVE_FUSION_NOT_JUSTIFIED` |
| E07-COMBO | — | — | — | — | not run | both strong-pass prerequisites failed | `NOT_TRIGGERED` |

## E05

- Fixed configuration: incumbent 116D representation plus six preregistered GYRO5 features.
- Event accounting: 11 TP recovered, 9 TP lost, 20 FP introduced, and 16 FP removed; net `TP +2 / FP +4`.
- Short events: 5 TP recovered and 3 TP lost.
- The predefined sealed-outer FP mode labels do not have a leakage-safe identity mapping to the inner-OOF event universe. The run therefore reports exact inner FP event accounting but does not infer or redefine FP modes.
- Result: the representation changed decisions, but did not improve the aggregate TP/FP trade-off or a majority of folds.

## E06

- Fixed scalar control and inner-retuned scalar control were identical at `0.7655986509`.
- Adaptive logistic meta-fusion reached `0.7567114094`.
- It neither beat Control A nor Control B, so adaptive fusion is not justified.

## E07

E07 was not started because neither E05 nor E06 passed its strong gate. This is the required stopping behavior.

The official incumbent remains `0.6514285714 / 114 / 153 / 197`.
