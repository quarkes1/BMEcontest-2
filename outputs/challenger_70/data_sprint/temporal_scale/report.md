# Temporal-scale audit

Fixed scales: 5/10/15/30/60 s. All measurements use cached raw IMU and the fixed incumbent populations.

Strongest registered proxy: `gyro_variance` at 5 s; TP–FP direction agreed in 5/5 folds, robust separation=0.722, reachable FN TP-like=True.
Coverage audit: 6/1170 scale rows are below 0.8 coverage; no imputation was used.
Mechanism flag: **present**. No feature or classifier was promoted.

