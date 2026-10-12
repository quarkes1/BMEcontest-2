# A6 T1 — 四臂严格 inner 开发实验

本报告不是正式 outer 成绩。T24已反复用于开发；不更新dist，不代表达到正式0.70。

| 臂 | Event F1 | TP | FP | FN | PPV | Recall |
| --- | --- | --- | --- | --- | --- | --- |
| b | 0.721311 | 66 | 20 | 31 | 0.767442 | 0.680412 |
| p | 0.739583 | 71 | 24 | 26 | 0.747368 | 0.731959 |
| c | 0.763819 | 76 | 26 | 21 | 0.745098 | 0.783505 |
| t | 0.760417 | 73 | 22 | 24 | 0.768421 | 0.752577 |

## 归因与下一步资格

- p: ΔF1=+0.018272，3/3 wrapper非下降；扩大验证gate=通过。
- c: ΔF1=+0.042508，3/3 wrapper非下降；扩大验证gate=通过。
- t: ΔF1=+0.039105，3/3 wrapper非下降；扩大验证gate=通过。

Transformer vs pooling: +0.020833; vs TCN: -0.003402。

这只能区分当前固定配置的输入/读出收益；任何gate通过也不授权outer或发布。

## 证据

summary.json包含短餐召回、受试者median/IQR/P10、改善/恶化人数和无GT主体FP；逐wrapper health文件保留配对计数与事件。
resources.json保留实际拟合次数、步数、训练/编码/CPU耗时和显存；manifest与registered_plan锁定身份、配置、源码和保护快照。

计算：producer 120 fits，base 24 fits，neural 36 fits / 36000 steps；neural 3413.2s，encoding 877.3s，CPU 125.9s。
