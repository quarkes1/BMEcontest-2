# 进食事件检测推理包（滑窗管线版）

对智能手表传感器会话（HUAWEI Research 格式目录，含 collect_data*.txt）
输出检测到的进食事件（Episode 起止时间，毫秒时间戳）。

## 运行

```bash
pip install -r requirements.txt
python predict.py --input <会话目录或目录列表txt> --output predictions.json
```

可选参数：--thr 0.717（复核阈值，默认取 slide_models/config.json 中位）
单会话纯 CPU 约 5-9 秒（2000+ 滑窗 × 5 模型 bag）。

## 输出

JSON：{ "<会话目录名>": [ [start_ms, end_ms], ... ], ... }

## 管线（主系统：全覆盖滑窗 + 两级检测）

```
会话 TSV → 240s/15s 全覆盖滑窗（真实时间戳，窗不跨缺口）
  → 62 维 ACC 特征 + 时刻先验（统计/1s 活动包络/峰率/频谱）
  → 5 折 HistGradientBoosting bag 概率（slide_models/wmodel_fold{k}.joblib）
  → 密度聚合候选（600s 中心 ≥10 越阈窗 + ≥80% 覆盖，越阈窗跨度定边界）
  → 37 特征 L2 复核器（slide_models/verifier.joblib；概率形态 + 上下文 + 时刻）
  → 阈值解码 → 事件
```

纯 CPU 推理（无 GPU/TCN 依赖）；scikit-learn + numpy + scipy 即足够。

## 性能（5 折 CV，官方口径 IoU≥0.25，eligible 质量审计分母）

本包为**纯 CPU 部署版（无 TCN 特征）**——组委会环境不依赖 GPU：

| 部署验证 F1 | fold0 | fold1 | fold2 | fold3 | fold4 | 均值 |
|---|---|---|---|---|---|---|
| | 0.490 | **0.750** | 0.414 | 0.594 | **0.759** | **~0.60** |

主仓库 v5.1（per-fold TCN 版，GPU 训练期评估）：均值 **0.617**、全局聚合
**0.632**——fold0-3 的 TCN 特征增益在部署 CPU 版不适用（fold4 无 TCN 反而最优）。

（部署模型 = 全数据复核器 + 5 折窗模型 bag——CV 验证 bag 优于单模型；
训练/复现见主仓库 README。阈值 --thr 可覆盖 config.json 中位 0.717。）

旧版推理（检测即排序 + FD 深度模型，对照系统）存档：predict_legacy.py。
