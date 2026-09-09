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

## 性能（官方口径 IoU≥0.25，eligible 质量审计分母）

**协议说明（重要）**：主仓库 v5.1 报告值 0.617/0.632 及本包早期"部署验证 ~0.60"
系 wbag 泄漏协议产物——评估折的窗口概率由 5 折模型平均得到，其中 fold m≠k 模型
训练过折 k val 受试者的其他会话（受试者记忆）。第二轮 peer review 指出后已作废，
重估采用**受试者互斥零信息协议**（本折模型只见过本折 train 受试者，与部署到
全新受试者同构；复核器只用本折 train 候选训练）：

| 干净协议 F1（单模型，无 TCN——与部署同构） | fold0 | fold1 | fold2 | fold3 | fold4 | 均值 |
|---|---|---|---|---|---|---|
| | 0.378 | 0.645 | 0.300 | 0.407 | 0.515 | **0.449** |

- 主仓库 v6 干净协议（含 TCN 特征，训练期 GPU 评估）：均值 **0.505**、聚合
  **0.512**——TCN 为干净协议下的真实增益（CPU 部署版不含）；
- 严格 LOSO（每留一受试者重训窗模型 + 复核器，CPU 口径，固定阈值 0.717）：
  聚合 F1 **0.416**（67/153）。
- 部署 bag（5 折模型平均，全部模型未见测试受试者——部署场景零泄漏）：
  干净的折内多样 bag（负样本重采样）在 no-TCN 口径 +0.024（均值 0.473），
  单模型 0.449 为保守估计。

阈值 --thr 可覆盖 config.json 中位 0.717。训练/复现见主仓库 README（复现命令即
默认单模型模式 = 干净协议）。

旧版推理（检测即排序 + FD 深度模型，对照系统）存档：predict_legacy.py。
