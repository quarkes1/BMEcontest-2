# 进食事件检测推理包

对智能手表传感器会话（HUAWEI Research 格式目录，含 collect_data*.txt）
输出检测到的进食事件（Episode 起止时间，毫秒时间戳）。

## 运行

```bash
pip install -r requirements.txt
python predict.py --input <会话目录或目录列表txt> --output predictions.json
```

可选参数：--tau 0.30（深度分阈值）--merge-gap 120 --min-dur 120 --dilation 60
--device cpu|cuda（默认自动选择）

## 输出

JSON：{ "<会话目录名>": [ [start_ms, end_ms], ... ], ... }

## 状态（2026-09-07）

- **本包当前为对照系统（检测即排序 + FD 预训练）推理实现**——保留可运行
  基线：提案（0.5-2Hz 包络 × 时刻先验连通域）→ 240s 候选 → 5 折 MM-Ranker
  深度 bagging → 阈值解码 → 形态学后处理。
- **主系统（滑窗管线 v4.2）正在切换中**：全覆盖滑窗（240s/15s）+ 62 特征
  HGB（5 折 bag）+ 密度聚合 + 33 特征 L2 复核器。5 折交叉验证（官方口径
  IoU≥0.25，eligible 质量审计分母）**F1 均值 0.595**（fold1 0.767 / fold4
  0.649 / fold3 0.540 / fold0 0.512 / fold2 0.508；全局聚合 0.592）。
  切换完成后本 README 更新为滑窗管线说明。

训练代码/复现/评估细节见主仓库 README（§复现）。
