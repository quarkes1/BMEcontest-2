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

## 说明

- 本包为纯推理（无需训练数据、无需 FD 外部数据集——预训练权重已随包发布）
- 管线：1s 活动包络（0.5-2Hz）× 内置时刻先验 → 多阈值连通域提案 → 240s 候选窗
  → 5 折深度排序模型 bagging（z-score 归一化）→ 阈值解码 → 形态学后处理
- 5 折交叉验证（全局官方口径 IoU≥0.25）F1 = 0.333（单折最高 0.447）；
  评估细节见主仓库 README
