# 进食事件检测推理包（滑窗管线版）

> 状态（2026-09-11）：本目录尚未包含 `event_stack/` 正式发布包。当前可运行的仍是下文
> 的旧滑窗管线。候选控制/stacking 的 `dist/event_stack` 只可由
> `scripts/package_event_stack.py` 从通过严格晋级门槛的 `role=deployment` bundle 原子生成；
> Task 6 在尚未产生合法 deployment bundle 前不得覆盖或创建该目录。

## 未来 event-stack 包（受限输入契约）

正式 bundle 就绪后，运行：

```bash
python event_stack/predict_event_stack.py \
  --bundle event_stack/bundle \
  --input-features candidates.json \
  --output predictions.json \
  --device auto
```

`--device` 仅接受 `auto|cpu|gpu|cuda`，其中 `gpu` 等同 `cuda`。当前 sklearn/LightGBM
组件均为 CPU：`auto` 会报告 `resolved_device=cpu`，而强制 `gpu/cuda` 会明确失败，绝不将
CPU 树模型伪报为 GPU 推理。当前 CUDA adapter registry 为空，因此无需安装 torch；打包或加载
发现 `cuda_adapter.py` 或任何 CUDA/component 声明都会拒绝该包。未来只有代码内显式、审计过的
注册协议可以启用设备实现；CPU/CUDA 输出相近或事件几何一致都不能证明推理实际使用 CUDA。

输入不是原始 `collect_data*.txt` 会话，而是已生成的、可审计的候选特征 JSON：

```json
{
  "feature_schema": {"macro": 63, "micro": 47, "verifier": 56},
  "schema_hash": "SHA-256 of canonical feature_schema JSON",
  "sessions": [{
    "subject_id": "stable-subject-id",
    "sid": "session-id",
    "candidates": [{
      "start_ms": 0, "end_ms": 1000,
      "macro": ["63 finite values"],
      "micro": ["47 finite values"],
      "verifier": ["56 finite values"]
    }]
  }]
}
```

运行时会校验 bundle manifest 的每个模型/metadata SHA-256、完整模型集合和 deployment role，
并拒绝 schema/hash 不匹配。`subject_id` 与 `sid` 都是必填；`sid` 仅定义会话几何，冻结的
candidate NMS/准入阈值/每受试者 cap 以及 event 阈值/每受试者 event cap 均按 `subject_id`
执行。未知或遗留字段、缺失 subject_id、任何重复 sid（包括同一 subject_id 内）均会被拒绝。输出为稳定
排序、紧凑 canonical JSON：
`{"events":[{"sid":...,"start_ms":...,"end_ms":...,"score":...}],"resolved_device":"cpu"}`。

发布只接受正式 `promote_summary` 路径写出的 deployment bundle：同一 run 根目录必须包含
canonical aggregate summary 与 `promotion_attestation.json`，后者绑定 run key、严格五个 outer
fold、门槛/F1 和全部五折及 deployment manifest 的 SHA-256。该 attestation 是结构化可验证
来源，不是秘密签名；手写 deployment manifest 中的 F1 不能绕过晋级门槛。打包器的默认目标是
仓库 `dist/event_stack`，并只允许显式可信 `dist` 根下的精确 `event_stack` 子目录；不会触碰
`dist/` 的旧文件。

这是一个有意的临时限制：Task 4 artifact 未包含从原始会话构造 macro-63、micro-47、
verifier-56 特征的已验证运行时模块、候选器配置或模型输入适配器；旧 `predict.py` 的
MM-Ranker 深度管线不兼容，不能作为替代。Task 6 前必须补齐并验证该 raw-session adapter，
才能宣称 event-stack 支持原始会话输入。

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
