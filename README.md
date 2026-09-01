# BMEcontest-2 — 基于智能手表传感器的进食检测算法

第十一届全国大学生生物医学工程创新设计竞赛 · 智能穿戴与运动健康赛道 · 赛题二。

## 任务与评估（Resources/试题.txt 锁定）

- 数据：华为手表 IMU（加速度计+陀螺仪 ~105Hz，raw ADC）+ PPG（44 通道，有效采样 ~2Hz）
  全天自由生活采集，餐次级（Episode）标注
- 场景（试题要求分别设计）：**惯用手**（IMU 动作特征明显）/ **非惯用手**（动作弱，
  需上下文/生理信号）
- 评估（全局事件级，官方口径）：预测 Episode 与 GT Episode `IoU ≥ 0.25` → TP；
  灵敏度 = TP/真实事件数；PPV = TP/预测事件数；**F1 = 2·Sens·PPV/(Sens+PPV)**；
  次要指标：正确匹配事件的起止时间 MAE
- 实现与验证：`scripts/official_iou_eval.py`（全局贪心匹配 + 官方后处理流水线）

## 算法架构（检测即排序 v2）

```
[1s 活动包络 × 时刻先验] → 提案池（活动连通域 + 时间先验两池）
  → 候选 240s 窗口（±120s 上下文, 10Hz, 6ch 或 7ch=+gyro 4-16Hz Bite 通道）
  → LGBM 14 特征排序 ⊕ MM-Ranker TCN 深度分（w 加权融合）
  → 会话门控（LGBM 会话级"有无餐", AUC≈0.88）
  → 解码（阈值选窗 / 官方 1Hz 平滑 Pipeline）→ 事件输出
```

关键组件：
- **MM-Ranker**（`src/models/ranker.py`）：IMU TCN 分支（膨胀因果卷积残差栈）+
  PPG 块统计（默认关闭，消融负向）+ MA 掩码 + Bi-GRU 融合 + meta 特征 → 候选深度分
- **FD 预训练**（`checkpoints/fd_pretrained_s1.pt`，阶段二）：FD-I/FD-II（KU Leuven
  全天双腕 IMU 64Hz）→ Episode 级候选分类预训练（任务对齐 + z-score 归一化），
  微调后 5 折全正向（全局口径 +0.035）
- **双分支路由**（`src/models/dualbranch.py`，阶段三）：会话级 gyro 4-16Hz 能量 →
  惯用手分支权重 α 连续插值（数据驱动，不假定 dominant_hand 元数据）
- **官方后处理**（`scripts/official_iou_eval.py`）：候选概率 → 1Hz 连续时间轴 →
  60s 平滑 → 阈值截断 → 180s 融合 → 120s 过滤 →（可选）边界膨胀

## 结果（全局官方口径，5 折均值）

| 系统 | 现有管线 F1 | 官方后处理 F1 |
|---|---|---|
| ens3 基线（无预训练） | 0.298 | 0.159 |
| **FD 预训练微调（最终）** | **0.333** | **0.210** |
| FD + 7ch（gyro 高频通道） | 0.325 | —（阶段三验证负向） |

逐折（FD 预训练，现有管线）：0.367 / 0.258 / 0.364 / 0.229 / 0.447。
阶段三验证（设计文档 M3）：7ch 通道 / AdaptiveRouter 路由 / 冻结微调均负向——FD
全量微调为最终配置（FD 预训练 valAUC 0.982，5 折全正向 +0.035）。

## 项目文件结构

```
├── src/                      # 核心库
│   ├── config.py             # 全局路径与常量（采样率/窗口/种子）
│   ├── data/                 # manifests（会话/标签）/ loader（TSV 解析）/ splits（受试者 5 折）
│   ├── eval/metrics.py       # 事件 IoU 匹配（评估组件）
│   ├── infer/events.py       # 阈值连通域 → 事件
│   └── models/               # ranker.py（MM-Ranker）/ dualbranch.py（自适应路由）
├── scripts/
│   ├── build_candidate_windows.py  # 候选窗口缓存（6/7ch, 240s±120s）
│   ├── train_ranker.py             # MM-Ranker 训练（--init-from FD 权重, --freeze-first）
│   ├── rank_events.py / rank_events_v2.py  # 提案 + 解码（prepare_fold 供评估复用）
│   ├── official_iou_eval.py        # 官方评估 + 后处理 + 对比表
│   ├── prep_fd.py / pretrain_fd.py # FD 预处理（通道校验）/ Episode 级预训练
│   ├── validate_baselines.py       # env 包络缓存 + V1 基线
│   └── run_7ch_chain.sh / run_fd_finetune.sh  # 全链实验脚本
├── docs/                     # 三阶段重构设计.md（当前架构）/ 历史报告
├── checkpoints/              # FD 预训练权重（fd_pretrained_s1.pt, fd_pretrained_s1_7ch.pt）
├── FDdatasets/               # FD-I/FD-II 原始数据（不提交）
├── cache/                    # 可重建缓存（cand_windows/validate_baselines/sessions/splits/fd_windows）
├── outputs/                  # 实验产物（archive_best_20260831/ 存档最优配置）
├── refs/                     # 参考代码与外部数据（MO 等）
├── Archieves/  Data/         # 历史与原始数据（保留）
```

## 复现

```bash
conda activate bme
# 1. 候选窗口缓存（6ch；7ch 见 build_candidate_windows GYRO_HI_BAND）
python scripts/build_candidate_windows.py --fold {0..4}
# 2. FD 预训练（可选，需 FDdatasets）
python scripts/prep_fd.py --check && python scripts/prep_fd.py --build
python scripts/pretrain_fd.py          # → checkpoints/fd_pretrained_s1.pt
# 3. 训练排序头（--init-from 加载 FD 权重）
python scripts/train_ranker.py --fold {0..4} --no-ppg --init-from checkpoints/fd_pretrained_s1.pt
# 4. 解码 + 官方评估
python scripts/rank_events_v2.py --fold {0..4} --prior-grid 15m
python scripts/official_iou_eval.py --all
```

设计文档：`docs/三阶段重构设计.md`（含阶段一/二/三设计、验证矩阵、变更日志）。
