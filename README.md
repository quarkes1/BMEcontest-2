# BMEcontest-2 — 基于智能手表传感器的进食检测

第十一届全国大学生生物医学工程创新设计竞赛 · 智能穿戴与运动健康赛道 · 赛题二。

从智能手表 IMU（加速度计+陀螺仪 ~105Hz）与 PPG（44 通道多波长，占空比采样）检测进食事件。
评估：事件 IoU>0.25 判正确 → **F1** 主指标 + 起止时间 MAE 次指标。
匹配**按受试者分组**（跨受试者事件不得匹配他人餐次——多人同时段采集，全局匹配会产生
跨受试者伪命中，详见 `docs/架构回顾与重构决策.md`）。

## 算法架构：检测即排序（rank_events.py，2026-08-30 起为唯一主路径）

```
[1s 网格] 0.5-2Hz 带通包络 × 时刻先验（train 折估计的 24h 高斯先验）
    ↓
[提案] 多参数连通域并集（pct∈{75,82,88,92,95} × gap∈{30,60} × dur∈{5,10}，
        IoU>0.6 相邻合并）→ 候选事件（召回优先，fold0 覆盖 24/38 餐）
    ↓
[排序头] LightGBM 12 特征：时长/峰值/均值/std/p90/尖峰性/时刻先验/上下文对比/
         会话时长/env_p95/p95_ratio/会话内相对位置/会话门控概率
    ↓
[会话门控] LightGBM 会话级"有无餐"分类器（7 统计特征，AUC≈0.88，train→val 不泄漏）
    ↓
[解码] 门控概率 ≥ g × 每会话 top-k 个候选 → 事件输出（g、k 按折网格搜索）
```

**结果（受试者级匹配，5 折均值）**：**F1 = 0.161 ± 0.058**（最佳 fold4 0.239；敏感度均值
0.263，PPV 均值 0.128）。对照：诚实启发式基线 V2-B = 0.085，V1 级联栈 ≈0.105（旧全局
匹配口径，含跨受试者伪命中）。

## 项目文件结构

```
├── src/                      # 核心代码
│   ├── config.py             # 全局路径与常量（采样率/窗口/阈值/种子）
│   ├── data/                 # 数据管线
│   │   ├── manifests.py      # 会话清单/用餐标签/用户表加载与清洗
│   │   ├── loader.py         # TSV 流式解析（行空间时间网格 + 占空比掩码）
│   │   └── splits.py         # 按受试者 GroupKFold 5 折划分
│   ├── eval/metrics.py       # 事件 IoU 匹配 → F1/灵敏度/PPV/MAE（受试者级，组委会口径）
│   └── infer/events.py       # 阈值连通域 → 事件列表（提案与评估共用）
├── scripts/
│   ├── validate_baselines.py # 1s 包络缓存构建 + V1/V2 基线验证（只读实验）
│   └── rank_events.py        # 主路径：提案 + 排序头 + 会话门控 + 解码网格
├── docs/                     # 架构回顾与重构决策 / W1/W2 报告 / 数据处理说明
├── Data/                     # 原始数据（gitignore，只读）
├── Archieves/                # 废弃架构（V1 级联栈、DensityNet）完整存档
└── cache/ models/ outputs/   # 运行时产物（gitignore）
```

> 旧架构（L1-L4 级联、L3a/L3b 深度网络、融合+HMM、ONNX/PyInstaller 部署件、
> DensityNet 密度回归）已整体移入 `Archieves/V1_deprecated/`，失败根因分析见
> `docs/架构回顾与重构决策.md`。

## 环境

```bash
conda create -n bme python=3.11 -y
conda activate bme
pip install numpy scipy pandas scikit-learn lightgbm torch
```

## 快速开始

```bash
conda activate bme
python scripts/validate_baselines.py --folds 0,1,2,3,4   # 构建 1s 包络缓存 + 诚实基线（首次 ~30 分钟，幂等）
python scripts/rank_events.py --fold 0                   # 单折排序管道（提案+排序+门控解码）
python scripts/rank_events.py --fold 0 --fold 1 --fold 2 --fold 3 --fold 4   # 5 折
```

产物：`outputs/rank_events_fold{k}.json`（含解码网格全表与最佳配置）、
`outputs/validate_baselines.json`（诚实基线对照）。

## 关键决策记录

| 日期 | 决策 |
|---|---|
| 2026-08-29 | 端-云协同：移除参数预算，允许大模型；准确率优先于功耗 |
| 2026-08-30 | 架构 pivot：窗口分类栈 → 检测即排序（本 README） |
| 2026-08-30 | 评估口径修正：事件匹配按受试者分组（原全局匹配 79% TP 为跨受试者伪命中） |
