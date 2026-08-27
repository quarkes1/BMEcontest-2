# BMEcontest-2 — 基于智能手表传感器的进食检测

第十一届全国大学生生物医学工程创新设计竞赛 · 智能穿戴与运动健康赛道 · 赛题二。

从智能手表 IMU（加速度计+陀螺仪 ~105Hz）与 PPG（44 通道多波长，占空比采样）检测进食事件；
两种佩戴场景分别建模（惯用手：IMU 动作；非惯用手：PPG 生理信号）。
评估：事件 IoU>0.25 判正确 → **F1** 主指标 + 起止时间 MAE 次指标。

## 项目文件结构

```
├── src/                      # 核心代码
│   ├── config.py             # 全局路径与常量（采样率/窗口/阈值/种子）
│   ├── data/                 # 数据管线
│   │   ├── manifests.py      # 会话清单/用餐标签/用户表加载与清洗
│   │   ├── loader.py         # TSV 流式解析（行空间时间网格 + 占空比掩码）
│   │   ├── windows.py        # 窗口切分 + IoU 标签（正/灰区/负）
│   │   └── splits.py         # 按受试者 GroupKFold 5 折划分
│   ├── features/             # 特征提取（基线 37 维窗口特征；姿态角 W2）
│   ├── models/               # L1-L4 级联模型（基线 LightGBM；DL 分支 W2+）
│   ├── eval/metrics.py       # 事件 IoU 匹配 → F1/灵敏度/PPV/MAE（组委会口径）
│   └── infer/                # 端到端推理 + 测试集 I/O 适配层（W3）
├── web/                      # 跨平台网页应用（FastAPI + 原生JS + Three.js，W3）
├── scripts/
│   ├── data_acquisition/     # 数据下载记录（脚本已删除，仅留说明）
│   ├── validate_data.py      # 全量数据质量校验（损坏/重复/置零/跳跃）
│   ├── build_feature_cache.py# 并行构建窗口特征缓存
│   └── run_baseline.py       # LightGBM 基线 5 折评估
├── tests/                    # 单测（pytest，8 项全绿）
├── ReferenceDocs/            # 参考文献笔记（编号对应报告引用）
├── docs/                     # 设计文档/实施计划/数据处理说明
├── Data/                     # 原始数据（gitignore，只读）
├── Archieves/                # 竞赛官方材料（禁止改动）
└── cache/ models/ outputs/   # 运行时产物（gitignore）
```

## 算法架构：四级级联轻量化流水线

```
传感器流 ──► L1 事件唤醒层 ──► L2 场景门控层 ──┬─► L3a 动作专家网络（惯用手, IMU）
 (IMU+PPG)    双通道：IMU活动唤醒  倾角标准差判惯用/非惯用 │
              + 周期PPG巡值        (wearHand 标签监督)     └─► L3b 生理专家网络（非惯用手, PPG）
                                    ──► 动态可靠性加权融合 ──► L4 状态解码与事件输出
                                         α=f(PPG SNR,灌注指数)   HMM + 边界膨胀±6s ──► 事件列表
```

| 层 | 职责 | 方法 | 规模 |
|---|---|---|---|
| L1 守门员 | 低功耗唤醒 | 合值滑动方差+决策树；PPG 周期巡值补非惯用手静态进食 | <1KB |
| L2 分流器 | 惯用/非惯用场景 | 互补滤波姿态角+倾角统计量分类 | ~100 参数 |
| L3a 动作专家 | 惯用手进食概率 | 深度可分离时序卷积（11 通道含 0.5-2Hz 咀嚼带）+ 餐具多任务头 | ≤300K |
| L3b 生理专家 | 非惯用手进食概率 | 多波长噪声抵消 + HR/RMSSD/LF-HF + 波形嵌入 → GRU | ~150K |
| 融合 | 模态可信度加权 | α=f(PPG SNR, 灌注指数)，30s 平滑 | 小 MLP |
| L4 解码 | 事件输出 | 2 态 HMM Viterbi + 合并/过滤/边界膨胀 | O(T) |

设计依据详见 `docs/superpowers/specs/2026-08-27-eating-detection-design.md` 与 `ReferenceDocs/`。

## 环境

```bash
conda create -n bme python=3.11 -y
conda activate bme
pip install -r requirements.txt
```

## 快速开始

```bash
conda activate bme
python scripts/validate_data.py          # 全量数据质量校验（~20 分钟）
python scripts/build_feature_cache.py    # 窗口特征缓存（~30 分钟）
python scripts/run_baseline.py           # LightGBM 基线 5 折 F1
pytest tests/ -v                         # 单测
```

## 里程碑

| 周 | 内容 |
|---|---|
| W1 | 环境+数据管线地基+LightGBM 基线 F1 |
| W2 | L1/L2/L3a 打通（姿态解算+IMU 深度模型） |
| W3 | L3b+融合+HMM 全链路 5 折、ONNX/PyInstaller 可执行文件、web v1 |
| W4 | 3D 前臂骨骼、报告/PPT/复现 README、最终提交物 |
