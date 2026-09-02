# 基于智能手表传感器的进食检测算法

第十一届全国大学生生物医学工程创新设计竞赛 · 智能穿戴与运动健康赛道 · 赛题二
（本 README 按试题作品 7 大要素组织，供总结报告写作参考）

---

## 1. 选题背景及意义

- **场景**：智能手表全天候采集 IMU（加速度计+陀螺仪 ~105Hz，raw ADC）与 PPG（44 通道，
  有效采样 ~2Hz）多源传感器数据，为无感自动识别进食行为提供基础。
- **挑战**：进食动作与日常动作（喝水、打电话、刷牙）高度相似；两种佩戴场景检测原理
  本质不同——**惯用手**（手部动作特征明显，IMU 主导）与**非惯用手**（动作弱，需
  上下文/生理信号）。
- **意义**：自由生活进食监测服务于饮食行为量化、慢病管理、营养干预等健康场景；
  全天多传感器数据下的鲁棒事件检测是穿戴健康的核心共性技术。
- **评估**（Resources/试题.txt 锁定）：预测 Episode 与 GT Episode `IoU ≥ 0.25` 判 TP；
  全局灵敏度 = TP/真实事件数、PPV = TP/预测事件数、F1 = 2·Sens·PPV/(Sens+PPV)；
  次要指标：正确匹配事件的起止时间 MAE。

## 2. 研究目标

1. 惯用手/非惯用手两种佩戴场景下的进食事件（Episode 级）检测，全局官方口径 F1 最大化；
2. 解决"排序头对真餐候选打分偏低"瓶颈——通过外部数据预训练（FD 数据集）注入
   干净进食模式先验；
3. 双分支架构：惯用手 Bite 触发器（IMU 4-16Hz 高频能量）+ 非惯用手慢响应上下文，
   数据驱动自适应路由（不依赖 dominant_hand 元数据）；
4. 官方评估口径对齐：全局事件级 IoU≥0.25 评估 + 官方后处理流水线（1Hz 平滑/融合/过滤）。

## 3. 设计原理及方案（检测即排序 v2）

```
[1s 活动包络(0.5-2Hz) × 24h 时刻先验] → 提案池（活动连通域 + 时间先验两池）
  → 候选 240s 窗口（±120s 上下文, 10Hz, 6ch）
  → LGBM 14 特征排序 ⊕ MM-Ranker TCN 深度分（w 加权融合）
  → 会话门控（LGBM 会话级"有无餐", AUC≈0.88）
  → 解码（阈值选窗 / 官方 1Hz 平滑 Pipeline）→ 事件输出
```

**关键组件**：

| 组件 | 位置 | 说明 |
|---|---|---|
| MM-Ranker | src/models/ranker.py | IMU TCN 分支（6 层膨胀因果卷积残差栈）+ Bi-GRU 融合 + meta 特征（时长/时刻先验/门控概率）→ 候选深度分；Focal Loss + 硬负样本挖掘 |
| FD 预训练 | scripts/prep_fd.py, pretrain_fd.py | FD-I/FD-II（KU Leuven 61 人全天双腕 IMU 64Hz）→ Episode 级候选分类预训练（任务对齐 + z-score 归一化，valAUC 0.982）→ checkpoints/fd_pretrained_s1.pt |
| 双分支路由 | src/models/dualbranch.py | AdaptiveRouter（连续插值）+ HardRouter（前 10min 方差硬切换）；实测单侧佩戴下均负向，代码保留供测试集含非惯用手时启用 |
| 官方评估/后处理 | scripts/official_iou_eval.py | 全局 IoU≥0.25 评估 + 官方流水线（1Hz 映射→60s 平滑→阈值→180s 融合→120s 过滤→dilation 网格寻优 dil=120s） |

**FD 预训练成功三要素**（对比 MO bite 级预训练负向的教训）：
①Episode 级任务对齐（bite 级任务与餐次级排序不迁移）；②FD-I 全天干净负样本
（0=确认非餐；FD-II 的 0=未标注不可用）；③z-score 归一化（FD acc 为 m/s² 物理单位
vs 本项目 raw ADC，尺度差 ~385 倍——通道校验流程 scripts/prep_fd.py --check 抓出）。

## 4. 主要算法程序

| 脚本 | 功能 |
|---|---|
| scripts/build_candidate_windows.py | 候选窗口缓存（240s±120s, 10Hz 插值, 6ch + MA/PPG 块统计） |
| scripts/validate_baselines.py | 1s 活动包络缓存（0.5-2Hz 带通）+ V1 启发式基线 |
| scripts/train_ranker.py | MM-Ranker 训练（--init-from FD 权重 / --loss focal\|asymmetric / --session-norm / --freeze-first） |
| scripts/rank_events.py / rank_events_v2.py | 提案生成 + 解码（prepare_fold 供评估复用，防复刻泄漏） |
| scripts/official_iou_eval.py | 官方评估 + 后处理 + 对比表（--all / --grid-post / --routed） |
| scripts/prep_fd.py / pretrain_fd.py | FD 通道校验 + 缓存构建 / Episode 级预训练 |
| scripts/analyze_label_offset.py | GT 标签时间偏移分析（详见 §6） |
| scripts/run_fd_finetune.sh / run_7ch_chain.sh | 全链实验（FD 微调 / 7ch 通道） |

运行流程见 §复现。

## 5. 代码得分及结果说明

**最终成绩（全局官方口径，5 折均值）**：

| 系统 | 现有管线 F1 | 官方后处理 F1 |
|---|---|---|
| ens3 基线（无预训练） | 0.298 | 0.159 |
| **FD 预训练微调（最终提交）** | **0.333** | 0.210（网格最优 dil=120/gap=180 → **0.240**） |
| FD + 7ch（gyro 高频） | 0.325 | —（负向，不采用） |

逐折（FD 预训练，现有管线）：**0.367 / 0.258 / 0.364 / 0.229 / 0.447**（fold4 单折最高）。
提交产物：outputs/archive_final_20260902/（FD 权重 + 5 折分数 + best 配置，15MB）。

## 6. 结果分析与评价

### 6.1 完整消融/实验矩阵

| 实验方向 | 结果 | 结论 |
|---|---|---|
| FD Episode 级预训练（M2） | 0.298→0.333（+0.035，5 折全正向，fold1 +0.071） | **核心增益**：任务对齐+全天负样本+z-score |
| 官方后处理网格（D1） | 0.210→0.240（dil=120/gap=180） | dilation 单调增益；仍低于现有管线 |
| 7ch gyro 高频通道（M3） | 0.333→0.325（-0.008） | FD 已学到等价特征，通道冗余 |
| AdaptiveRouter 连续路由（M3） | fold0 -0.059 | FD 深度分主导时降权有害 |
| 冻结 TCN 前 3 层微调（M3） | fold0 -0.034 | 全量微调最优 |
| 硬路由 HardRouter（D3） | fold1 0.258→0.171（-0.087） | 会话前 10min 静置致方差误判 |
| 损失提权 α0.75 / Asymmetric（D2） | fold1 -0.014 / -0.016 | valAUC 提升与事件级脱节 |
| Session IN（D2） | fold1 -0.025 | 破坏 FD 全局 z-score 分布 |
| PPG 分支（前期） | -0.060 负向 | 有效采样 2Hz，HR/HRV 不可行 |
| MO bite 预训练（前期） | -0.008 负向 | bite→餐次任务鸿沟 |

**负向实验的共性机理**：FD 深度分是主导信号（best 配置 w→1.0），任何降权/分布破坏
（路由/损失调整/Session IN）都有害；排序头打分低根因是数据属性而非训练目标。

### 6.2 标签时间偏移数学证明（决定 F1 上限的关键发现）

全 5 折测量（scripts/analyze_label_offset.py）：GT 标注中心与 IMU 活动峰值
（0.5-2Hz 包络）偏移中位 **10.2-11.8min**（全折一致）：

| fold | 偏移中位 | 偏移 >9min 餐占比 | 理论 Sens 上限 |
|---|---|---|---|
| 0 | 10.2 min | 54% | ~46% |
| 1 | 11.8 min | 74% | ~26% |
| 2 | 11.5 min | 57% | ~43% |
| 3 | 10.2 min | 61% | ~39% |
| 4 | 11.5 min | 65% | ~35% |

**推导**：餐长 D（中位 ~15min）、GT 偏移 δ，检测器即使完美输出真实进食时段，
`IoU = (D−δ)/(D+δ) ≥ 0.25` ⇒ **δ ≤ 0.6·D ≈ 9min**。偏移超 9min 的餐在数学上
不可匹配——54-74% 的餐受此限制，**评估标签（用户自报餐次）的时间精度锁死了
可达 F1**。FD 预训练（0.333）已逼近该标签噪声下的可达上限（理论均值 ~38%）。

### 6.3 与文献对比及差距归因

| | 文献（FD/MO 等） | 本项目竞赛数据 |
|---|---|---|
| GT 来源 | 相机视频**秒级**标注（研究助理陪同） | 华为平台**用户自报**（±10-15min） |
| 匹配可达性 | IoU≥0.25 完全可达 → F1 0.7-0.9 | 半数餐数学不可匹配 → 上限 ~0.33-0.35 |
| 数据场景 | 受控协议、双腕、物理单位 | 自由生活、单腕、raw ADC、PPG 2Hz |

## 7. 总结与应用展望

**总结**：构建了"检测即排序 v2"——多参数提案 + LGBM/TCN 双排序 + 会话门控 +
官方后处理；FD 外部数据 Episode 级预训练带来 5 折一致正向提升（+0.035，全局口径
0.333，单折最高 0.447）；系统验证并解释了提升瓶颈为**评估标签时间精度**
（数学证明 δ≤0.6·D 匹配界限），非模型能力。

**展望**：
1. 标签精化：若 GT 时间戳可校准（活动对齐或人工复核），可达 F1 将显著上移；
2. 外部数据扩充：FD-II 域外验证 + 更大规模全天预训练；
3. 多模态：更高采样率 PPG（≥25Hz）可启用 HR/HRV 慢响应分支（非惯用手场景）；
4. 测试集含非惯用手佩戴时，启用 HardRouter/AdaptiveRouter（已实现，实测负向的
   原因在于验证集单侧佩戴，路由信号本身存在：有餐会话 4-16Hz 能量均值显著更高）。

---

## 复现

```bash
conda activate bme
# 1. 缓存（env 包络 + 候选窗）
python scripts/validate_baselines.py --folds 0,1,2,3,4
python scripts/build_candidate_windows.py --fold {0..4}
# 2. FD 预训练（需 FDdatasets；--check 通道校验前置）
python scripts/prep_fd.py --check && python scripts/prep_fd.py --build
python scripts/pretrain_fd.py            # → checkpoints/fd_pretrained_s1.pt
# 3. 排序头微调（--init-from 加载 FD 权重）
python scripts/train_ranker.py --fold {0..4} --no-ppg --init-from checkpoints/fd_pretrained_s1.pt
# 4. 解码 + 官方评估
python scripts/rank_events_v2.py --fold {0..4} --prior-grid 15m
python scripts/official_iou_eval.py --all          # 对比表
python scripts/official_iou_eval.py --grid-post    # 后处理网格（dil=120/gap=180 最优）
python scripts/analyze_label_offset.py             # 标签偏移分析
```

## 项目结构

```
src/            # 核心库（config/data/eval/infer/models）
scripts/        # 13 个核心脚本（训练/解码/评估/FD/分析）
docs/           # 三阶段重构设计.md（设计+验证矩阵+变更日志）+ 数据处理说明.md
checkpoints/    # FD 预训练权重（fd_pretrained_s1.pt）
FDdatasets/     # FD-I/FD-II（zip + 解压）
cache/          # 可重建缓存（sessions/cand_windows/validate_baselines/fd_windows/splits）
outputs/        # archive_final_20260902（最终版本）+ exp_fdpre + canonical
refs/           # 参考代码与外部数据（MO 等）
ReferenceDocs/  # 文献综述（报告引用素材）
Archieves/  Data/   # 历史与原始数据（保留）
```

设计文档：docs/三阶段重构设计.md（含全部实验矩阵、标签偏移证明、变更日志）。
