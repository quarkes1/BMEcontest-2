# 第二轮 peer review 回应与审计结果（2026-09-07）

对 peer review 四条指控的逐条审计结论、修复动作与干净协议重估数字。
审计全部在本仓库复现（脚本见下），审计日 hash/产物见 git log。

## 审计结论

### ① bag 泄漏——成立（v5.1 0.617/0.632 作废）

受试者划分本身互斥（每个受试者全部会话只在单折 val；各折 train/val 受试者
交集 = 0），但 **BME_WBAG=1 的 bag 在评估折 k 时加载全部 5 折模型平均**，其中
fold m≠k 模型的训练集包含折 k val 受试者的其他会话（session 级泄漏）。复核
特征/密度概率均由此泄漏概率派生。

干净协议重估（窗模型只在本折 train 训练；复核器只在本折 train 候选训练；
TCN 全折统一）：

| fold | v5.1（泄漏 bag） | 干净单模型 TCN | 干净单模型 no-TCN（部署口径） |
|---|---|---|---|
| 0 | 0.512 | 0.426 | 0.378 |
| 1 | 0.767 | 0.655 | 0.645 |
| 2 | 0.517 | 0.418 | 0.300 |
| 3 | 0.540 | 0.436 | 0.407 |
| 4 | 0.750 | 0.590 | 0.515 |
| 均值 | 0.617 | **0.505** | **0.449** |
| 聚合 | 0.632 | **0.512** | 0.458 |

泄漏幅度每折 0.09-0.24（fold4 0.750→0.590 最大——"fold4 高分"主要来自他折
模型对 fold4 受试者的会话记忆）。干净分数落入 peer review 估计区间 0.45-0.53，
与组内嵌套 OOF 0.5191 同量级。

### ② LOSO verifier 泄漏——成立（已修复）

原 LOSO 加载全数据 dist verifier（见过留出受试者的候选）。修复：每留一受试者
重训窗模型 + 复核器（只用其他受试者候选，窗模型对候选受试者自打分——与 5 折
per-fold 复核训练同构）；阈值固定 0.717 不利用受试者标签；eligible 口径与 5 折
一致（153）。

| LOSO | 口径 | 聚合 F1 |
|---|---|---|
| 旧版（verifier 泄漏） | eligible 180（宽松，无 ≥120s 窗覆盖检查） | 0.407 |
| 严格 v2（零信息） | eligible 153（与 5 折一致） | **0.416**（67/153，sens 0.438 ppv 0.396） |

verifier 泄漏本身贡献微小（0.407→0.416，方向为修复后略升）；旧版 0.407 与新版
0.416 的差异主要来自 eligible 口径（180→153，sens 分母收紧）。

### ③ 验证折架构选择（适应性过拟合）——成立，部分证伪

- **fold4 NO_TCN 模式选择**：干净协议下证伪——fold4 TCN 0.590 > NO_TCN 0.515，
  TCN 干净协议下全折增益（+0.05~0.12）。v5.1 的 NO_TCN 结论是泄漏 bag 上
  "已见过该受试者 → 无需 TCN 泛化表示"的伪影。
- 密度参数/负例数量逐折扫描：no_meal 150 为全局统一（非逐折）；干净协议下
  6 组全局窗阈值×密度参数均无一组超基线——该方向本身不产生增益。
- 阈值逐折在 val 上选：全局 pooled 阈值对照 ≈ 0.00-0.01（已测），残余乐观
  已标注。

### ④ "0.44/0.53/0.60 推算未验证"——承认

v5.1 与 dist "~0.60 部署验证"均为泄漏协议产物，作废。诚实估计（干净协议）：
TCN 研究配置聚合 0.512；CPU 部署（无 TCN）单模型 0.458/多样 bag 0.473；严格
LOSO 0.416。不再将 0.60 作为中心估计。

## 干净协议下的新诊断（本轮工作增量）

- 误差结构（153 eligible）：TP 78；候选层漏 28（短餐为主：漏检中位 7.8min vs
  TP 16.2min）；复核层拒 42（弱窗口证据餐）；复核 FP 74（窗口证据与真餐同强、
  聚集餐时——部分为未记录进食，PPV 底噪）。候选级 PR 上限 ~0.54。
- 窗级 AUC：fold2 0.785 最弱（fold0/3 损失在事件层而非窗层）。
- 密度覆盖率语义缺陷（conv 'same' 段边缘零填充稀释缺口邻域餐段 cov）已定位，
  修复（BME_DENS_COVFIX=1）使候选层漏 28→16，但复核器未适配缺口邻域候选，
  净 -0.01——待复核层适配后启用。
- sklearn HGB（early_stopping=False）确定性：同数据同种子结果 → BME_SEEDS 种子
  bag 无效；负样本重采样多样 bag：no-TCN +0.024、TCN -0.015。

## 复现

```bash
# 干净协议 5 折（默认即干净：单模型 + 本折 train）
python scripts/slide_verifier.py --fold {0..4}
# 严格 LOSO
python scripts/loso_eval.py --workers 8
# 干净多样 bag（CPU 口径）
BME_SEEDS=5 BME_SEEDS_RESEED=1 BME_NO_TCN=1 python scripts/slide_verifier.py --fold {0..4}
# 泄漏版（仅复现审计，不作泛化声明）
BME_WBAG=1 python scripts/slide_verifier.py --fold {0..4}
```

产物：outputs/slide_verifier_fold{k}.json（干净协议）、outputs/loso_result_strict.json、
outputs/slide_diag_fold{k}.json（误差结构）、outputs/slide_cand_fold{k}.npz（候选级分析）。
