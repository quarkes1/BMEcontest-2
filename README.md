# 基于智能手表传感器的进食检测算法

第十一届全国大学生生物医学工程创新设计竞赛 · 智能穿戴与运动健康赛道 · 赛题二

```text
竞赛任务     智能手表 IMU/PPG 全天数据 → 进食事件（Episode）检测
当前发布     event-stack / run 160afaf81debf1ee（strict nested-CV F1 = 0.6514285714）
推理入口     dist/inference（原始 collect_data*.txt → 预测 JSON）；dist/submission（竞赛提交包）
算法真源     src/pipeline/（唯一 canonical 实现；dist/ 全部为生成物）
```

---

![可视化展示](/intro_pngs/visual.png)

---

## 1. 竞赛任务与背景

- **场景**：智能手表全天候采集 IMU（加速度计+陀螺仪 ~105Hz，raw ADC）与 PPG（44 通道，
  有效采样 ~2Hz）多源数据，无感识别进食行为。
- **挑战**：进食动作与喝水、打电话等日常动作高度相似；惯用手（IMU 主导）与非惯用手
  （动作弱，依赖上下文/生理信号）两种佩戴场景检测原理不同。
- **评估（Resources/试题.txt 锁定）**：预测 Episode 与 GT Episode `IoU ≥ 0.25` 判 TP；
  sensitivity = TP/真实事件数、PPV = TP/预测事件数、F1 = 2·S·P/(S+P)；次要指标为正确
  匹配事件的起止时间 MAE。组委会在独立测试集上评分。
- **目标**：官方全局口径 F1 最大化；全程真实时间戳（包级恢复 + 缺口切段），杜绝时间轴
  类评估伪影；估计口径只使用可评估（eligible）餐集合。

## 2. 当前发布模型（CURRENT_PROMOTED_RELEASE）

| 项 | 值 |
|---|---|
| run key | `160afaf81debf1ee` |
| 严格五折聚合 F1 | **0.6514285714** |
| TP / eligible / pred / FP | 114 / 153 / 197 / 83 |
| PPV / recall | 0.5786802030 / 0.7450980392 |
| 候选 recall / 短餐最终 recall | 0.7712418301 / 20/39 = 0.5128205128 |
| 特征 schema | v2：macro 63、micro 47、verifier 116（56 基础 + 60 Context-v1） |
| 部署策略 | admission 阈值 .2、NMS IoU .3、subject cap 6、blend .25、解码阈值 .6595272837359293 |
| 运行时 ABI | Python 3.11.x；numpy 2.4.6 / joblib 1.5.3 / scikit-learn 1.9.0 / lightgbm 4.7.0 |

逐折证据（五份 outer-fold bundle + deployment + `promotion_attestation.json` 均经
SHA-256 锚定，位于 `models/event_stack/160afaf81debf1ee/`；指针为
`release/event_stack_incumbent.json`）：

| fold | config hash | TP/eligible/pred | F1 |
|---:|---|---:|---:|
| 0 | `afcf9609a2f6925a` | 16/23/35 | 0.5517241379 |
| 1 | `8b3e27bd2796d038` | 30/31/46 | 0.7792207792 |
| 2 | `b463226db1e6073a` | 15/27/33 | 0.5000000000 |
| 3 | `3f26fcb2172cd882` | 25/32/42 | 0.6756756757 |
| 4 | `0199041f39a09f5d` | 28/40/41 | 0.6913580247 |
| **聚合** | — | **114/153/197** | **0.6514285714** |

它相对前一已发布 run `035644cf0889a5dd`（0.5589743590）提升 `+0.0924542125`，满足
项目 F1≥0.65 目标与技术/推荐晋级门。**该数字是反复开发后的 nested-CV 开发证据，不是
独立测试集泛化承诺。**

## 3. 泄漏安全评估协议

当前全部正式数字来自**严格 subject-disjoint nested 五折**（`evaluate_event_stack.py`）：

- outer 五折按受试者互斥划分；inner 四路的窗模型只给未见受试者产生 OOF 候选，
  inner verifier 再在未见受试者候选上选择单一阈值与 admission/预算配置；
- 阈值、blend、准入与事件策略只在 outer-train 内部选择，outer-val 全程不可见；
- eligible 分母 = 可评估餐（会话数据覆盖 ≥50% 且餐时段窗覆盖 ≥120s），不罚数据
  不可达餐。

协议审计历史（细节见 `docs/repository_cleanup_audit.md`、`docs/superpowers/plans/` 与
git 历史）：

- **时间轴审计**：修复 `WINDOW_MS` 毫秒/行数混用与行号轴漂移，旧 0.319/0.333 结果作废；
- **跨折 bag 泄漏审计**：一版 5 折 bag 平均使评估折受试者被其他折模型见过，虚增
  0.08–0.13（0.617 作废）；当前发布不使用该结构；
- **LOSO 复核器泄漏修复**：留一受试者评估改为窗模型+复核器双双重训（严格 LOSO
  聚合 0.416，与同协议五折同量级），确认低分不能归因于采样方差；
- **组内互证**：组内平行方案在完整嵌套 OOF 下为 0.5191，与本项目 CPU 口径
  （0.45–0.47）同量级，交叉确认两级架构的有效性。

## 4. 架构

```text
原始 collect_data*.txt（真实时间戳、53 列 TSV）
  → SessionReader：有效 IMU 段/缺口切分（不跨缺口、不跨会话）
  → macro：240s/15s 全覆盖窗 → 62 维 ACC 特征 + 冻结 time-prior 适配器（63 维）
  → micro：15s/7.5s ACC+GYRO 重力对齐窗 → 47 维特征（micro 模型）
  → 候选并集：macro 密度候选 ∪ micro 阈值候选
  → 同会话稳定 NMS → subject admission（阈值/IoU/cap 由 outer-train OOF 冻结）
  → 事件复核：LogisticRegression + 受限 LightGBM 概率 blend（116 维 = 56 基础
     + Context-v1 60 列确定性同会话上下文）
  → 冻结 event policy（阈值、事件几何、subject budget）→ canonical Episode JSON
```

设计要点：

1. **全覆盖滑窗替代活动提案**：proposal 依赖"餐时段有连续活动段"（几何上仅 47–58%
   餐可达 IoU≥0.25），滑窗使每餐必然被多窗覆盖，候选层 recall 0.25 → 0.60+；
2. **两级精度架构**：窗口模型"宁滥勿缺"，假阳性由事件级复核器压制（融合了组内方案的
   密度候选 + 复核思路并加以扩展）；
3. **微窗口 ACC+GYRO 并集**：短餐与非惯用手场景的关键补充（表示保留为默认组件的
   micro 分支）；
4. **质量审计分母与真实时间戳**：见 §3；
5. **发布追溯**：每次晋级原子保留 canonical summary、五个 outer-fold evidence、
   deployment bundle 与 attestation；manifest 记录模型/输入指纹与 SHA-256。

对照系统（检测即排序 + FD 预训练微调）均值 ~0.27，仅作历史对照，其代码已清理
（见 §10 与 `tests/fixtures/deletion_manifest.json`）。

## 5. 快速推理与可视化

**一键启动（竞赛演示）**：双击 `dist/start.bat` —— 启动本地推理服务
（`dist/inference/serve.py`，仅绑定 127.0.0.1）并打开可视化应用；在页面中选择
`collect_data*.txt` 文件/文件夹即可运行 canonical 推理，查看证据时间轴、原始 IMU、
候选/事件与三维动作回放（`dist/visual/`，前端不含任何算法实现）。

`dist/inference/`（自包含、纯 CPU、无仓库依赖）：

```bash
cd dist/inference
python -m pip install -r requirements.txt
python predict.py path/to/collect_data1_2_3.txt --output prediction.json
python predict.py path/to/subject-folder --output prediction.json --include-timeline --include-candidates
```

或直接用 canonical API：

```python
from src.pipeline.inference import Predictor
predictor = Predictor.from_bundle("models/event_stack/160afaf81debf1ee/deployment")
result = predictor.predict_file("path/to/collect_data1_2_3.txt")
```

输出遵循 `dist/schema/prediction.schema.json`（schema_version `1.0`）。`--device cpu`
支持；无审计 CUDA adapter，`gpu/cuda` 明确拒绝。竞赛提交包 `dist/submission/`：

```bash
cd dist/submission
python main.py --raw path/to/collect_data1_2_3.txt --output result.json
```

## 6. 复现训练 / 评估 / 发布

```bash
# 只验证当前发布链（不重训）：registry + attestation + 5+1 bundle + dist 清单
python scripts/reproduce_release.py --run-key 160afaf81debf1ee

# 严格 subject-disjoint nested 五折评估（CPU-only，复用内容寻址缓存）
python scripts/evaluate_event_stack.py --fold all --inner-splits 4 --no-tcn --workers 5 \
    --micro-enabled --candidate-control-enabled --admission-minimum-recall 0.80 \
    --context-features v1 --summary-alias outputs/crossfit/context_v1_summary.json
python scripts/build_micro_features.py --fold all --split all --workers 8   # micro 缓存（如缺失）

# 合法 full-target 训练与晋级（只在 aggregate F1 严格超过 incumbent 时原子更新）
python scripts/release_event_stack.py --summary outputs/crossfit/context_v1_summary.json
python scripts/train_event_stack.py --summary outputs/crossfit/context_v1_summary.json

# 重建分发包（从 canonical 源生成，勿手改 dist/）
python scripts/build_inference_distribution.py
python scripts/build_submission.py

# 生成参赛作品报告（输出至 Archieves/；截图链路需 Node + 本机 Edge/Chrome）
node dist/visual/app/tools/capture-report-shots.mjs   # 界面截图（仅合成演示数据）
python scripts/report/build_report.py                 # 报告 docx
```

历史实验脚本（时间轴修复前的滑窗/对照系统开发命令）已按 §10 清理，均可从 git 历史
检出；正式证据为 `outputs/crossfit/`（summary + 5 逐折 JSON/diagnostics）。

## 7. 仓库结构

```text
BMEcontest-2/
├── src/                                  # 算法唯一真源（canonical implementation）
│   ├── pipeline/
│   │   ├── io/                           # 原始会话发现、时间线与缺口切分
│   │   ├── preprocessing/                # 有效段与窗口网格原语
│   │   ├── features/                     # macro（62→63）与 micro（47）特征生产器
│   │   ├── inference/                    # Predictor API、预测 schema、竞赛 adapter 边界
│   │   ├── event_stack.py                # 候选/复核特征/解码（canonical 事件图）
│   │   ├── runner.py                     # 数据集组装与训练编排
│   │   ├── candidate_control.py          # 同会话 NMS 与 subject admission
│   │   ├── context_features.py           # Context-v1（60 列确定性上下文）
│   │   ├── artifacts.py                  # bundle/manifest/attestation/incumbent 契约
│   │   ├── crossfit.py                   # subject-disjoint OOF 协议
│   │   ├── imu_features.py               # 47 维 ACC+GYRO 微窗口特征
│   │   ├── micro_cache.py                # micro 特征缓存 ABI
│   │   └── diagnostics.py                # 逐受试者诊断
│   ├── data/                             # raw/会话与 fold split 读取
│   └── eval/                             # 官方 IoU 匹配与指标
├── scripts/                              # 入口与实现脚本
│   ├── bootstrap_event_stack_diagnostics.py # 证据诊断修复
│   ├── build_inference_distribution.py   # 生成 dist/inference
│   ├── build_micro_features.py           # micro 特征缓存构建
│   ├── build_submission.py               # 生成 dist/submission
│   ├── crossfit_event_stack.py           # nested 评估实现
│   ├── evaluate_event_stack.py           # 严格 nested 评估（薄封装）
│   ├── official_iou_eval.py              # 官方匹配/评估
│   ├── package_event_stack.py            # 生成 dist/event_stack
│   ├── predict_event_stack.py            # serialized-payload 运行时入口
│   ├── promote_event_stack.py            # 晋级实现
│   ├── rank_events.py                    # 历史对照系统（依赖闭包保留）
│   ├── rank_events_v2.py                 # 历史对照系统（依赖闭包保留）
│   ├── release_event_stack.py            # 发布事务（原子替换）
│   ├── report/                           # 竞赛报告生成（build_report.py + 界面截图链路）
│   ├── reproduce_release.py              # 免训练验证发布链
│   ├── slide_features.py                 # 历史 macro 生产器（fixture 锚定）
│   ├── slide_verifier.py                 # 历史滑窗管线（parity 锚定）
│   └── train_event_stack.py              # full-target 训练/晋级（薄封装）
├── tests/                                # unit / integration / parity / release / pipeline（+fixtures）
├── models/
│   └── event_stack/<run_key>/            # promoted 5+1 bundle（不可变；attestation 锚定）
├── release/
│   └── event_stack_incumbent.json        # 当前发布指针（不可变）
├── outputs/
│   └── crossfit/                         # 正式五折证据（summary + 逐折 JSON）
├── dist/                                 # 发布工作区（结构见 dist/README.md）
├── docs/                                 # 审计、任务书、实施计划与数据处理说明
├── cache/                                # 可重建缓存（.gitignore）
├── FDdatasets/                           # KU Leuven FD-I/FD-II 外部数据
├── ReferenceDocs/                        # 文献综述（报告引用素材）
├── Archieves/                            # 历史归档（.gitignore）
├── Data/                                 # 原始传感器数据（.gitignore）
└── intro_pngs/                           # 可视化效果展示
```

## 8. 竞赛提交

`dist/submission/` 是竞赛最终交付物，由 `scripts/build_submission.py` 确定性地从
仓库唯一真源生成。包根只保留 `README.md`、`requirements.txt`、`start.bat`、`main.py`，
其余按职责成组：

- **`app/`**：推理运行时与本地桥（`serve.py` + vendored `event_stack/`）；
- **`models/`**：`event_stack/<run_key>/`（deployment bundle + 五个 outer-fold
  evidence bundle + promotion summary 与 attestation）；
- **`src/`、`scripts/`、`tests/`**：canonical 源码与训练/评估/发布/构建全链；
- **`release/`、`outputs/crossfit/`**：发布指针与五折证据（让包内 repro 可直接运行）；
- **`visual/`、`schema/`、`examples/`**：可视化与预测契约/安全示例；
- **`meta/`**：`manifest.json`（逐文件 SHA-256 与 source/model 闭包）与
  `feature_schema.json`；精确依赖见 `requirements.txt`。

README 内容由 `scripts/submission_readme.md` 维护（直接编辑该文件，重建生效）。
官方竞赛模式默认输出 CSV（每行一次进食事件）；自定义 adapter 的注册教程见包内 README。

验证：

```bash
cd dist/submission
python main.py --raw path/to/session.txt --output result.json   # 与 Predictor 输出一致
python main.py --official-input IN --output OUT                 # 退出码 2：adapter 未注册
```

**已知约束**：官方机器 I/O 规范未在已审计材料中定义，官方模式在注册具体 adapter
前显式拒绝，绝不猜测线格式（`src/pipeline/inference/competition_adapter.py`）；
可用接口（raw CLI / Predictor API / 预测 schema）见包内 README。

## 9. 可视化应用与契约

`dist/visual/` 是已实现的可视化应用（React + TypeScript + Vite + Three.js，源码在
`visual/app/`，构建产物 `index.html`/`assets/`/`runtime/` 在发行根）：证据时间轴
（Motion/Macro/Micro/Events 四轨、区间框选、缩放）、会话浏览器、原始 IMU 与姿态
查看器、事件列表、发布指标页与三视图刚体动作回放（IMU 姿态重建，缺标定即如实显示
不可用，绝不伪造轨迹）。两种模式：静态/演示（无后端，打开即用，明确标注 DEMO DATA）
与完整分析（本地推理桥在线时，选择 TXT 文件/文件夹运行 canonical 推理）。

契约：`dist/schema/prediction.schema.json`（v1.0，含可选 timeline series）+
`dist/examples/example_prediction.json`（合成安全示例）+ `dist/visual/README.md`
（使用与开发说明）。前端只消费 canonical 预测 JSON 与运动遥测（`motion.bin`
`f64_ms_6xf32_le`），**不得重实现**阈值、准入、融合、解码或特征提取。

## 10. 限制与后续

1. 0.6514 为 nested-CV 开发证据；最终泛化以组委会独立测试集为准，且当前只支持
   CPU 后端；
2. 短餐（<10min）最终 recall 20/39 与餐时高分误报（FP 83，部分疑为未记录进食）
   是主要剩余误差；
3. 官方提交 I/O 适配器未注册（见 §8）；
4. 迁移学习（FD-I/FD-II）未启动，须遵守 CC BY-NC-ND 4.0 并保留随机初始化与
   `external_weight=0` 对照。

### 历史实验简表（完整记录见 git 历史与 `docs/superpowers/plans/`）

| 阶段 | 关键改动 | 结果 | 状态 |
|---|---|---|---|
| 时间轴审计 | 修复毫秒/行数混用与行号轴漂移 | 旧 0.319/0.333 作废 | 已并入 |
| locked nested 基线 | 受试者互斥 inner/outer 两层交叉拟合 | F1 0.416（89/153/275） | 历史基线 |
| subject-budget / raw-summary 复核消融 | 事件上限 K、原始 62 维聚合复核 | 0.421 / 0.411–0.428 | 未采纳 |
| 微窗口 ACC+GYRO 并集 | 15s/7.5s 微窗候选∪macro | 0.478632（112/153/315） | 表示保留，候选体积门未过 |
| 候选控制 + LR/LGBM blend | 同会话 NMS + subject admission + blend | 0.5589743590 | 已被替代 |
| **Context-v1** | 确定性同会话上下文 → 116 维复核 | **0.6514285714** | **当前发布** |
| wbag 泄漏协议（v5.1/0.617） | 跨折 bag 见 val 受试者 | 虚增 0.08–0.13 | **作废** |

课题背景详见 `docs/三阶段重构设计.md`、`docs/组内算法对比与外部数据审计.md`；
仓库清理与删除清单见 `docs/repository_cleanup_audit.md` 与
`tests/fixtures/deletion_manifest.json`。
