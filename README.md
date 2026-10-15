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

- **场景**：智能手表全天候采集 IMU（加速度计+陀螺仪 ~105Hz，raw ADC）与 PPG（44 个导出列）
  多源数据。2026-10-13 T24抽样发现PPG每个约602ms时间戳含15个不同向量，不能将包率~1.66Hz
  当作波形采样率；约25Hz/列的包内序列假设尚待导出协议核实，详见[佩戴侧与PPG探查](outputs/challenger_70/representation_v2/hand_ppg_audit/report.md)。
  [下载脚本与组内解析核查](docs/PPG字段与下载解析证据.md)进一步支持25Hz包内序列，并找到PPG3/PPG5绿光映射声明；尚不等同于厂家协议确认。
  2026-10-14完成24人完整文件波形探查：3405个窗口中双路谱峰一致率77.39%，低/高运动组为84.07%/70.65%；
  原始包内行序的连续性明显优于倒序。仅支持进一步质量门控研究，不代表HR准确性或F1提升；详见[波形报告](outputs/challenger_70/representation_v2/hand_ppg_audit/waveform_report.md)。
  该面板仅覆盖1/97个T24餐和24/794个候选，不能外推进食识别效果。[A7候选级PPG研究设计](docs/challenge_070_amendment_a7_ppg.md)
  [P0候选级审计](outputs/challenger_70/representation_v2/amendment_a7_ppg/attempt_03/interpretation.md)已完成：227文件、2034唯一候选，85.38秒、峰值内存1.09GiB；41项专项测试通过。
  794个health候选中340个至少一维PPG差值可用；三分区总体频率差值方向一致，但相同IMU分数箱内样本不足。
  [P1固定融合](outputs/challenger_70/representation_v2/amendment_a7_ppg_fusion/attempt_03/interpretation.md)已完成且失败：同T24对照B=.72131、Q=.70408、P=.68421。
  [P2冻结先验残差](outputs/challenger_70/representation_v2/amendment_a7_ppg_residual/attempt_02/interpretation.md)亦失败：B=.72131、Q=.70930、P=.71264；P减少5FP但丢4TP、没有新增GT，171.74秒、峰值RSS3.61GiB。
  61项PPG专项测试通过。固定PPG代理路线收束，转向全天时序定位；未进入V7/outer或dist，正式模型仍为0.65143。
  [A8固定定位探针](outputs/challenger_70/representation_v2/amendment_a8_dense_localization/interpretation.md)：旧候选可达93/97GT，加入1317个源域候选只增加1GT、1235个候选无GT重叠，不直接部署；下一轮优先检验事件均衡复核损失。
  [A9主体×事件均衡损失](outputs/challenger_70/representation_v2/amendment_a9_event_balance/interpretation.md)已完成：.72131→.72043，TP+1/FP+2，gate失败，101.74秒；76项相关专项通过。下一方向优先改进hard-negative时序表示/源域监督，而非继续调本轮权重或阈值。以上均未改变正式发布。
- **挑战**：进食动作与喝水、打电话等日常动作高度相似；惯用手（IMU 主导）与非惯用手
  （动作弱，依赖上下文/生理信号）两种佩戴场景检测原理不同。
- **赛题评估（Resources/试题.txt）**：预测 Episode 与 GT Episode `IoU > 0.25` 判 TP；
  sensitivity = TP/真实事件数、PPV = TP/预测事件数、F1 = 2·S·P/(S+P)；次要指标为正确
  匹配事件的起止时间 MAE。组委会在独立测试集上评分。
- **历史实现差异**：当前发布评估器使用 `≥0.25`。2026-10-10核对封存的114个TP匹配，
  最小IoU为0.2604783、无恰好0.25的命中；因此这些冻结预测的发布计数不受严格比较影响。
  训练标签/阈值选择等路径的全面更正仍需版本化核查，不能从此推出重训后策略完全相同。
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

后续以实际事件F1≥0.70为研究目标，不给模型复杂度设全局硬上限，但必须检查跨人泛化及
过拟合。2026-10-14用户明确授权自主推进，在新的明确指令前不再请求计划审批。
正式可比F1提升确认后更新`dist/`可执行推理程序、模型和接口文档，并验证前端接入；
超过0.03的增益属于关键节点，须明确报告证据，不能以inner/diagnostic成绩替代。
当前首次触发线为>0.6814285714；候选AP、inner F1或GT-aware诊断不作为发布成绩。

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
  → 官方EXE：DataReader.iter_data；前端/API：原始文件读取（同一解析器）
  → SessionReader：有效 IMU 段/缺口切分（不跨缺口、不跨会话）
  → macro：240s/15s 全覆盖窗 → 62 维 ACC 特征 + 冻结 time-prior 适配器（63 维）
  → micro：15s/7.5s ACC+GYRO 重力对齐窗 → 47 维特征（micro 模型）
  → 候选并集：macro 密度候选 ∪ micro 阈值候选
  → 事件复核：LogisticRegression + 受限 LightGBM 概率 blend（116 维 = 56 基础
     + Context-v1 60 列确定性同会话上下文）
  → 同会话稳定 NMS → subject admission（阈值/IoU/cap 由 outer-train OOF 冻结）
  → 冻结 event policy（阈值、事件几何、subject budget）→ 事件
      ├─ 前端/API：canonical Episode JSON
      └─ 官方EXE：save_result → results.csv（externalid,beforeTime,afterTime）
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
.\EatingDetection.exe     # ACC_test_data放EXE旁；无参数，生成同级results.csv
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
python -m pip install -r scripts/requirements-build.txt  # 仅构建环境需要
python scripts/build_submission.py                      # 官方自包含EXE

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
│   │   ├── inference/                    # Predictor/JSON、official_runner/官方API、历史CSV边界
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
│   ├── build_competition_exe.py           # 官方无参数Windows EXE、模型/依赖内置、SHA清单
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

2026-10-15官方测试接口已适配，详见[交付与验收合同](docs/official_submission_20261015.md)。
`python scripts/build_submission.py`（或`build_competition_exe.py`）从canonical代码和当前
正式deployment构建`dist/submission/EatingDetection.exe`。它内置模型/CPython/依赖，目标
Windows10/11 x64无需安装Python或联网。onefile会在临时目录释放DLL，不是所有库的传统静态链接。

EXE无参数运行，以自身目录读取`ACC_test_data/userinfobean.txt`和每个人全部TXT，实际调用
官方DataReader/InfoReader/save_result。结果为同级`results.csv`，只有三列
`externalid,beforeTime,afterTime`，时间是13位原始Unix毫秒。成功运行完整替换，不混入旧结果；
零预测仍有表头。`EatingDetection.exe --self-check`可验证内置模型与依赖。

提交时直接将`dist/submission/`内容压缩为ZIP，EXE置于ZIP顶层。包内仅EXE、README、
build_manifest.json、official_api来源及licenses；无数据/金标准/旧CSV/训练脚本。README
真源为`scripts/official_submission_readme.md`；`build_manifest.json`可用
`python scripts/build_competition_exe.py --verify-only`核验逐文件SHA。可视化继续使用
`dist/visual`和`dist/inference`，不混入官方自动评分入口。

历史源码复现包可用`--source-bundle --destination dist/source_submission`单独生成，
其四列CSV适配器仅供历史自测，不能当作本次官方CSV。F1仍为0.6514285714，本轮不涉及新模型晋级。

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
3. 官方EXE已适配；隔离环境验收仍不能等同于另一台实体电脑或组委会独立测试成绩；
4. FD-I迁移等研究已开展但未获正式晋级；外部研究不自动进入本次提交模型。

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

### Challenge 阶段（诊断，不改变当前发布）

E00 已复现当前发布的五折计数 114/153/197；E01 将 39 个漏检拆分为
C0=6、C1=27、C2=4、C3=2，其中短餐漏检 19 个、短餐 C1 为 11 个。
E02 Context-v2 与 E03 hard-negative 均未通过预注册 inner gate，未进入
outer challenger 或发布流程。

Amendment A1 暂停 E04/E05，先完成只读的 E03.5 C1 attrition audit：
11 个短餐 C1 的首次全量丢失为 admission threshold 5 个、subject cap 6 个、
NMS 0 个。冻结 incumbent 分数及其余策略的单操作反事实中，取消 NMS
未救回短餐；取消 admission threshold 新增可达 1 个但最终未匹配；取消
subject cap 新增可达 6 个、最终匹配 1 个，同时新增 21 个最终 FP。
GT-aware 单候选 oracle 仅有 1/11 可通过原 event policy，其余 10 个低于
原 event threshold。以上均为已使用 outer folds 上的 **DIAGNOSTIC ONLY**
证据，不是新 F1、挑战者成绩或晋级依据。明细见
`outputs/challenger_70/e03_5_c1_attrition/` 与
`docs/superpowers/specs/2026-09-21-amendment-a1-e03-5-c1-attrition-design.md`。
人工审阅 A1 后，以 Amendment A2 将原 E04 改为只读的 verifier failure audit，
原 micro-v2 E04/E05 仍暂停。E04 在五折中审计了全部 31 个 raw 可达却最终
漏检的 GT（C1=27、C2=4；32 条 GT/候选配对），并以官方最终匹配的
114 个 TP 与 83 个 FP 作参考。按预注册的 best-score 见证，25/31 的 blend
分数比原 event threshold 低超过 0.10，3/31 位于其下 0.05 以内；
15/31 的 LR/LGBM 概率差至少 0.20。严格定义的 Context 过度压制 B 型为
0/31；两模型的 Context raw-margin 贡献中位数对 FN 均为正，但低于 TP，
不能据此声称“去掉 Context 会救回 FN”。完整分组、逐折特征方向和双见证
结果见 `outputs/challenger_70/e04_verifier_audit/`；它们均为
**DIAGNOSTIC ONLY / USES GT**，不是新 outer F1 或可直接使用的救援规则。
下一位 challenger 须经人工审阅 A2 后独立预注册，只在 outer-train 的 inner
OOF 上开发；当前发布与 `dist/` 均未变更。

当前已批准的下一轮研究按 [`0.70+ 任务书`](docs/challenge_070突破任务书_v2.md)
执行，授权仅覆盖 P0–P3，P4 全量 outer 与 P5 发布仍需分别人工批准。P0
溯源及资源预检、P1 时间轴/5 秒 token/小型有序时序网络与训练 smoke 已完成；
P2 的 Q0/Q1 目标数据 probe 已按 5×4 个内层分区完成，Q1 scratch 五组平均
candidate AP 0.26439，低于 Q0 frozen-random 的 0.28847，且仅 1/5 组正向、
log loss 更差；状态为 `P2_NO_SUPPORTED_BRANCH`，依照 gate **不进入 P3**。
这些不是事件 F1，也不是新 outer 成绩。FD-I 外部数据的陀螺仪物理单位与
本项目 raw counts 尚未验证，迁移臂 Q2/Q3 未运行、不能记作失败。阶段账本见
[`representation_v2/summary.md`](outputs/challenger_70/representation_v2/summary.md)。

后续审计发现上述 P2 实现在固定200ms间隙、candidate-local滤波、subject-first
采样、真实时间coverage及完整训练轨迹上存在协议偏差。旧分数保留，但不能据此
声称完整协议已验收或表示学习路线无效。已批准
[`RV2 Amendment A1`](docs/challenge_070_amendment_rv2_a1.md)：先修复合同，再仅在
原inner T内部额外划分train/health进行配对Q0/Q1健康检查（各2000步，不选epoch）。
入口 `python -m scripts.research.representation_v2.health`，独立证据目录
`outputs/challenger_70/representation_v2/amendment_a1/`。旧checkpoint显式保留
legacy coverage语义，新checkpoint记录模型配置；不修改旧P2产物、outer或dist。

R0已完成，状态 `R0_COMPLETE_REVIEW_REQUIRED`：16人训练、8人health，各臂2000步。
有效health候选上Q0/Q1 AP为0.34629/0.40242，但raw-logit BCE为1.18877/2.32075，
均差于同子群先验基线0.36737；两臂均触发末400步loss下降>10%的收敛告警。
这支持检查下游事件网络的过拟合，不支持自动延长训练或把单split AP提升当作成功。
按A1停在审阅点，建议下一轮预注册“相同低容量读出层下随机冻结与SSL冻结表示对照”。
详见[R0报告](outputs/challenger_70/representation_v2/amendment_a1/report.md)。正式F1仍为
0.6514285714；此次无新增outer、无dist/release变更。

用户随后授权自主探索，下一轮按[A2注册](docs/challenge_070_amendment_rv2_a2.md)
推进：原inner T内部三折，比较coverage/duration、随机冻结表示、candidate-conditioned
VICReg冻结表示；统一固定正则化线性读出，不训练事件TCN。每折SSL5000步，无参数搜索。
入口 `python -m scripts.research.representation_v2.explore`；三折probe不能替代事件F1，
outer与正式发布仍不在本轮授权范围。

A2已完成三折：随机冻结表示R的macro candidate AP/BCE为0.39779/0.32274，
candidate-conditioned SSL的S为0.29587/0.35450，0/3折AP改善，状态
`R1_NO_SUPPORTED_SSL_BRANCH`。见[A2报告](outputs/challenger_70/representation_v2/amendment_a2/report.md)。
不扩大此SSL配置；后续独立考虑无量纲FD手势监督预训练，不将其称为物理单位已校准。

自主探索的下一条注册为[A3无量纲迁移](docs/challenge_070_amendment_rv2_a3.md)：
先仅在FD-I右腕、27/7人源数据划分上训练固定5000步手势encoder，检验source可学习性；
入口 `python -m scripts.research.representation_v2.unitfree`。目标信号、outer和dist均不参与S0；
source gate通过后才登记目标域随机/预训练冻结表示的配对比较。

S0源域验证已通过：candidate AP 0.84405、raw BCE 0.25704（采样先验基线0.61827），
仅为FD源域结果，不是本项目F1。下一步按[S1固定对照](docs/challenge_070_amendment_rv2_a3_target.md)
在原inner T内部三折比较同一初始随机encoder与FD最终encoder，全部冻结，统一133D线性读出。
源权重SHA固定并严格核对来源、参与者和完整性；无新神经训练、无outer、无dist变更。

S1三折已完成并通过gate：R→FD的macro candidate AP为0.38033→0.54050，
raw BCE为0.31308→0.26956，3/3折正向。23名AP有定义的受试者中15名改善、7名下降、
1名持平；不是事件F1，也尚未证明与现有verifier互补。耗时约300秒，无新增神经训练或token缓存。
见[S1报告](outputs/challenger_70/representation_v2/amendment_a3_target/report.md)。下一方向为
inner-only的116D基线/随机扩维/FD扩维互补性与事件TP/FP验证；正式F1仍为0.6514285714。

下一轮[S2事件互补性注册](docs/challenge_070_amendment_rv2_a3_events.md)采用更低容量的固定评分融合，
暂不直接拼接高维特征：116D LR/LGBM基线，以及75%基线+25%随机/FD读出两个对照。
每个health折内再独立三折OOF选择策略，全部候选生产器与verifier均排除对应验证人。
入口 `python -m scripts.research.representation_v2.event_probe`；仅inner事件证据，不更新正式成绩或dist。

S2已完成，固定25%融合未通过gate：严格三折health基线为66TP/20FP/31FN、F1=0.72131，
FD融合为65TP/18FP/32FN、F1=0.72222，仅1/3折改善。**这里的0.72不是正式outer F1**，
正式成绩仍为0.6514285714。候选AP则三折均改善，均值0.78711→0.81136；说明有排序增量，
但尚未稳定转化为事件收益。详见[S2报告](outputs/challenger_70/representation_v2/amendment_a3_events/report.md)。
下一方向优先审计训练内评分尺度/策略稳定性，不根据health漏检直接调阈值，也不立即扩大网络。
完整仓库回归646 passed、5 skipped；无新token缓存、无dist/release变更。

[S3训练内稳定性审计](outputs/challenger_70/representation_v2/amendment_a3_stability/report.md)已完成：
仅使用12份policy OOF产物，无训练/health读取/新F1。基线与运动支路训练权重及评分尺度不同；
运动无效时混入训练先验会压低强基线分数。固定策略下阈值降低.05，三个训练内wrapper均未增加TP，
因此不直接放宽阈值。下一窄假设是“运动无效时保持基线评分”，有效分支维持原融合，并保留随机对照。
本轮164项相关测试通过；策略祖先图补充认证通过。正式F1与dist仍不变。

下一轮[S4中性回退对照](docs/challenge_070_amendment_rv2_a3_neutral.md)已注册：无效运动输入直接使用B，
有效行保持S2评分逐位不变。复用已提交的严格OOF账本，0模型fit；全部三折策略落盘并校验后才读取health。
入口 `python -m scripts.research.representation_v2.neutral_probe`。仍为重复开发人口评估，不是正式outer。

S4已完成、未通过gate：FD中性回退74TP/34FP/23FN、inner health F1=0.72195；基线66TP/20FP，
原S2 FD为65TP/18FP。召回增加但FP成本过高，仅1/3折改善；候选AP也从S2 FD的0.81136降至0.76109。
见[S4报告](outputs/challenger_70/representation_v2/amendment_a3_neutral/report.md)。不推广中性回退；
下一方向是低容量verifier联合使用冻结FD证据与有效性，停止手调统一概率平均/回退常数。
本轮0模型fit、约57秒、175项相关测试通过；正式F1仍为0.6514285714，dist不变。

[S5联合verifier](docs/challenge_070_amendment_rv2_a3_joint.md)已锁定为116D原特征+38D冻结运动表示/质量，
保留同宽随机encoder对照，不再固定混合两种不同训练目标的概率。全部三折policy先冻结，再计算health；
B须回放S2候选、评分、policy与完整指标。入口 `python -m scripts.research.representation_v2.joint_probe`。
S5已完成且未通过gate：B 66TP/20FP、inner health F1 .72131；随机表示70TP/27FP、.72165；
FD联合59TP/13FP、.69822，仅1/3折改善。候选AP虽然提高，但没有转化为事件收益，不推广该方案。
见[S5报告](outputs/challenger_70/representation_v2/amendment_a3_joint/report.md)。成功运行约13.2分钟，
184项相关测试通过，零新神经训练/持久token缓存。正式F1仍为0.6514285714，dist不变。

[S6队友方案启发的几何诊断](outputs/challenger_70/representation_v2/geometry_policy_audit/report.md)已完成：
冻结S2基线分数与策略，仅在policy OOF将双来源最终区间裁到宏/微支持交集。17个区间改变，
三个wrapper均无TP/FP变化、无GT救回/丢失，MAE无一致改善；关闭这一裁边方案，未运行health。
52项相关测试通过、约16KiB证据、无模型训练/新缓存。对方后期实验分支尚未公开，234与153个GT
口径仍待对齐；评分分布差异不能单独证明OOF/refit校准失效。

Installed-artifact hashes, the independent frozen-blend identity check, and
the formal non-exclusive mechanism legend are recorded in
[`docs/challenge_e04_validation_addendum.md`](docs/challenge_e04_validation_addendum.md).

[S7时间顺序对照](outputs/challenger_70/representation_v2/amendment_a3_order/report.md)已完成，未通过gate。
冻结同一FD encoder，训练513参数时序读出，并保留随机encoder与打乱token顺序对照。
三折candidate AP：有序FD .46318、打乱 .48405、原133D线性读出 .54050；有序仅1/3折改善，
raw BCE亦恶化，部分训练轨迹收敛未决。关闭本配置，不将首折的提升推广为总体结论。
约6.2分钟、18000个小head优化步、约679KiB证据，零新token缓存、正式F1及dist均不变。
下一方向是受控的目标域事件适配/源目标联合监督，以严格nested事件F1和TP/FP作主指标，
不再仅靠冻结encoder上的池化或固定概率融合追分；更大TCN/Transformer并未被排除。
本轮全仓回归688 passed、5 skipped（15条既有warnings），发布包隔离运行与接口测试通过。

[S8目标域监督适配](docs/challenge_070_amendment_rv2_a3_adapt.md)已注册：B为原116D，
R为冻结FD的251D联合verifier，F为相同251D但encoder在当前训练分区内经目标事件BCE+
FD源手势监督适配。本轮R不是随机encoder；12个partition分别初始化/训练1000步，
所有policy冻结后才计算health事件F1，基线完整回放S2。入口
`python -m scripts.research.representation_v2.adapt_probe`；不自动outer或改dist。

S8已完成、未通过gate：B 66TP/20FP、inner health F1 .72131；冻结FD 69TP/20FP、.74194；
适配FD 67TP/20FP、.72826。适配未优于冻结对照，且仅1/3折胜B；冻结对照的pooled改善
亦仅来自1/3折，不事后更换主臂推广。两种FD臂短餐均11/22（B 9/22），但逐人F1的10分位数
从B .35333降至 .12，泛化稳定性仍不足。见[S8完整报告](outputs/challenger_70/representation_v2/amendment_a3_adapt/report.md)。
12次独立适配、12000步，约24.2分钟，约17MiB证据、无新token缓存；全仓694 passed、5 skipped。
下一步先做train-only优化/任务耦合审计，不据重复health手调source权重或继续冻结概率融合追分。
正式F1仍 .6514285714，dist/release不变；正式可比提升>.03时提请审批后才更新可执行发布包。

[S9训练侧优化审计](docs/challenge_070_amendment_rv2_a3_optimization.md)固定S8首个policy训练分区，
只生成train OOF候选，重放初始readout，比较固定训练集的encoder/head初始与最终组合及梯度。
不训练新encoder、不生成health评分或新事件F1；用于决定适配是否值得继续，或转向新的时序路线。
S9实测完成：固定train BCE .22184→.26143，AP .76437→.74873，仅3/11人BCE改善；
梯度可达encoder，不能归因为断图。四组梯度诊断全negative，不能证明source损害positive。
见[S9报告](outputs/challenger_70/representation_v2/amendment_a3_optimization/report.md)。
下一步先在train-only比较取消裁剪/扩大batch两个单因素，失败则停止该修复路线。

[S10/S11优化对照规范](docs/challenge_070_amendment_rv2_a3_optimizer_controls.md)：
S10固定11人train，只比较无裁剪batch4与norm1裁剪batch32，各1000步；
训练BCE降低≥.01、至少6/11人不劣、固定source BCE上升≤.03才允许进入S11。
固定优先选择通过的无裁剪臂，否则选择通过的大batch臂；两者失败则转向新时序表示。
S11仅验证唯一封存配置，沿用完整nested事件门槛，不将训练改善当作发布F1。
S10完成：无裁剪改善.00839未过门槛；batch32改善.01963、9/11人不劣，source BCE未恶化。
因此只允许batch32/norm1进入S11。见[S10报告](outputs/challenger_70/representation_v2/amendment_a3_optimizer_controls/report.md)。
S11完整nested事件验证完成：B/R/F inner F1 .72131/.74194/.73743；F为66TP/16FP，
虽比B少4FP且2/3折胜B，未胜冻结FD对照，因此gate失败，关闭此适配配置。
见[S11报告](outputs/challenger_70/representation_v2/amendment_a3_optimization_event/report.md)。
全仓704 passed、5 skipped；正式发布仍.6514285714。现在转向已设计的
[S12冻结手势时序形态](docs/challenge_070_amendment_rv2_a3_gesture_morphology.md)，
不继续向同一适配配置追加参数；S12现已完整完成。
S12实现已完成：固定24D手势形态+116D基线，独立140D verifier，对照冻结FD251D。
S12结果：B/R/F inner F1 .72131/.74194/.72340，F68TP/23FP，比B+2TP/+3FP；
未达+.015且未胜R，因此关闭24D schema，不追加阈值搜索。
见[S12报告](outputs/challenger_70/representation_v2/amendment_a3_gesture_morphology/report.md)。
17项新增测试、162项相关研究测试通过；本次全仓回归721 passed、5 skipped、
31条既有/预期警告，无失败（1207.41s）。正式发布与dist均未改动。

后备路线的[外部预训练模型核查](docs/challenge_070_external_pretrained_review.md)已整理：
HARNET/LIMU-BERT-X等需先通过输入单位、采样合同、域差异与许可门，再注册实验；
当前未下载新权重，也不将公开HAR成绩当作本项目事件F1。

下一路线已登记为[S13完整动作感受野源域研究](docs/challenge_070_amendment_rv2_a3_receptive_field.md)：
固定输入与16D输出，只比较更强126544参数/129点感受野encoder的源域学习。
S13已完成：source-val AP .84304（S0 .84405）、raw BCE .50416（S0 .25704）；
未过固定源域gate，关闭此架构/预算，不进入目标事件验证。
训练末段loss .04666与验证BCE的分离提示过拟合/过度自信，而非容量不足已被解决。
见[S13报告](outputs/challenger_70/representation_v2/amendment_a3_wide_source/report.md)。
22项新增测试、184项表示研究测试通过；本次全仓回归743 passed、5 skipped、
31条既有/预期警告，无失败（1079.33s）。证据已归档，无新token缓存；正式F1仍.6514285714。

[S14外部预训练源域探针](docs/challenge_070_amendment_rv2_a4_foundation.md)已预注册并实现：
固定LIMU-BERT-X的unit-free桥接，与同架构随机表示、冻结FD表示做源域线性读出比较。
只用train子窗口拟合32D PCA后固定池化，不训练encoder，不读取新目标raw或跑outer。
未知物理单位与工业预训练主体身份限制均明确披露；源域结果不等于event F1。
S14已完成：R/F/D source AP .76428/.69747/.85144，raw BCE .27218/.29392/.22677。
固定预训练臂不及随机与FD对照，完整gate失败，关闭此桥接，不进入目标probe。
见[S14报告](outputs/challenger_70/representation_v2/amendment_a4_foundation_source/report.md)。
27项新增、211项相关研究测试通过；修复后全仓770 passed、5 skipped、31条既有/预期警告，
无失败（1029.36s）。已清理研究pyc与本轮约96MiB测试临时占位文件，无持久token缓存；
正式F1仍.6514285714，dist/release未改动。源域负样本覆盖不足保留为未验证假设，暂不启动。

A5已有收益扩大验证见[A5设计](docs/challenge_070_amendment_a5_inner_validation_design.md)。
固定比较116D研究对照、冻结FD251D和S11适配FD251D，在原T24上采用两组预定三折划分；
这是扩大划分而非新增独立人群，原V7/outer继续隔离。分别记录收益、划分稳定性和错误互补，
不再将“适配未胜冻结对照”等同于“相对baseline毫无收益”。设计已获审阅通过，
[实施计划](docs/superpowers/plans/2026-10-10-a5-inner-validation-plan.md)已获审阅通过，代码已实现，
50项相关测试通过；Luna/medium审查发现的启动异常状态问题已修复。
截至2026-10-12，全仓回归在中途停止，没有完整通过记录；manifest与三臂训练尚未启动。
用户要求先讨论Transformer，故A5保持暂停，代码保留，不记为算法失败，不自动恢复。

新的架构设计见[A6双任务Transformer与FD序列扩充](docs/challenge_070_amendment_a6_transformer_design.md)，
书面设计已获审阅通过，[T0/T1实施计划](docs/superpowers/plans/2026-10-12-a6-transformer-t0-t1-plan.md)
已获审阅通过，T0/T1代码及严格inner四臂实验已完成，39项新增针对性测试通过。
全仓基线799 passed、5 skipped、31 warnings；Luna/medium独立审查发现的提前raw读取与
活动计算预算遗漏已按RED→GREEN修复；完成后全仓回归838 passed、5 skipped、31 warnings，
无失败（1033.50s），全部表示研究测试279 passed。manifest与不读取raw内容的T0审计已提交锁定，
全部policy冻结并提交后才开始health验证；完整三wrapper/24名受试者、97个GT的结果如下。

| 对照 | inner event F1 | TP/FP/FN | 相对基线 |
| --- | --- | --- | --- |
| incumbent | .721311 | 66/20/31 | — |
| pooling | .739583 | 71/24/26 | +.018272 |
| TCN | .763819 | 76/26/21 | +.042508 |
| Transformer | .760417 | 73/22/24 | +.039105 |

三新增臂均3/3 wrapper非下降并通过扩大验证gate；TCN短餐recall从9/22提高到13/22，
24名受试者中9名提高、2名下降、13名持平。Transformer为12/22，10名提高、4名下降、10名持平。
这些是反复用于开发的T24结果，不能声称正式突破.70，也不能认定attention优于TCN。
详见[A6 T1报告](outputs/challenger_70/representation_v2/amendment_a6_transformer/t1/report.md)
及[执行收口记录](outputs/challenger_70/representation_v2/amendment_a6_transformer/t1/closeout.md)。
路线为冻结FD局部表示→候选时序复核→全天定位与候选联合，
分别处理漏候选和难负例；FD连续序列扩充单独归因。已核实本地FD压缩包仅含逐点手势X/Y，
未找到独立餐段边界；FD-II餐外0不能作为可靠负例，不能直接混入餐段训练。
用户已确认目标与FD-I来自独立受试者群体；匿名ID本身无法独立证明跨数据身份，证据中保留此限制。
正式F1仍.6514285714，dist/release不变；用户已批准冻结TCN主检验、Transformer次要对照的V7验证。
[V7协议](docs/challenge_070_amendment_a6_v7_validation.md)已预注册；入口和13项针对性测试完成，
全部表示研究测试292 passed。Luna/medium审查的元数据访问边界已澄清，登记前发现的
index/特征表session覆盖差异已修复，限定原特征表890个session。修复后全仓回归851 passed、
5 skipped、31 warnings（1145.64s），T24 refit及V7验证已完成。
复用已提交的T24主体OOF证据选择policy，仅新增10次producer、2次base、2次时序拟合（2000步）。
新增SubsetSource在eligible-truth/raw-array读取前过滤受试者；共享特征表及元数据仍物理整表加载，
V7文件size/mtime可做完整性审计，但冻结前不加载其raw数组。旧加载器可能读取过V7时间戳/mask，
加上该人群曾参与早期CV，V7不属于全新独立队列；不得把结果包装成正式outer成绩。
V7完整7人/33个GT结果：incumbent F1 **.745763**（22/4/11），TCN **.689655**（20/5/13），
Transformer **.724138**（21/4/12）。TCN相对基线-.056108，主检验未通过；Transformer也低于基线。
三臂短餐均4/10；候选可达32/33。TCN为0人改善/2人下降/5人持平，Transformer为0/1/6。
T24的增益未在V7复现，不扩大outer、不更新dist、不自动更换主检验或重调V7阈值。
详见[V7报告](outputs/challenger_70/representation_v2/amendment_a6_transformer/v7/report.md)。
A6原始设计/实施计划作为已锁定实验合同保持原文，不追改其历史状态段。

### A6 冻结 checkpoint 归因诊断

用户批准后，将三个旧T24 checkpoint与最终refit对同一批223个V7候选评分；
只读取已有token、保留各自训练侧归一化，不训练、不调阈值、不解码新事件或生成新F1。
旧TCN对有效真候选的平均motion概率为.488–.736、旧Transformer为.502–.661，
最终refit仅.266/.260；七名受试者的真候选平均分均下降。负候选分数也同步降低，
说明应优先检查最终拟合稳定性和概率迁移，而非简单认定V7受试者无法识别。
最终refit分数重算与封存证据误差≤4.48e-8；检查未发现推理重构失配。
训练人数/组成、种子和样本暴露仍相互混杂，不能把根因直接定为某个坏种子。
下一步建议在严格训练侧OOF做固定多种子对照；不得用V7挑旧checkpoint或把跨wrapper平均伪装成OOF。
详见[冻结诊断报告](outputs/challenger_70/representation_v2/amendment_a6_transformer/checkpoint_audit/report.md)。
本次正式F1及dist/release保持不变，尚未启动新模型训练。
新增测试6 passed，表示研究回归298 passed，全仓回归857 passed、5 skipped（31项已知/预期warnings）；
Luna/medium独立只读审查未发现Critical/Important问题。必要诊断证据约1.4MiB，无新token缓存。

### A6 固定三种子对照（已完成，未通过推进门槛）

[三种子实验合同](docs/challenge_070_amendment_a6_seed_ensemble.md)锁定原T24嵌套划分、
原种子及+100000/+200000两个种子。仅在同一训练分区内等权平均，不选最佳种子，
保留单模型对照；TCN主检验、Transformer次要对照。新增48次神经拟合/48000步，
原候选和基础评分必须先复现，缓存只读复用。V7/outer不重新评分，dist/release不变。
实现及审查修复共11项新增测试RED→GREEN；表示研究309 passed，全仓868 passed、5 skipped，
31项已知/预期warnings。Luna/medium审查的缓存哈希锚定、OOF覆盖与次要对照汇总问题已修复，
登记已提交（56b225a），policy冻结提交e5a8226后完成health，结果提交294b59a。
策略搜索空间/选择流程不变，集成数值策略仅在训练侧OOF选定后冻结。

| T24开发对照 | event F1 | TP/FP/FN |
| --- | ---: | --- |
| incumbent | .721311 | 66/20/31 |
| 原单TCN | .763819 | 76/26/21 |
| 三种子TCN | .760870 | 70/17/27 |
| 原单Transformer | .760417 | 73/22/24 |
| 三种子Transformer | .752688 | 70/19/27 |

TCN集成相对原单模型减少9个FP，但损失6个TP，F1下降.002950；Transformer下降.007728。
两者均仅1/3 wrapper相对原单模型非下降，主检验失败，不扩大V7/outer验证。
结果不支持直接用概率平均解决最终refit失配，但不能据此排除种子波动；
训练人数/组成、固定步数下样本暴露、分数尺度与policy迁移仍需分开归因。
优先考虑下一份训练侧限定设计来检验这些因素，不在本轮追调融合权重或阈值。
完整性复核通过：120/24/48次producer/base/neural拟合、48000步，CUDA神经拟合累计约16.90分钟；
证据与模型22.45MiB，无新增token缓存。正式F1仍.6514285714，dist/release不变。
详见[集成报告](outputs/challenger_70/representation_v2/amendment_a6_transformer/seed_ensemble/report.md)
及[执行记录](outputs/challenger_70/representation_v2/amendment_a6_transformer/seed_ensemble_execution.md)。
