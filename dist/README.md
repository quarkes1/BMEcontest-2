# Distribution Workspace

## 最终提交（2026-10-28）

最终官方推理锁定为 `official_deployment_f1_policy_v3`，反馈F1=0.533333（TP56/FP42/FN56）。
直接送交仓库根目录 `submission/EatingDetection.exe`，不放ZIP；本目录 `submission/` 是同一EXE加
完整源码、模型、可视化与详细复现说明。最终包重建使用 `python -B scripts/build_final_submission.py`，
校验加 `--verify-only`，不使用历史构建命令选择模型。

历史run160afaf81debf1ee的.651428仅保留本地训练复现；`inference/`、`event_stack/`仍为该历史版本，
不是最终评分程序。最终选择见根目录 `release/final_submission.json`；以下旧发布和试测记录保留备查，不再启动实验。

`dist/` 是发布与交付工作区。算法唯一真源在仓库 `src/pipeline/`；`dist/` 下的
算法生成物由仓库脚本重建，不手工修改。`submission/README.md`允许维护提交说明；
构建会保留其中的用户修改，其初始模板位于`scripts/official_submission_readme.md`。

```text
dist/
├── inference/      # canonical 独立推理包（生成物）：原始 collect_data*.txt → 预测 JSON；
│                   # 附 serve.py 本地推理桥（可视化同源调用）
├── visual/         # 团队可视化应用（React+TS+Vite）：源码在 visual/app/，
│                   # 生产构建为自包含单文件 index.html（双击即用，无任何依赖）
├── submission/     # 官方EXE + 模型源码/权重 + visual/独立应用 + 无Python依赖start.bat
├── official_trials/ # 独立送测EXE：与正式发布隔离，版本与风险见各目录README
├── examples/       # 合成安全示例 example_prediction.json（符合 schema，无真实数据）
├── schema/         # 稳定契约 prediction.schema.json（v1.0）
├── start.bat       # 竞赛一键启动器：本地推理服务 + 浏览器
└── event_stack/    # 旧 serialized-payload 运行时（发布证据，保留，勿手改）
```

发布线：`160afaf81debf1ee`（严格 subject-disjoint nested 五折开发 F1 = 0.6514285714，
TP/eligible/pred = 114/153/197）。发布证据与模型在 `models/event_stack/<run_key>/`，
指针在 `release/event_stack_incumbent.json`；正式发布包必须与它们一致。`official_trials/`
是明确隔离的试测例外，不会改写正式发布指针或自动接入前端。

## 独立送测（2026-10-22）

本轮只需向组委会发送选定目录内的 `EatingDetection.exe`，不用附源码或模型。无参数官方
接口与下文一致，CPU单文件自包含；不是传统所有库静态链接，运行时会解包内置依赖。

- `official_trials/consistent_refit_v1/`：上一轮一致性重训试测版，保持不变；最新用户转述
  官方反馈49TP/27FP/63FN、F1=.521277，收件SHA及隐藏主体一致性未独立核验。
- [deployment_f1_policy_v3](official_trials/deployment_f1_policy_v3/README.md)：本轮主试测，
  最终阈值 .45827，优先此版。
- `official_trials/deployment_f15_policy_v3/`：阈值 .26383 的高召回风险对照；本地F1下降，
  不是性能升级或正式提交替换。

两份新EXE的模型字节均与上一轮试测相同，只更改最终事件阈值。完整五折政策开发诊断中，
主档TP+10/FP+23、F1+.00077（基本持平）；高召回档TP+16/FP+43、F1−.00666。
这些不是最终EXE的无偏CV成绩；两份新EXE官方效果均未知。交付检查以
[R3交付记录](../docs/official_recovery_policy_delivery_v3.md)为准，不因目录存在而视为完成。

## 重建命令（均需仓库根目录运行）

```bash
python scripts/build_inference_distribution.py     # 重建 dist/inference/
python -m pip install -r scripts/requirements-build.txt  # 仅构建机
python scripts/build_submission.py                 # 重建官方单文件EXE submission
python scripts/build_competition_exe.py --verify-only   # 核对最终提交文件与SHA
```

## 官方提交EXE（2026-10-15接口）

直接压缩`submission/`内容提交；EXE应在ZIP顶层。`EatingDetection.exe`无参数启动，以自身
目录读取同级`ACC_test_data`，通过官方DataReader/InfoReader处理所有人的全部TXT，使用
save_result输出同级`results.csv`，只有`externalid,beforeTime,afterTime`三列原始毫秒时间。
每次成功运行覆盖完整结果，零预测也输出表头。目标Windows10/11 x64无需Python/pip/网络。
onefile内置依赖并在临时目录解包DLL，不是所有库传统静态链接；详情见submission/README.md。
不含测试数据/金标准/旧结果；模型仍是正式run160afaf81debf1ee，未混入研究分支模型。

可视化与JSON推理接口保持独立，仍使用下面的`start.bat`和`inference/`，不要将历史四列
CSV或JSON当成官方三列结果。源码复现包仅在`--source-bundle`时生成到另一个指定目录。

## 一键启动（可视化 + 本地推理服务）

```text
双击 dist/start.bat
```

启动器基于脚本相对路径定位发行目录，启动 `dist/inference/serve.py`（仅绑定
127.0.0.1，默认端口 4173），同源提供 `dist/visual/` 静态应用与 `/api/*` 推理接口，
就绪后自动打开浏览器；在页面中选择 `collect_data*.txt` 文件/文件夹即可运行
canonical 推理并查看结果。

## 快速推理（inference 包）

```bash
cd dist/inference
python -m pip install -r requirements.txt
python predict.py path/to/collect_data1_2_3.txt --output prediction.json
python predict.py path/to/subject-folder --output prediction.json --include-timeline
python serve.py --open     # 本地推理桥 + 可视化（等价于 dist/start.bat）
```

`--device cpu` 支持；当前发布没有经过审计的 CUDA 适配器，`--device gpu/cuda`
会被明确拒绝（退出码 2），不会伪装成 CPU 成功。输出 JSON 遵循
`dist/schema/prediction.schema.json`。

## legacy 运行时（保留审计，勿用于新集成）

`dist/event_stack/` 是旧的 **serialized-payload** 运行时（输入为预计算的
63/47/116 维候选特征 JSON，不读取原始传感器文件），由
`scripts/package_event_stack.py` 生成，文件集与哈希由
`dist/event_stack/runtime_manifest.json` 锁定（**不得手工添加/删除其中任何文件**）。
它是当前 release 的部署 bundle 的可独立运行副本，也是发布证据链的一部分；
新集成一律使用 `dist/inference/`。

- 输入契约：`feature_schema`（schema v2：macro 63 / micro 47 / verifier 116，
  含 Context-v1 60 列）、`schema_hash`、`sessions[].candidates[]` 有限值数组；
  未知字段、NaN/Inf、宽度或哈希不匹配均拒绝。
- 输出：`{"events": [{"sid","start_ms","end_ms","score"}], "resolved_device": "cpu"}`，
  稳定排序；`subject_id` 只影响冻结的准入/预算，不出现在输出中。
- 运行时 ABI：Python 3.11.x + `event_stack/requirements.txt` 的精确 pin
  （numpy 2.4.6 / joblib 1.5.3 / scikit-learn 1.9.0 / lightgbm 4.7.0），
  加载任何模型前校验，不匹配即拒绝。

历史滑窗/检测即排序对照产物（旧 predict 脚本、滑窗模型权重与只读 src 副本）
已按交付清理移除，可从 git 历史检出；删除台账见
`tests/fixtures/deletion_manifest.json`，审计记录见 `docs/repository_cleanup_audit.md`。
