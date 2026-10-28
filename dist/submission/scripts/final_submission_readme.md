# 赛题2 最终提交包

本包提供可独立运行的 Windows EXE、与其相同模型的 Python 推理源码、离线可视化，以及历史本地评估的训练复现代码。

## 1 版本和成绩口径

| 用途 | 版本 | 结果 |
|---|---|---|
| 最终推理和可视化 | `official_deployment_f1_policy_v3` | 官方反馈 TP56、FP42、FN56；F1 **0.533333**，灵敏度0.500000，阳性预测率0.571429 |
| 历史本地训练复现 | `160afaf81debf1ee` | 受试者互斥嵌套五折开发 F1 **0.6514285714**；TP114、FP83、FN39，153个可评估真实事件 |

官方反馈使用112个有效真实事件，历史开发评估对窗口覆盖设置了准入条件。两者数据与分母不同，不能直接比较，也不能将0.65作为本提交程序的官方成绩。组委会未提供隐藏标签，源代码运行可复现同一权重的推理，但不能在缺少官方原始测试数据和标签时自行重算官方F1。

最终模型选择在 `release/final_submission.json`；`release/event_stack_incumbent.json` 仅保留历史复现证据。默认 `main.py`、`serve.py`、`start.bat` 和 EXE 均用于最终推理，不能把历史复现权重替换为当前默认权重。

EXE原样保留已获上述反馈的程序，未重新编译：

- 大小：62,893,974字节。
- SHA-256：`fa6987b6244fe73c76ee741ea352fcc74e8e4de06de332812e99b1f1d963aa44`。
- 官方反馈由参赛方收到并记录；官方实际收件文件的哈希及三次测试人群一致性未独立核验。

## 2 无需安装环境的官方推理

适用 Windows 10/11 x64。EXE内置模型、CPython及必要依赖，CPU推理，无需Python、pip、CUDA、Node或网络。运行时会在系统临时目录解包DLL，不是所有依赖的传统静态链接；请保证临时目录及EXE目录可写，首次启动耐心等待，不要连续重复启动。

将组委会完整数据放在 EXE 同级，保持官方结构：

```text
测试目录/
├── EatingDetection.exe
├── ACC_test_data/
│   ├── userinfobean.txt
│   ├── HNU00001/
│   │   ├── collect_data0_....txt
│   │   └── collect_data1_....txt
│   └── HNU00002/
│       └── collect_data0_....txt
└── results.csv                   # 成功运行后生成，不随包提供
```

双击 EXE，或在命令提示符中运行：

```text
EatingDetection.exe
EatingDetection.exe --self-check
```

官方评分使用第一条无参数命令。自检不读取测试数据；输出 `run_key` 应为 `official_deployment_f1_policy_v3`，`device` 为 `cpu`。程序以 EXE 自身目录定位输入输出，不依赖启动时工作目录；同一人的全部TXT由官方 DataReader 读取，InfoReader读取信息，结果经官方 save_result 写出。

输出仅包含进食事件，CSV 列名及顺序固定：

```csv
externalid,beforeTime,afterTime
```

用户编号按官方规则大写并移除空白；起止时间为原始采集轴13位 Unix 毫秒整数，开始必须小于结束。无事件时只输出表头；UTF-8 BOM便于Excel读取。每次成功运行替换整份结果，不追加旧结果。退出码0表示成功，2表示失败；失败时不发布半成品，但上次成功的results.csv可能仍在，请勿将旧结果当作本轮输出。不要让多个进程同时写同一个结果文件。

核对文件：

```powershell
Get-FileHash .\EatingDetection.exe -Algorithm SHA256
```

## 3 可视化应用

双击 `start.bat`，启动仅绑定本机127.0.0.1的服务并打开浏览器，无需Python或联网。默认端口4173；选择原始TXT文件或文件夹后，可查看进食事件、证据时间轴、原始IMU及三维姿态回放。

```text
start.bat
start.bat --port 4180
EatingDetection.exe --visual --open
EatingDetection.exe --visual --port 0 --no-open
```

保持控制台窗口运行，Ctrl+C退出；关闭网页不会关闭服务。正常退出清理本次临时上传文件。网页模式不读写官方results.csv，官方评测仍无参数运行EXE。直接打开 `visual/index.html` 可查看合成演示与已有JSON；原始TXT推理需要本地服务。Model页分别显示当前官方成绩和历史本地复现成绩，演示数据不是竞赛受试者数据。前端源码和重建方式见 `visual/README.md`。

## 4 源码与权重结构

```text
submission/
├── EatingDetection.exe              # 官方最终推理，依赖和权重内置
├── main.py                          # 同一模型的官方源码入口
├── serve.py、start.bat               # 源码服务入口、无需Python的EXE启动器
├── README.md、requirements.txt
├── src/                             # 解析、特征、候选、复核与事件输出
├── scripts/                         # 训练复现、检查和最终包重建
├── models/event_stack/
│   ├── official_deployment_f1_policy_v3/deployment/  # 当前最终推理模型
│   └── 160afaf81debf1ee/             # 历史本地五折及部署复现证据
├── release/final_submission.json     # 最终推理选择与文件SHA
├── release/event_stack_incumbent.json # 历史本地复现指针
├── outputs/crossfit/                 # 历史评估记录，不含原始数据
├── official_api/                    # 原样官方输入输出接口及来源
├── visual/、schema/、examples/       # 网页、JSON契约、合成示例
├── build_manifest.json              # 全包文件SHA-256清单
└── THIRD_PARTY_NOTICES.txt           # 合并的第三方声明
```

当前算法为双尺度IMU事件检测：宏窗63维HGB、微窗47维LightGBM产生候选，56维基础特征加60维Context-v1构成116维复核表示，由LR/LightGBM融合后输出事件。最终参数固定：LR权重0.25、LGBM权重0.75、候选准入0.2、NMS IoU0.3、每主体候选总上限6、事件阈值0.4582664763345077。当前不启用PPG或Transformer，不支持GPU推理；不要将设备参数伪装为GPU执行。

本包无训练数据、官方测试数据、缓存或旧预测结果，不含未完成实验脚本和独立licenses目录。第三方声明保留在合并文件中。必要运行模型与历史复现资产均带哈希校验。

## 5 使用源码复现最终推理

仅源码运行需要安装环境；运行EXE无需执行下面命令。使用 Windows x64 Python 3.11（原构建环境3.11.15），在包外创建虚拟环境，避免改变提交文件清单：

```powershell
py -3.11 -m venv ..\.venv-bme-final
..\.venv-bme-final\Scripts\python.exe -m pip install -r requirements.txt
..\.venv-bme-final\Scripts\python.exe -B main.py --self-check
..\.venv-bme-final\Scripts\python.exe -B main.py
..\.venv-bme-final\Scripts\python.exe -B serve.py --open
```

命令在解压包根目录执行。源码同样读取该目录的ACC_test_data并输出results.csv；也可以从其他工作目录用main.py绝对路径启动。入口强制校验最终模型的全部文件SHA，缺少选择文件或权重被修改会报错，不会静默退回历史模型。当前推理仅复用冻结权重，不需要也不会重新训练。

检查完整最终包：

```powershell
..\.venv-bme-final\Scripts\python.exe -B scripts/build_final_submission.py --verify-only
```

文件清单校验要求包内容不变；运行产生的ACC_test_data/results.csv、虚拟环境、缓存等不属于交付清单。请在独立测试副本中运行数据，再对未改动的提交包做清单检查。

## 6 历史0.65本地结果复现

本节仅复现 `160afaf81debf1ee` 的历史开发评估，不替换最终推理选择，不声称重新得到官方0.533333。

### 6.1 不训练，核对已有证据

```powershell
..\.venv-bme-final\Scripts\python.exe -B scripts/reproduce_release.py --run-key 160afaf81debf1ee
```

核对历史指针、五折模型、部署模型、评估汇总及签名哈希。该命令复核已有证据，不等价于从数据重新计算F1；反复开发使用的外层结果也不是全新独立测试集的保证。

### 6.2 从训练数据生成特征并重新评估

在另一个工作副本中准备原始训练数据，避免将个人数据、缓存或运行产物混入最终提交包：

```text
Data/
├── t_<...>_sensororiginaldata_<...>.csv
├── t_<...>_mealinfo_<...>.csv
└── t_<...>_sensororiginaldata_system<...>/
    └── sensorData-<时间戳>-<uuid>/
        ├── collect_data<N>_<起>_<止>.txt
        └── info.json
```

传感器TXT为53列TSV：3个时间戳、44列PPG、ACC三轴与GYRO三轴。模型不把PPG包率当作生理波形采样率；当前正式推理仅使用IMU。

```powershell
..\.venv-bme-final\Scripts\python.exe -B scripts/build_micro_features.py --fold all --split all --workers 8
..\.venv-bme-final\Scripts\python.exe -B scripts/evaluate_event_stack.py --fold all --inner-splits 4 --no-tcn --workers 5 --micro-enabled --candidate-control-enabled --admission-minimum-recall 0.80 --context-features v1 --summary-alias outputs/crossfit/context_v1_summary.json
```

首次生成特征耗时并占磁盘，需自行提供相同版本的训练数据。各fold按受试者隔离，阈值在训练侧选择。重训会受数值库和环境差异影响，不保证joblib逐字节一致；更不能将历史eligible评估的F1与官方完整有效GT评分混为一谈。

### 6.3 重训历史部署模型或重新组装最终包

```powershell
..\.venv-bme-final\Scripts\python.exe -B scripts/train_event_stack.py --summary outputs/crossfit/context_v1_summary.json --output-root models_reproduced
```

这只训练历史复现模型到另一个目录，不会自动修改最终模型指针或本包EXE。当前0.533333模型的精确推理复现使用本包冻结权重；本地没有官方隐藏标签，无法从头重训保证同一官方分数。

最终包重建使用：

```powershell
..\.venv-bme-final\Scripts\python.exe -B scripts/build_final_submission.py
```

应在独立工作副本中运行。该命令复用已验证SHA的原始EXE与最终模型，重新生成工作副本的dist/submission，并将相同EXE放到工作副本的submission/，不重新编译或训练。旧 `build_competition_exe.py`/`build_submission.py` 是历史0.65发布工具，不应用于重新选择最终推理模型；最终交付以本节命令及 `final_submission.json` 为准。包内容变动会改变全包清单，但原EXE必须保持上述SHA。

## 7 验收范围与限制

验收包含EXE与源码自检、中文路径、不同工作目录、真实同人多文件CSV逐字节一致、空输入、错误输入不覆盖已有结果以及可视化模型身份。原始官方示例已不在本机，不能声称重新跑过该示例；自检和接口一致性也不代表在新数据上保证F1。官方结果以组委会反馈为准。

## 8 源码注释与文件校验

本包的项目源码注释及文档字符串采用中文，保留接口字段、数学符号、类型检查指令和必要的技术标识。第三方软件的版权及许可证声明保留原文。源码中的开发阶段指令改为模块功能说明。构建入口生成的注释也采用中文。

此次仅调整说明文字及命令行帮助，不改变模型参数、特征计算、候选处理或事件解码。正式 EXE 不重新编译，仍使用官方 F1 为0.533333的固定版本。`build_manifest.json`记录当前源码文件的哈希；源码注释变化与原 EXE 的历史构建记录分别保留。
