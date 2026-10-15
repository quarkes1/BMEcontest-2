# 赛题2 提交包

本包适配官方测试接口的提交包，EatingDetection.exe 可以独立运行

## 运行

将官方完整测试数据放在EXE同级，保持以下结构：

```text
测试目录/
├── EatingDetection.exe
├── ACC_test_data/
│   ├── userinfobean.txt
│   ├── HNU00001/
│   │   ├── collect_data0_....txt
│   │   └── collect_data1_....txt
│   └── HNU00002/
└── results.csv                 # 运行成功后生成，不是提交包预置数据
```

双击EatingDetection.exe，或在命令行直接运行，不需要参数。程序总是以EXE所在目录定位ACC_test_data和results.csv，不依赖启动时的工作目录。每个人的全部TXT通过官方DataReader.iter_data读取，InfoReader读取信息，save_result保存预测。

输出只包含预测进食事件，CSV列名及顺序固定：

```csv
externalid,beforeTime,afterTime
```

编号对应个人目录并按官方规则大写/移除空白；时间为原始采集轴13位Unix毫秒整数，开始小于结束。没有进食预测也会产生表头。UTF-8 BOM便于Excel读取。

每次成功运行替换本轮全部结果，不追加旧结果。退出码0为成功；退出码2为失败。失败时不会发布半成品，但之前成功的results.csv可能仍在，请勿将旧结果当作本次结果。请在命令提示符中运行以查看错误。

可选依赖/模型自检（不读测试数据）：

```text
EatingDetection.exe --self-check
```

## 模型与依赖

EXE采用PyInstaller onefile自包含打包，内置CPython、科学计算库、LightGBM和部署模型。运行时在Windows临时目录释放内置DLL，结束后由bootloader清理；这不是所有库的传统静态链接，也不依赖机器预装Python。首次启动解包可能需要数秒，请勿连续重复启动。需有Windows临时目录及EXE目录的写入权限。

build_manifest.json记录EXE/包内文件SHA-256、正式模型文件哈希、源代码标识与依赖版本。official_api为原样官方接口来源。必要的第三方版权与许可声明已合并在THIRD_PARTY_NOTICES.txt中，无需运行；不设置单独的licenses目录。

## 源代码与模型结构

本包同时提供可独立运行的EXE和当前模型Python源代码

```text
submission/
├── EatingDetection.exe           # 内置模型与依赖，无参数启动
├── main.py                       # 相同官方接口的源码入口
├── requirements.txt              # 源码运行和复现的精确依赖版本
├── src/                          # 特征、候选、评分、策略及数据读取源码
├── scripts/                      # 当前模型训练、评估、验证与EXE构建
├── models/event_stack/           # 当前部署权重与五折模型证据
├── release/                      # 正式模型指针
├── outputs/crossfit/             # 冻结评估记录，非逐样本标签
├── official_api/                 # 官方输入输出接口与来源校验
├── build_manifest.json           # 全包逐文件SHA-256
└── THIRD_PARTY_NOTICES.txt        # 所用库的第三方声明文件
```

算法为双时间尺度IMU特征模型：宏窗口HGB和微窗口LightGBM产生候选，56维基础特征与60维Context-v1组成116维事件表示，经逻辑回归/LightGBM融合评分、候选准入与冻结事件策略输出。模型权重和阈值不因接口适配变化。

## 源码运行与复现

以下命令在解压后的submission目录执行。仅使用源码时需安装Python；运行EXE不需要安装依赖。

### 1. 创建环境，运行相同官方推理接口

使用Windows x64 Python 3.11（构建环境为3.11.15）：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py --self-check
.\.venv\Scripts\python.exe main.py
```

源码同样读取本包目录下的ACC_test_data，输出本包目录下的results.csv，不依赖启动时工作目录。原始训练数据和官方测试数据均不随包提供。

### 2. 免训练验证发布证据

```powershell
.\.venv\Scripts\python.exe scripts/reproduce_release.py --run-key 160afaf81debf1ee
```

验证发布指针、部署模型、五折模型、评估汇总与签名哈希。此命令复核已有证据，不等价于从数据重新计算指标。多次开发使用的outer结果只是开发证据，不是未接触测试集的泛化保证。

### 3. 从原始训练数据重建特征并评估

将训练数据放在本包Data目录，组织为以下形式：

```text
Data/
├── t_<...>_sensororiginaldata_<...>.csv        # 会话索引（受试者 ↔ sensorData 映射）
├── t_<...>_mealinfo_<...>.csv                  # 餐次标注（受试者 / 起止时间 / 手别）
└── t_<...>_sensororiginaldata_system<...>/      # 会话数据
    ├── sensorData-<时间戳>-<uuid>/
    │   ├── collect_data<N>_<起>_<止>.txt        # 53 列 TSV：3 个时间戳 + PPG×44 + ACC×3 + GYRO×3
    │   └── info.json
    └── ...
```

```powershell
.\.venv\Scripts\python.exe scripts/build_micro_features.py --fold all --split all --workers 8
.\.venv\Scripts\python.exe scripts/evaluate_event_stack.py --fold all --inner-splits 4 --no-tcn --workers 5 --micro-enabled --candidate-control-enabled --admission-minimum-recall 0.80 --context-features v1 --summary-alias outputs/crossfit/context_v1_summary.json
```

评估使用严格受试者隔离nested五折。首次生成cache特征，耗时与磁盘用量依数据规模而定；不要将cache、Data目录或运行结果加入提交ZIP。仅验证交付包时无需执行本步骤。重训可能受数值差异影响，不承诺joblib文件逐字节一致。

### 4. 重训部署模型与重建EXE（可选）

在另一个工作副本中操作，避免覆盖本包冻结权重与证据：

```powershell
.\.venv\Scripts\python.exe scripts/train_event_stack.py --summary outputs/crossfit/context_v1_summary.json --output-root models_reproduced
.\.venv\Scripts\python.exe -m pip install -r scripts/requirements-build.txt
.\.venv\Scripts\python.exe scripts/build_competition_exe.py
```

训练命令验证五折证据后全量重拟合模型到models_reproduced，不覆盖正式发布；上面的EXE构建仍使用正式发布指针指向的冻结权重，不会自动改用重训输出。新包生成在工作副本的dist/submission，不以工作副本根目录为清理目标。构建会校验权重、隔离环境自检并生成文件清单，无须原仓库Git历史。不同机器生成的EXE不承诺二进制哈希一致。

