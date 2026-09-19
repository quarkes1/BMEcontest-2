# 竞赛提交包

本目录由 `python scripts/build_submission.py` 从原仓库自动生成。
请勿手工修改任何生成文件；需要变更时重建整个包。

## 可视化应用打开方式

**方式一 · 离线查看（零依赖，任何机器）**：双击 `visual/index.html`。
自包含单文件网页（无需 Node / Python / 网络），内置演示数据，并可用
"Open existing prediction" 加载已有的 `prediction.json`（可附带 `motion.json` +
`motion.bin`）查看事件、时间轴、原始 IMU 与三维动作回放。本包内
`models/event_stack/<run_key>/deployment` 之外不含任何真实受试者数据，演示页同样
只使用合成数据。

**方式二 · 完整分析（本地 canonical 推理）**：双击 `start.bat`。
启动本地推理服务（仅绑定 127.0.0.1）并自动打开浏览器；在页面中选择
`collect_data*.txt` 文件或所在文件夹，即可由本包的 canonical Predictor 完成推理，
返回预测契约与运动遥测（浏览器不做任何特征/阈值/解码计算）。若缺少 Python 依赖，
`start.bat` 首次运行时会自动执行 `pip install -r requirements.txt`（无需任何交互），
Python 缺失 / 版本过旧 / 安装失败 / 端口占用都会给出明确提示。

## 包结构

```text
dist/submission/
├── start.bat                      # 一键启动器：本地推理服务 + 浏览器（方式二）
├── main.py                        # 命令行入口：--raw / --official-input（默认 CSV 输出）
├── README.md                      # 本文件
├── requirements.txt               # 精确依赖 pin（来自 deployment manifest）
├── app/                           # 推理运行时与本地服务
│   ├── serve.py                   # 本地推理桥（/api/health、/api/upload、/api/analyze）
│   └── event_stack/               # canonical 运行时（机械 vendored）
├── meta/                          # 包元数据
│   ├── manifest.json              # 逐文件 SHA-256 与 source/model 闭包
│   └── feature_schema.json        # 特征 schema v2（macro 63 / micro 47 / verifier 116）
├── models/
│   └── event_stack/<run_key>/
│       ├── deployment/            # 冻结推理模型（macro/micro/verifier + policy）
│       ├── outer-fold-0..4/       # 五折 outer-fold evidence bundle
│       ├── promotion_summary.json
│       └── promotion_attestation.json
├── visual/                        # 可视化应用（自包含单文件 index.html + 前端源码）
├── src/                           # canonical 算法源码（复现与审阅用）
├── scripts/                       # 训练/评估/发布/构建全链脚本
├── tests/                         # 单元/集成/parity/release 测试
├── schema/                        # prediction.schema.json（预测契约 v1.0）
├── examples/                      # example_prediction.json（合成安全示例）
├── release/                       # 发布指针 event_stack_incumbent.json
└── outputs/crossfit/              # 五折证据（summary + 逐折 JSON/diagnostics）
```

## 命令行推理（绕过界面直接调用）

```bash
python -m pip install -r requirements.txt
python main.py --raw path/to/collect_data1_2_3.txt --output result.json
python main.py --raw path/to/subject-folder --output result.json --include-timeline --include-candidates
```

## 可用接口

1. **一键可视化（推荐）**：`start.bat` —— 本地推理服务 + 浏览器界面（见"可视化应用打开方式"）。
2. **本地 HTTP 桥**：`python app/serve.py [--port 4173] [--visual-dir visual]` ——
   接口：`GET /api/health`、`GET /api/capabilities`、`POST /api/upload`、
   `POST /api/analyze`、`GET /api/artifacts/<id>/<n>/motion.bin`，并同源托管
   `visual/` 静态页面。仅绑定 127.0.0.1。
3. **命令行（本包）**：`main.py --raw INPUT --output OUTPUT [--include-timeline]
   [--include-candidates] [--device cpu]`；输出为 `schema/prediction.schema.json`
   （v1.0）定义的预测文档（events / 可选 timeline、candidates、gaps）。
4. **Python API（本包）**：
   ```python
   import sys; sys.path.insert(0, "app")           # event_stack 运行时位于 app/
   from event_stack.inference import Predictor
   predictor = Predictor.from_bundle("models/event_stack/<run_key>/deployment",
                                     run_key="<run_key>")
   result = predictor.predict_file("path/to/collect_data1_2_3.txt")
   ```
5. **可视化**：`visual/` 是团队可视化应用（用法见 `visual/README.md`）；它只消费
   预测 JSON 与运动遥测（见 `schema/`、`examples/`），不重实现任何算法决策。

## 测试接口说明

由于目前本组没有在官方赛题及官网找到具体的测试接口信息，故该功能目前默认输出包含进食开始，结束的csv。其中每一行 为一次检测到的进食事件的起始与终止时。 不指定输出位置时默认放在./predict中。--cls会清空./predict使之只包含当前预测的结果。

```bash
python main.py --official-input INPUT_FILENAME.txt  # 输出预测结果到 ./predict/predict_INPUT_FILENAME.csv

python main.py --official-input INPUT_FILENAME.txt --output disignated/fold/name.csv   # 输出预测结果到指定的位置

python main.py --official-input  INPUT_FOLD --output disignated/fold/  # 对包含多个数据文件的文件夹进行预测，目录组织为 
INPUT_FOLD
    ├── sensorData-...
    |     ├── collect_data...
    |     └── info.json
    ├── sensorData-...
    |     ├── collect_data...
    |     └── info.json
    └── ...
```

**CSV 格式**（UTF-8 带 BOM，Excel 可直接打开；按开始时间升序）：

| 列 | 含义 |
|---|---|
| `start_time` / `end_time` | 本地时区 ISO 时间，毫秒精度（如 `2026-07-19 12:03:24.500`） |
| `start_ms` / `end_ms` | 与预测文档一致的 epoch 毫秒时间戳 |

未检出事件时输出仅含表头。`--output` 指向目录（或以 `/`、`\` 结尾）时，在其中写入
`predict_<输入名>.csv`。

## 自定义 adapter（替换官方接口）

官方真实契约发布后，无需改动算法、模型与证据链，只需实现一个 adapter 并注册。
接口定义在 `app/event_stack/inference/competition_adapter.py`：

1. **实现接口**（``load``：官方输入 → canonical raw 路径；``dump``：预测文档 → 官方输出）：

   ```python
   class MyOfficialAdapter:
       def load(self, path):
           """返回 (collect_data*.txt 文件或目录, 可选受试者 id)。"""
           return translated_input, subject_id

       def dump(self, prediction, path):
           """把 canonical 预测文档写成官方要求的格式。"""
           ...
   ```

2. **注册并使用**（在包根运行；注册表是进程内的）：

   ```python
   from pathlib import Path
   import sys
   sys.path.insert(0, "app")
   from event_stack.inference import Predictor
   from event_stack.inference.competition_adapter import register_adapter, registered_adapter

   register_adapter("official", MyOfficialAdapter())
   predictor = Predictor.from_bundle("models/event_stack/<run_key>/deployment", run_key="<run_key>")
   adapter = registered_adapter("official")
   raw_input, subject_id = adapter.load(Path("官方输入"))
   adapter.dump(predictor.predict_file(raw_input, subject_id=subject_id), Path("官方输出"))
   ```

3. **命令行**：`main.py --official-input INPUT --adapter official`；未注册的名字会**显式拒绝**
   （退出码 2，`not registered`），绝不会猜测官方格式。默认 adapter 为 `csv`（见上节）。

adapter 契约要点：`load` 只做输入翻译（不做特征/阈值/解码）；`dump` 只做输出序列化
（文档结构见 `schema/prediction.schema.json`）。长期部署时把 adapter 写入
`competition_adapter.py` 并重建本包即可。

## 复现与证据

### 1. 数据集组织

将竞赛数据集按以下结构放在本包根目录的 `Data/`（入口脚本按包根相对定位）：

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

入口脚本按包根相对定位（`src/config.py`：`DATA_DIR = <包根>/Data`），其中
`SENSOR_DIR` 固定为竞赛发放的目录名（`t_zsstnnrj_sensororiginaldata_system附件0826_1857`）；
如你的数据目录名不同，修改 `src/config.py` 中该常量即可。文件夹输入（`--official-input`
指向批次目录）会逐层发现 `sensorData-*/collect_data*.txt`。

### 2. 免训练验证发布链（最快；本包已内置全部证据，无需数据集）

```bash
python -m pip install -r requirements.txt
python scripts/reproduce_release.py --run-key <run_key>
```

验证发布指针、attestation、五折证据、deployment bundle 与两份 manifest 一致；
通过时输出冻结的聚合指标。

### 3. 从数据集重建缓存并复评（完整复现）

```bash
# micro 特征缓存（首次运行会解析 Data/ 下全部会话并写入 cache/，耗时与磁盘占用较高）
python scripts/build_micro_features.py --fold all --split all --workers 8

# 严格 subject-disjoint nested 五折评估
python scripts/evaluate_event_stack.py --fold all --inner-splits 4 --no-tcn --workers 5 \
    --micro-enabled --candidate-control-enabled --admission-minimum-recall 0.80 \
    --context-features v1 --summary-alias outputs/crossfit/context_v1_summary.json
```

### 4. 训练与晋级（可选，消耗更大）

```bash
python scripts/train_event_stack.py --summary outputs/crossfit/context_v1_summary.json
python scripts/release_event_stack.py --summary outputs/crossfit/context_v1_summary.json
```

### 证据链

- `models/event_stack/<run_key>/promotion_attestation.json` 绑定 canonical summary、
  严格五折与每个 manifest 的 SHA-256；
- `meta/manifest.json` 记录本包每个文件的 SHA-256（`hashes`）与 source/model 闭包，
  可在任意环境复核；
- `outputs/crossfit/` 为正式评估证据（summary + 逐折 JSON/diagnostics）。

## 运行时 ABI

Python 3.11.x，配合 `requirements.txt` 中来自 deployment manifest 的精确 pin：
numpy、joblib、scikit-learn、lightgbm。支持 `--device cpu`；`--device gpu/cuda`
会被拒绝，因为当前发布没有经过审计的 CUDA 适配器。
