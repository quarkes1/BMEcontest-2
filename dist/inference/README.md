# 独立 event-stack 推理包

本目录由当前 promoted canonical release 自动生成，请勿手工修改 `event_stack/`；
重建命令：`python scripts/build_inference_distribution.py`。

## 包结构

```text
dist/inference/
├── predict.py            # 独立推理入口（原始 collect_data*.txt → prediction JSON）
├── event_stack/          # canonical 源码的机械 vendored 副本（勿手改）
├── models/               # 冻结 deployment bundle（macro/micro/verifier 模型 + policy）
├── manifest.json         # 逐文件 SHA-256 与 source/model 闭包
├── feature_schema.json   # schema v2（macro 63 / micro 47 / verifier 116）
├── requirements.txt      # 精确依赖 pin（来自 deployment manifest）
└── README.md
```

安装精确记录的依赖后，即可对官方原始 `collect_data*.txt` 文件或其所在目录做预测：

```bash
python -m pip install -r requirements.txt
python predict.py path/to/collect_data1_2_3.txt --output prediction.json
python predict.py path/to/subject-folder --output prediction.json --include-timeline
```

支持 `--device cpu`。当前发布没有经过审计的 CUDA 适配器，强制 `--device gpu` 或
`--device cuda` 会被明确拒绝，而不是静默回退到 CPU。输出 JSON 遵循
`dist/schema/prediction.schema.json`。

## 可视化本地服务（serve.py）

`python serve.py --open` 启动本地推理桥（仅绑定 127.0.0.1，默认端口 4173）：同源
提供 `../visual/` 的静态前端与 `/api/*` 接口（`/api/health`、`/api/upload`、
`/api/analyze`、`/api/artifacts/...`）。浏览器中选择 collect_data*.txt 文件/文件夹后，
由本包的 canonical Predictor 完成推理并返回预测契约与运动遥测；前端不实现任何模型逻辑。
`dist/start.bat` 即为该服务的一键启动器。

## 官方自动评分接口

2026-10-15官方比赛接口由相邻的`../submission/EatingDetection.exe`实现：无参数读取
EXE同级ACC_test_data，输出results.csv三列externalid,beforeTime,afterTime。此处predict.py
与前端仍使用JSON契约，不能将JSON或历史四列CSV当作官方结果。比赛EXE无需Python或联网，
构建与提交说明见`../submission/README.md`；本包用于前端开发/可视化，模型版本相同。
