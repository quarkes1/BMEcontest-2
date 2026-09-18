# EatingSense Visual（可视化应用）

腕部 IMU 进食事件检测的实验台界面：证据时间轴、原始 IMU/姿态查看器、进食事件列表、
发布指标与三维动作回放。前端**只消费** canonical 预测契约与运动遥测，不实现任何模型
逻辑；原始 TXT 推理始终由本地 Python 服务完成。

## 概述

- 技术栈：React + TypeScript + Vite + Three.js；Canvas 时间轴；三视图（3D/Side/Top）
  刚体动作回放（前臂/腕/手/手表一体）。
- 两种运行模式：**静态/演示模式**（直接打开构建产物即可，无后端）与**完整分析模式**
  （本地推理服务在线时，可选择 TXT 文件/文件夹做真实 canonical 推理）。
- 物理诚实：真实数据的姿态回放需要显式标定（单位、坐标映射）；缺标定时显示
  "Orientation unavailable"，绝不伪造动作或绝对轨迹。

## 目录结构

```text
dist/visual/
├── index.html                 # 构建产物：自包含单文件（生成，勿手改；双击即用）
├── runtime/
│   └── release-metadata.json  # 生成：发布指标（来自 incumbent registry + crossfit summary）
├── README.md
├── IMPLEMENTATION_NOTES.md    # 审计与实现说明（含物理/坐标限制）
├── app/                       # 前端开发目录（Vite 根）
│   ├── index.html             # Vite 源入口
│   ├── package.json / package-lock.json / tsconfig.json / vite.config.ts
│   ├── tools/e2e-smoke.mjs    # file:// 端到端冒烟（playwright-core + 系统 Edge/Chrome）
│   └── src/
│       ├── main.tsx  App.tsx               # 入口与顶层状态（单一播放时钟）
│       ├── pages/                          # MonitorPage / EventsPage / ModelPage
│       ├── components/                     # TopNav / StatusHeader / SessionBrowser / DataLoader
│       │   ├── timeline/                   # Timeline.tsx + timelineMath.ts
│       │   ├── inspector/                  # SelectedInterval / RawImuChart / EventContext
│       │   └── motion/                     # MotionReplay.tsx + ForearmModel.ts
│       ├── motion/                         # quaternion / coordinateFrame / orientation / interpolation
│       ├── data/                           # types / prediction / telemetry / loader / format / demo
│       ├── runtime/                        # inferenceClient / capabilities
│       │                                   # + release-data.generated.ts（构建时生成，勿手改）
│       ├── styles/app.css
│       └── tests/                          # contract / timeline / orientation / loader / realDataRobustness
└── tools/
    ├── prepare-release.mjs                 # 生成 runtime/ JSON 与应用内联 TS（唯一来源）
    ├── make-standalone.mjs                 # 构建后内联为自包含单文件 index.html
    ├── clean-build.mjs                     # 构建前清理旧产物
    └── export-motion.mjs                   # 独立 CLI：TXT → motion.json + motion.bin
```

## 快速开始

```bash
cd dist/visual/app
npm ci            # 或 npm install
npm run dev       # 开发预览：http://127.0.0.1:5173/
npm test          # vitest（16 项）
npm run build     # 生产构建 → dist/visual/index.html（自包含单文件）
npm run e2e       # file:// 端到端冒烟（需系统 Edge 或 Chrome）
npm run e2e:bridge # 真实 TXT 全链路（需 Python 环境，自动拉起本地推理服务）
```

**双击 `dist/visual/index.html` 即可运行**：生产构建是自包含单文件（IIFE 内联脚本 +
内联样式，零外部资源、零 fetch、零后端），演示模式与"打开已有预测"在任意机器上
直接可用；发布指标在建时由 `tools/prepare-release.mjs` 内联，无需网络。

## 竞赛启动器

`dist/start.bat`（Windows 双击）：

1. 定位发行目录（基于脚本相对路径，无硬编码路径）；
2. 启动本地推理服务 `dist/inference/serve.py`（仅绑定 127.0.0.1，默认端口 4173）；
3. 同源提供 `dist/visual/` 静态页面与 `/api/*` 接口；
4. 服务就绪后自动打开浏览器。

若 Python 依赖缺失，启动器**首次运行时会自动安装**（`pip install -r inference\requirements.txt`，
无需任何交互）；Python 缺失、版本过旧、安装失败或端口被占用都会给出明确提示并暂停窗口。

## 开发

`cd dist/visual/app && npm run dev`。开发服务器仅用于前端预览；`predev`/`prebuild`
钩子会先刷新 `runtime/release-metadata.json`。

## 生产构建

`npm run build` = `prepare-release`（生成发布指标 JSON 与内联 TS 模块）+ `clean-build`
+ `tsc --noEmit` + `vite build`（IIFE 输出）+ `tools/make-standalone.mjs`（把 JS/CSS
内联为单文件、脚本置于 `</body>` 前保证在 DOM 就绪后执行）。Vite 根为 `app/`，
输出回写到发行根 `../index.html`；构建产物与前端源码分离、互不覆盖。
运行已构建页面不需要 Node、不需要 Python。

## 演示模式

打开页面即进入确定性演示数据（界面明确标注 **DEMO DATA**，模型 run key 为
`DEMO — not a release`）。演示不依赖本地推理服务，可离线展示全部交互。

## 分析 TXT 文件（需要本地推理服务）

1. 启动 `dist/start.bat`（或 `python dist/inference/serve.py --open`）；
2. **Select TXT files** 选择一个或多个 `collect_data*.txt`；
3. 前端把原始文件发送到本地推理服务，由 canonical Predictor 完成推理，返回预测契约
   与运动遥测（浏览器不做任何特征/阈值/解码计算）；
4. 状态栏依次显示 `Preparing files… → Running inference… → Loading timeline… → Ready`。

文件名须符合仓库既有约定 `collect_data*.txt`；无关文件会被忽略并在状态栏报告。

## 分析文件夹

**Select folder** 使用浏览器目录选择器（`webkitdirectory`）。相对目录结构会被保留，
用于会话识别；当浏览器不支持目录选择时，继续使用 **Select TXT files** 作为回退。

## 打开已有预测（高级）

**Open existing prediction** 加载已产出的 `prediction.json`（可同时选择配套的
`motion.json` + `motion.bin`）。该路径适用于调试、复现与离线分析；运动遥测的
`session_id` 必须与预测中的会话匹配。

## 预测契约

稳定契约：`dist/schema/prediction.schema.json`（v1.0）；合成安全示例：
`dist/examples/example_prediction.json`。界面消费 `events` / `candidates` / `gaps` /
`timeline`（含可选逐点 `series`）与 `diagnostics`。前端**不得重实现**阈值判定、
候选准入、事件融合、解码或任何特征提取；预测 JSON 里没有给出的决策，就不是前端该
计算的东西。

## 运动遥测

可视化遥测与算法预测分离：`motion.json`（v1 manifest）+ `motion.bin`
（`f64_ms_6xf32_le`：float64 毫秒时间戳 + 6 个 float32 原始 IMU 通道）。
本地推理服务从上传的 TXT 通过 canonical 会话解析器导出，与 `tools/export-motion.mjs`
的独立 CLI 格式一致；高采样数据保持二进制，时间轴仅在绘制时降采样。遥测以原始单位
（通常 `raw_adc`）声明，未标定前不宣称物理单位。

## Motion Replay

- 回放的是 **IMU 推导的姿态**（互补滤波，标定可用时），不是动作捕捉；
- 前臂/腕/手/手表为一个刚体；不推断独立腕关节角、肘部关节或绝对三维位置；
- 不对加速度做双重积分伪造轨迹；数据缺口不插值（分段显示）；
- 三视图 3D / Side / Top，OrbitControls 仅 3D 模式；播放 0.25×/0.5×/1×/2×。

## 传感器标定限制

本地桥接服务仅在找到至少 3 段连续 10 秒、加速度模长变异系数低于 5% 的静止数据时，
用模长中位数估计 `acceleration_counts_per_g`。真实数据的 Motion Replay 此时标记为
“Approximate”：仅按重力估计倾斜，不积分未标定的陀螺仪。轴向矩阵
`[1,0,0, 0,0,1, 0,-1,0]` 是与 demo 一致的查看器约定值，**不是设备规格**；
重力也无法确定航向。静止数据不足时不提供标定，显示 Orientation unavailable。
原始 ACC/GYRO 仍按声明单位查看。

## 浏览器兼容性

Chrome/Edge 等 Chromium 浏览器支持目录选择与完整交互；Firefox/Safari 的目录选择
支持不一，使用文件多选回退即可完成同样分析。

## 测试

```bash
cd dist/visual/app
npm ci
npm test              # contract / timeline / orientation / loader / realDataRobustness 五组（20 项）
npm run e2e           # file:// 演示模式冒烟（系统 Edge/Chrome）
npm run e2e:bridge    # 完整分析模式：真实 TXT → canonical 推理 → UI（需 Python 环境）
```

Python 侧桥接测试：`python -m pytest tests/integration/test_local_server.py`
（真实会话的 TXT → 桥 → canonical 预测契约一致性、遥测往返、路径遍历拒绝）。

## 清洁环境验证

```bash
cd dist/visual/app
npm ci && npm test && npm run build && npm run e2e   # e2e 直接以 file:// 打开构建产物
# 双击 dist/visual/index.html 验证零依赖演示模式
dist\start.bat   # 验证完整分析模式：选择单个 TXT / 多个 TXT / 文件夹
```
