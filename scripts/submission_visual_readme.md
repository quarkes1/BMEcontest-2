# EatingSense 可视化应用

在本包上一级双击`start.bat`，或运行`EatingDetection.exe --visual --open`。
不需要安装Python、Node、CUDA，不需要联网；浏览器访问`http://127.0.0.1:4173/`。

可选择原始TXT文件或文件夹，通过内置的正式模型生成进食事件、证据时间轴与原始IMU/三维姿态回放。
服务只监听本机127.0.0.1，浏览器不实现模型。关闭网页不会关闭服务，使用控制台Ctrl+C退出。
正常退出清理本次上传和分析的系统临时目录；强行终止或断电可能留下系统临时文件。
端口冲突可运行`start.bat --port 4180`；自动化运行可加`--no-open`。

直接双击`index.html`可离线查看合成演示或加载已有预测JSON；原始TXT推理需要通过start.bat启动本地服务。

## 与官方评分的区别

直接运行上一级`EatingDetection.exe`不启动网页，按官方接口读取同级ACC_test_data并写results.csv。
可视化模式使用JSON和运动遥测接口，不写入、清空或改变官方results.csv。
二者使用同一正式模型与决策策略，F1不因本次移植变化。

## 接口与源码

- GET /api/health、GET /api/capabilities：服务/正式模型信息。
- POST /api/upload：上传原始TXT，Content-Type为application/octet-stream，X-Session-Name为文件名。
- POST /api/analyze：JSON请求`{"tokens":["上传返回的token"]}`，响应包含prediction和sessions运动遥测。
- GET /api/artifacts/...：返回分析生成的motion.bin等资源。
- 页面与API同源，默认http://127.0.0.1:4173；服务端拒绝路径穿越。

前端源码位于app/src，构建工具位于tools。Python服务源码位于上一级src/pipeline/inference/。
需要调试Python时，安装上一级requirements.txt后运行`python serve.py --open`。
仅重建前端时需Node和npm，在visual/app执行`npm ci`、`npm test`、`npm run build`。
发布指标由本包release和outputs/crossfit证据生成，不手工填写。
端到端真实TXT测试需要另外提供合法原始数据，可通过BME_E2E_DATA_ROOT指定Data目录。
重建前端会改变文件哈希；正式提交应使用重新生成清单并校验后的完整包。
