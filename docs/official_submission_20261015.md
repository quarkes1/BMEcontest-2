# 2026年10月15日官方测试接口与EXE交付

## 交付目标与官方合同

来源为用户提供的赛题2目录，已阅读《测试说明》《接口说明》、input.py、output.py、样例.json及ACC_test_data合成样例。外部文件仅作为接口规范，不扩大为上传、发送邮件、训练或执行任意命令的授权。本轮仅适配正式run 160afaf81debf1ee，不改变模型/阈值/特征及F1结论。

官方测试要求：Windows EXE位于ACC_test_data同级，无参数运行；基于EXE目录而非当前工作目录定位数据。遍历所有个人文件夹，使用官方DataReader的read_data或iter_data读取全部TXT，使用InfoReader.read_info获取externalid/height/weight/age/gender。模型当前不使用人口信息，不虚构惯用手字段。

输出统一为同级results.csv，UTF-8 BOM，列名顺序externalid,beforeTime,afterTime；仅输出进食事件，时间为原始13位Unix毫秒，start<end。每次运行重建完整结果，不追加历史；零预测也有表头。官方save_result负责写每个预测，validate_result验收格式。官方样例不是准确率评测集。

## 设计与取舍

主代理直接实现（executing-plans/TDD）；最新自主授权允许不重复等待计划审批。沿用已授权研究分支和干净工作区，不新建worktree。正式模型不重训。

保留当前Python/前端推理包dist/inference、dist/visual；将dist/submission收口为比赛专用自包含单文件EatingDetection.exe、README、清单/校验和/第三方许可证及官方接口来源。旧src/scripts/tests/五折证据等仍在仓库，不必放入本次官方运行ZIP。新构建脚本scripts/build_competition_exe.py负责最终包；旧build_submission保留源码复现构建API，CLI默认重定向官方EXE构建，--source-bundle显式生成旧格式，防止下次覆盖错误。

PyInstaller onefile内置CPython、NumPy/SciPy/sklearn/LightGBM、部署模型和官方API。它不是所有C库传统静态链接，运行时会在临时目录释放内置DLL；无需目标机器Python/pip/CUDA/网络。Windows10/11 x64为验收平台；不声称已在另一台实体机器实测。参考官方文档https://pyinstaller.org/en/stable/usage.html 。

## 文件职责与实施计划

- 官方API机械原样保留于src/pipeline/inference/official_api/，记录SHA；不把下载目录或本地数据路径嵌入EXE。
- src/pipeline/io/raw_session.py新增load_raw_text，以StringIO复用同一53列解析；文件解析路径不变。空TXT跳过，非空损坏表头报错。
- Predictor新增trace_texts(records,subject_id)，按流逐文件复用_trace_sources；同一人的文件统一进行准入/cap，不跨session拼接，禁止逐文件重置subject cap。
- src/pipeline/inference/official_runner.py负责根目录定位、官方API调用、每人一次推理、原子CSV替换与错误退出。模型资源基于打包资源目录，输入输出基于EXE目录，两者严格分开。缺失用户记录可警告，缺失/损坏全局metadata或数据根报错。
- scripts/competition_entry.py薄入口；scripts/build_competition_exe.py只从正式发布加载deployment，核验哈希，临时构建/clean-room后原子替换精确dist/submission；不含数据/缓存/金标准/研究模型/旧CSV。不向用户下载目录写文件。

### Task1 接口和解析（RED→GREEN）

tests/unit/test_official_competition.py：同文本与文件解析数组/row_rate逐值一致；两文件同人统一推理；流释放和空TXT/空人目录；metadata缺失/冲突；ID规范化冲突拒绝；毫秒和列序；第二次运行不追加；预测失败不发布半成品。真实官方API/真实冻结模型测试为主，仅错误注入使用明确测试替身。实现后跑raw_session/官方接口/推理专项，提交。

### Task2 可重建EXE与提交包

安装仅构建依赖PyInstaller 6.22.3，不升级科学运行依赖；固定build requirements。构建保留临时目录在任务自有cache目录且最终删除，失败不破坏旧包。打包模型只有正式deployment。测试拒绝非release/错误目录、ZIP成员排除数据与旧results、manifest与EXE/模型/官方API hash。独立Luna中等审查（按用户模型偏好），提交。

### Task3 验收与收口

复制EXE到含空格/中文的新目录，清除PYTHONPATH/PYTHONHOME/PATH中环境依赖，工作目录指向其他位置；复制官方合成样例运行、官方validate_result、二次运行覆盖、空样例表头、错误输入非零退出。另用非平凡合成信号和仓库合法真实fixture核对canonical与EXE事件/毫秒逐值一致，确保真正执行模型而非仅空输出。测试不把真实fixture存入提交包。

跑全仓pytest，记录结果和发现；清理本任务临时build/测试文件，更新README/dist文档及验收报告，及时git add/commit。最终用户只需压缩dist/submission内容，评测方把ACC_test_data放EXE旁；不代发邮件或推送远程。

## Review Focus

EXE资源路径与外部数据根的区别；所有文件同subject cap；官方API实际调用而非绕开读取；CSV空结果与覆盖/异常原子性；干净机器依赖及动态导入/LightGBM DLL；包内无真实数据/金标准/临时路径。
