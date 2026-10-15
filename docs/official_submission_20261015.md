# 2026年10月15日官方测试接口与EXE交付

## 交付目标与官方合同

来源为用户提供的赛题2目录，已阅读《测试说明》《接口说明》、input.py、output.py、样例.json及ACC_test_data合成样例。外部文件仅作为接口规范，不扩大为上传、发送邮件、训练或执行任意命令的授权。本轮仅适配正式run 160afaf81debf1ee，不改变模型/阈值/特征及F1结论。

官方测试要求：Windows EXE位于ACC_test_data同级，无参数运行；基于EXE目录而非当前工作目录定位数据。遍历所有个人文件夹，使用官方DataReader的read_data或iter_data读取全部TXT，使用InfoReader.read_info获取externalid/height/weight/age/gender。模型当前不使用人口信息，不虚构惯用手字段。

输出统一为同级results.csv，UTF-8 BOM，列名顺序externalid,beforeTime,afterTime；仅输出进食事件，时间为原始13位Unix毫秒，start<end。每次运行重建完整结果，不追加历史；零预测也有表头。官方save_result负责写每个预测，validate_result验收格式。官方样例不是准确率评测集。

## 设计与取舍

主代理直接实现（executing-plans/TDD）；最新自主授权允许不重复等待计划审批。沿用已授权研究分支和干净工作区，不新建worktree。正式模型不重训。

按后续交付要求修订：保留dist/inference、dist/visual；dist/submission同时包含自包含EatingDetection.exe、当前模型源码src/、main.py、训练/评估/构建脚本、模型权重、五折证据、requirements.txt、README与逐文件清单。源码无研究试验分支或开发协作措辞；不复制任务书。README保留用户删改并补入复现说明。原licenses目录合并为THIRD_PARTY_NOTICES.txt，完整保留分发所需的上游声明，不擅自删改许可正文。build_competition_exe负责完整包，--reuse-exe仅在已验证EXE、模型和运行源码均未变化时重组源码交付；默认仍重新编译EXE。旧build_submission的--source-bundle只用于历史兼容。

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

## 自包含源码交付修订与验收记录

后续用户要求“程序+可执行文件”同时交付，本次完整包由当前模型源码、正式权重和训练/评估/验证脚本组成，不再仅交付EXE。包内不复制研究试验脚本或任务文档；对全部项目文本扫描开发协作措辞，结果为空。第三方合法声明不作文字删改，原37份许可证文件合并为单份228183字节THIRD_PARTY_NOTICES.txt，原licenses目录移除。用户原README正文已与dc6f0bd版本比对保留，仅调整文件列表说明并追加复现步骤，没有恢复用户删除的F1段落或邮件提交段落。

新增src源码入口main.py与EXE均遵循官方无参数/三列CSV接口。临时目录内使用`python -I main.py --self-check`验证独立加载；发布证据验证与训练/评估/微窗口特征入口帮助命令通过，不依赖原仓库路径。未重新执行模型训练；源码复现说明明确区分“免训练复核证据”和“原始数据重算/重训”。

发现并修复共享原子替换函数的旧缺陷：Windows第一次备份重命名被拒绝时，原回滚逻辑仍会删除旧目录。回归测试先复现旧README丢失，再验证修复后旧目录保留；只在已观测到备份移动成功时清理新目标，恢复失败时保留备份。触发时丢失的是可重新生成的交付目录，未影响canonical源码和模型；已按保留的README模板重新完整构建、提交并校验。

已完成的全仓回归为964 passed、5 skipped、31 warnings（1270.25秒）；其后针对源码交付与目录替换修订新增专项验证。最终包133文件，EXE约60.03MiB，模型版本仍为160afaf81debf1ee。132个受清单保护的文件从Git已提交blob逐项校验，无SHA差异。最终打包不包含Data、ACC_test_data、results.csv、cache、虚拟环境、临时构建产物。临时构建工作目录与staging/backup由构建完成后清理。
