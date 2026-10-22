# 进食检测独立试测版：高召回对照

版本：`official_deployment_f15_policy_v3`，2026-10-22。**本版是风险对照，本地F1下降，不称改进版。优先发送相邻目录 deployment_f1_policy_v3 中的主试测EXE。** 不替换正式提交包。

## 运行接口

只需发送本目录 `EatingDetection.exe`。Windows x64 单文件自包含，内含模型及依赖，无须安装Python或CUDA，使用CPU。

```text
EatingDetection.exe
ACC_test_data/
  userinfobean.txt
  HNU00001/
    collect_data1_....txt
    collect_data2_....txt
```

输入沿用官方文本格式，可包含多主体、多文件。无参数运行 EXE，在其所在目录生成 `results.csv`，不受启动时工作目录影响。输出列固定 `externalid,beforeTime,afterTime`，起止时间为原始13位 Unix 毫秒整数，仅写进食事件；无事件仅表头。成功退出0，输入错误退出2。不要让多个进程同时写同一结果文件。

`EatingDetection.exe --self-check` 可检查版本及内置模型。运行时会在系统临时目录解包依赖，并非所有库传统静态链接；请留出空间。

## 风险与证据

四个模型及特征定义与上一轮一致性试测版逐字节相同。最终阈值 .26383368823631487 来自五个训练侧 OOF 选择值中位数，目标为F1.5；未在外层回扫。准入阈值 .2、NMS .3、每主体候选上限6、融合权重 .25 不变。

39主体/254餐冻结折模型政策诊断：旧政策123TP/36FP/131FN、F1=.595642；本政策139TP/79FP/115FN、F1=.588983，TP+16/FP+43、F1下降 .006659。五折召回均不减，但仅两折F1提高。更低阈值可能召回更多餐次，也可能因误报变多降低官方F1。该表不是最终EXE的无偏CV成绩，不能与历史eligible餐F1=.65143直接比较。上一轮用户转述官方F1=.521277；本版官方效果未知。

真实双文件加空文件的源码/EXE CSV逐字节一致、中文路径仅EXE运行、异地工作目录、自检、空输入及错误输入保护均通过。官方原示例已不在本机，未测试。全仓冻结回归1148通过、7跳过、39警告，无失败；结束后版本、SHA及保护文件再次核对通过。跳过与警告不等于全部环境均已验证，详见仓库 `docs/official_recovery_policy_delivery_v3.md`。

## 核对与复现

- 大小：62,894,337字节。
- SHA-256：`7f0d29ce914b1be4472b7cd12f7a6b2fac9d389605934ada2269a1f365fd03db`。
- PowerShell：`Get-FileHash .\EatingDetection.exe -Algorithm SHA256`。

完整仓库模型在 `models/event_stack/official_deployment_f15_policy_v3/deployment/`，证据在 `outputs/official_recovery/deployment_f15_policy_v3_*`。冻结环境中的复现阶段命令为：

```text
python -m scripts.research.official_recovery.policy_trial_export --mode deploy_f15
python -m scripts.research.official_recovery.policy_trial_exe --mode deploy_f15
python -m scripts.research.official_recovery.policy_trial_acceptance --mode deploy_f15
```

PyInstaller6.22.3/hooks2026.8；命令拒绝覆盖已有版本，不能直接在当前已有产物上重复执行。送检只需EXE，不必附源码、模型或此文档。两份EXE若一起发送应保持可辨认的版本标识，不能相互覆盖。
