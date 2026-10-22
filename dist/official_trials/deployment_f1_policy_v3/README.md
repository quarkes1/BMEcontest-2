# 进食检测独立试测版：F1-policy

版本：`official_deployment_f1_policy_v3`，2026-10-22。优先送测本版；不替换正式提交包。

## 运行接口

只需发送本目录的 `EatingDetection.exe`。Windows x64 自包含单文件，内含模型及依赖，无须安装 Python、CUDA 或另行提供模型；使用 CPU 推理。

保持官方输入结构，将数据放在 EXE 旁：

```text
EatingDetection.exe
ACC_test_data/
  userinfobean.txt
  HNU00001/
    collect_data1_....txt
    collect_data2_....txt
  HNU00002/
    collect_data1_....txt
```

`userinfobean.txt` 和传感器文本沿用官方接口，不手工转换时间轴。直接双击 EXE，或在命令行运行 `EatingDetection.exe`，不需要参数。程序始终在 EXE 所在目录输出 `results.csv`，与启动时的工作目录无关。CSV 列固定为 `externalid,beforeTime,afterTime`，时间为原始13位 Unix 毫秒时间戳，仅输出进食事件；无事件时只有表头。成功退出码为0，输入错误为2；不要同时运行多个进程写同一个结果文件。

身份检查：`EatingDetection.exe --self-check`，输出版本及内置模型检查结果。首次启动需要解压运行依赖到系统临时目录，请留出空间。

## 变化与效果边界

四个模型和特征定义与上一份一致性重训试测版完全相同，仅将最终事件阈值从 .6595272837359293 改为 .4582664763345077。新值来自五个训练侧 OOF 阈值的中位数；准入 .2、NMS .3、每主体候选上限6及融合权重 .25 不变。

39主体/254餐的冻结折模型政策开发诊断：旧政策123TP/36FP/131FN、F1=.595642；本政策133TP/59FP/121FN、F1=.596413。召回增加但FP也增加，F1基本持平。它不是最终EXE的无偏CV成绩，不能与历史eligible餐F1=.65143直接比较。最新用户转述官方成绩 .521277 属于上一轮反馈；本版官方F1未知，不能保证提高。

真实双文件加空文件的源码/EXE CSV逐字节一致、中文路径下仅EXE运行、其他工作目录启动、自检、空输入及错误输入保护均已通过。官方原示例已不在本机，未声称测试过。全仓冻结回归1148通过、7跳过、39警告，无失败；结束后版本、SHA及保护文件再次核对通过。跳过与警告不等于全部环境均已验证，详见仓库 `docs/official_recovery_policy_delivery_v3.md`。

## 文件核对与复现

- 大小：62,893,974字节。
- SHA-256：`fa6987b6244fe73c76ee741ea352fcc74e8e4de06de332812e99b1f1d963aa44`。
- PowerShell核对：`Get-FileHash .\EatingDetection.exe -Algorithm SHA256`。

完整仓库中的来源：`models/event_stack/official_deployment_f1_policy_v3/deployment/`；诊断、导出、构建、验收分别在 `outputs/official_recovery/deployment_f1_policy_v3_*`。使用冻结环境运行 `python -m scripts.research.official_recovery.policy_trial_export --mode deploy_f1`、`policy_trial_exe --mode deploy_f1`、`policy_trial_acceptance --mode deploy_f1` 可重现各阶段；构建环境为 PyInstaller6.22.3/hooks2026.8。命令拒绝覆盖已有交付目录，不能直接在当前已有产物上重复执行。用户送检只需EXE，不必发送源码、模型或此文档。
