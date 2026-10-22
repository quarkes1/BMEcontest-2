# R3独立EXE交付实施与验收

2026-10-22，依用户更激进自主授权执行，正式release/submission不替换，不发送邮件。算法研究规格见official_recovery_deployment_policy_v3.md；此文件只登记其已通过试测门后的独立模型复制/构建/验收，避免改动该研究源SHA。

## 已有结果与两档身份

完整39人/254餐政策开发诊断：部署policy参考D123TP/36FP/131FN、F1=.595641646；F1档133/59/121、.596412556（+.000770910），F1.5档139/79/115、.588983051（−.006658596）。两档五折TP均不减，分别净TP+10/FP+23与TP+16/FP+43，符合外部试测门，不是正式算法promotion或确定官方提高。三组初筛较大收益没有在五折保持相同幅度，必须如实披露。

最终阈值仅五个train-only选择的median，不能在outer重扫：F1 .4582664763345077；F1.5 .26383368823631487；首EXE .6595272837359293。NMS.3、admission.2、cap6、blend.25、micro threshold.1与四模型字节不变。没有新训练。

- 主试测：official_deployment_f1_policy_v3，models/event_stack/该版本/deployment，dist/official_trials/deployment_f1_policy_v3/EatingDetection.exe。
- 高召回对照：official_deployment_f15_policy_v3，对应deployment_f15_policy_v3目录。它的本地F1下降，不能称改进版，作为用户新授权下的有界风险探索。
- 均为独立单文件Windows x64 EXE，官方无参数输入ACC_test_data/、输出results.csv。只有exe送检即可；版本、SHA、接口与风险写到各目录README，源代码/模型留仓库复现，用户不必发送它们。

## 源码与安全边界

policy_trial_export.py：重算两门/逐折原分数事件及计数、检查来源/封存/保护66和首轮模型报告SHA；新threshold只取train-only median。新模型目录严格白名单、不覆盖、拒绝symlink/reparse。直接复制8文件bundle，仅改policy/run_config/manifest的版本和来源元数据；4 joblib与feature_schema逐字节SHA相同。旧模型训练source_fingerprints保留，新政策依据另记metadata，actual_exe_cv=False、no_official_score明确。

policy_trial_exe.py：复用首trial build_command叶函数，不改首builder目的约束。检查export/模型/源码/保护跨阶段身份，官方API SHA、冻结PyInstaller6.22.3/hooks2026.8、CPU依赖；只编译两个精确新路径，单文件MZ、neutralcwd清环境selfcheck后原子不覆盖安装。

policy_trial_acceptance.py：仅拷贝新EXE到中文scratch，无仓库/Python/Node PATH，其他cwd运行。真实固定同人双原文件+empty，Python与EXE CSV逐字节一致且至少1个非空事件；missing/empty/malformed metadata必须正确并保护已有结果。build/export summary SHA、EXE与模型身份必须一致。原官方样例缺失明确未测试，不能冒称通过。

每个生成目录拒覆盖，异常留INCOMPLETE；新构建和验收scratch用独立mkdtemp namespace，只在绝对父路径/前缀/无link核对后删除。已有删除被策略拒绝的约1.71GiB旧scratch不绕过。日志/小型审计不视为可丢临时文件。

## 实施步骤及验收门

1. TDD复制模型/固定路径/拒覆盖/坏阈值/坏前置状态、已完成诊断重算、builder身份与不触及submission、missing EXE不建cache；相关回归。Luna medium一次只读review，严重项一次RED→GREEN修复，小项延期。
2. 提交全部源码后冻结；完整pytest（不与改src/scripts混跑）和两档export→commit→build→commit→accept并行但不共享scratch。旧/新保护66贯穿，不修改正式注册或frontend。
3. 全套、EXE-only验收通过后才交付；生成各目录README及总README/报告，保留旧实验信息并明确当前官方仍49/27/63、.521277。EXE SHA/字节数准确，推荐先送主试测，高召回仅对照。检查干净工作树并commit，不自动push/发送/正式提交。

预算：export≤10min、每编译≤30min、accept子进程≤300s，需10GiB空闲；不保留重复模型训练/增强特征缓存。完整pytest预计约22min，接受与构建并行可节约墙钟；完整测试时不再改代码或测试，若改了须重新冻结对应验证。

Ruling：独立试测代替正式升级——满足用户限时送检且保留回退；代价是官方分布未知，两档最终F1可能不升，高召回档尤其有更多FP风险。不存在“已达到.70”的声明。

## 实施与最终审查ledger

- 新export/builder/accept模块先缺失接口RED→GREEN；导出前置检查异常不记录状态的测试RED→GREEN，整个try内现在会记INCOMPLETE。29相关测试通过；源数据与已完成诊断真实只读复算通过，不代表当前整套/EXE已通过。
- Luna medium一次只读review指出Important：exists()==False的dangling link会跳过路径guard。Windows权限限制下对link-detection边界构造失败测试，确认先红；移除exists前置条件，逐component始终检查link/reparse。
- Final: minor (deferred)：bundle原子发布后，最后漂移检查/summary落盘失败可能留新模型目录并阻止同名重试。不自动删该目录防覆盖与丢证据；如发生，仅在确认该新版本INCOMPLETE且无送测使用后，人工核对该精确目录再选择保留或清理重试。模型目录存在绝不算交付成功，必须export COMPLETE、build COMPLETE、accept PASS及冻结全套均通过。
- Final: Ruling：review未判断真实EXE/官方效果及完整回归——分别执行后续验收，最终官方未知；代价是开发诊断不能保证正式竞赛收益。当前不宣称已完成。

- 修复后专项30项通过（16.44秒），包含两档真实五折证据复算和临时目录中的真实模型复制字节核查。d527b1c冻结源码后启动最新全套；全套完成前不修改源码或测试，不以此前1108项替代本次验证。

## 实际构建与独立验收

两档导出COMPLETE，4个模型及feature_schema SHA与首版完全一致；主档构建73.79秒，高召回档56.28秒。主档验收64.66秒，高召回64.28秒，均PASS：中文路径、仅EXE、移除开发PATH、其他cwd、自检、缺失/空/错误输入及真实双文件+空文件CSV byte parity，真实结果均1个事件；66保护文件不变。只证明接口/实现一致，不证明官方召回收益。

| 目录 dist/official_trials/ 下 | 字节数 | EXE SHA-256 |
|---|---:|---|
| deployment_f1_policy_v3 | 62893974 | fa6987b6244fe73c76ee741ea352fcc74e8e4de06de332812e99b1f1d963aa44 |
| deployment_f15_policy_v3 | 62894337 | 7f0d29ce914b1be4472b7cd12f7a6b2fac9d389605934ada2269a1f365fd03db |

高召回第一次export在主档构建刚落盘、产物尚未commit时被clean-tree检查拒绝，没有复制模型或评分。将仅含state.json的目录原样移动保存到deployment_f15_policy_v3_export_preflight1后提交，重新从干净树导出成功；未改源码/参数/研究证据。该失败记录保留，不隐去。

新build/accept自建scratch均自动清理，源码与测试继续冻结；全套尚待结束，暂不写“整套通过”。正式release、submission、前端及首试测EXE均未替换，无自动邮件或push。

### 五折差异（冻结结果转录，不作选参）

| fold | D TP/FP/FN | 主档 TP/FP/FN | 主档ΔF1 | 高召回 TP/FP/FN | 高召回ΔF1 |
|---|---|---|---:|---|---:|
| 0 | 20/5/27 | 21/10/26 | −.017094 | 21/14/26 | −.043360 |
| 1 | 30/6/21 | 33/8/18 | +.027736 | 33/9/18 | +.020022 |
| 2 | 16/8/36 | 18/14/34 | +.007519 | 22/23/30 | +.032556 |
| 3 | 28/11/22 | 30/16/20 | −.004213 | 30/18/20 | −.016968 |
| 4 | 29/6/25 | 31/11/23 | −.005852 | 33/15/21 | −.004627 |

逐折表解释了为什么“所有折TP增加”仍不能称稳定F1提升：部分主体组增加的误报抵消召回收益。不能据此重新选择部署阈值、去掉差折或改变两档身份；官方反馈也只能归属于实际收到的EXE版本，送测时应保留SHA和发送日期。

## 最终验证与交付结论

冻结d527b1c源码的完整pytest以exit0结束：1148 passed、7 skipped、39 warnings、1164.84秒，无失败。验证期间及结束后`git diff --quiet d527b1c -- src scripts tests`通过；期间只提交产物/证据/文档。日志SHA与命令见outputs/official_recovery/policy_v3_frozen_regression.json，日志保留本机。skip的已知环境限制包括Windows符号链接权限与缺失官方原样例，不能称所有场景已测试；警告包括旧测试的Windows子进程stderr UTF-8解码异常、退化合成特征与缺失列插补等，不称无警告。新两档验收使用errors=replace捕获日志并独立核对CSV原字节，均PASS，不靠测试日志文本代替结果校验。

结束后两档validated_export再次通过，全部66保护文件不变，两EXE SHA与构建/验收一致。正式submission、release、inference及visual相对863ed3a无更改。交付门完成：建议优先送F1-policy主档，高召回档仅作可选风险对照；新官方F1未知，不作正式promotion。

清理：两次新build与accept的mkdtemp已自动删除。完整pytest只余383个文件、136841字节（约134KiB）；核对精确路径后删除被执行策略拒绝，未重试、未换工具绕过，留在cache/delivery_build/policy-v3-full-pytest-20261022，不进入dist。此前约1.71GiB旧scratch仍保留，亦未再尝试。没有不必要的训练权重/特征缓存，本轮两份模型复制与EXE为交付所需。

收尾按原授权在main提交，工作目录保留；不新建分支、PR、自动push或邮件，不要求用户重复审批已经授权的执行流程。科学结论仍是“召回/FP风险试测”，而非“.70突破”。
