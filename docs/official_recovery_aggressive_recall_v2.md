# Amendment R2 — 召回优先独立试测（2026-10-22）

用户最新明确允许更激进自主推进，时间目标为尽快送第二份程序。原正式promotion规则不改；新增**外部试测资格**，不再要求每个探索EXE先证明开发F1显著提高。不得称正式升级、确认改善或向官方保证收益。

## 根据与固定两档

当前官方49TP/27FP/63FN、112GT，F1=.521277。对于这同一评价集，增加TP=x、FP=y时F1改善的代数条件为y<(112+27)/49*x≈2.8367x。此前新增TP>新增FP工程门远比F1本身保守。该计算不是对隐藏数据恢复能力的证据，不能把本地x/y移植为官方收益。

固定两档beta=1.25（温和）与1.5（激进），只改变训练侧OOF的最终阈值选择目标Fbeta=(1+beta²)TP/[(1+beta²)TP+beta²FN+FP]。tie按F1、PPV、更高threshold。两档都登记，不在health搜索更多beta/阈值。保留现有模型、特征、原分数及原A admission（NMS/cap/threshold均用原p）；不使用RELATIVE-SCORE公式，不重训模型。

## 研究与试测资格

复用relative_score_probe.py的显式mode recall125/recall150，与mode relative分离；policy与score分别seal/commit。T24三组160GT，先记录原A89/62/71配对结果，只有召回资格通过才完整五折39人254GT，原A138/80/116。每档的五折父级必须同ID/mode已通过并复算gate，来源/角色图/旧分数融合恒等/资格来源/66保护检查不变。

试测资格：T24净TP≥+3、完整五折≥+5；净FP≤3×净TP+2；开发F1不低于A−.02；至少2/3或3/5折TP不减。**不要求F1提高**。结果若仅召回换FP，必须明示。任何正式F1提高/模型promotion仍须独立证据，不适用此资格。两档固定并列，不因health表现追加beta，也不将较好者描述为无选择偏差胜者。

## 最快交付路径

通过者不再重训相同模型：复用首个官方.521277版四个精确冻结模型及其63/47/116D schema、micro threshold、原完整deployment admission参数。每档full-target最终threshold为对应五折**train-only policy thresholds的median**；不在outer score上回扫该聚合阈值，不声称聚合后EXE已有nested F1。源bundle SHA必须与首轮已验收training summary一致、模型文件byte SHA逐项一致。该路径是**决策政策试测**，不是新算法权重。

注意原A每折admission可能与full deployment不同。导出前必须使用full deployment固定admission，在每折train OOF上重新按同beta选最终阈值并seal；然后才可对对应outer作配对诊断，原A和固定deployment-policy参考必须分别报告。median仅来自这五个训练侧阈值，不能混用原A各折admission的阈值。新相同admission诊断如灾难性下降（ΔF1<−.02或TP不增）则不导出该档；不得以另一admission的好成绩包装部署。

新bundle版本official_recall125_policy_v2 / official_recall150_policy_v2；新EXE分别在dist/official_trials/recall125_policy_v2 / recall150_policy_v2，不覆盖第一EXE或submission。四模型内容不变、新policy/metadata单独版本，使用官方无参数ACC_test_data→results.csv接口。程序接收官方受试者文件夹内全部文件，无新传感器/PPG假设。

打包只复用trial_exe.build_command通用PyInstaller叶函数，不放宽首轮严格builder的目标/身份检查；新增独立安全wrapper检查新两档目的路径、官方API hash、源bundle/模型SHA、保护66、源代码冻结、单文件MZ、自检、neutral cwd无开发PATH。EXE-only中文目录、真实同人双文件Python/EXE CSV逐字节一致、空/坏输入保留既有输出。最终冻结代码全套回归后交付，缺失官方原示例明确跳过。构建临时树只清本次验证过绝对路径的scratch，既有删除策略拒绝的目录不绕过。

## 实施顺序与预算

1. TDD扩展共享frozen-score runner，relative原行为不变；测试固定Fbeta、raw准入、两档身份/gate、父级拒绝混用。新实现Luna medium一次只读审查。
2. 三档relative/recall125/recall150各自policy→sealcommit→score；relative仍原严格F1门。recall过试测门直接五折，避免新模型训练。
3. 对通过召回档进行full-deployment-admission训练侧阈值及outer诊断，先seal再评分；源bundle读取与新独立copy/builder在研究门之后单独TDD与review。
4. 冻结最新代码全套回归与EXE-only验收并行，打包结束前不得声称完成。README清楚区分官方已知.521277/本地开发诊断/新EXE未知官方结果。不能发送邮件或替用户正式提交。

每无训练phase≤10min/RSS12GiB/证据16MiB/余量10GiB；两档最多两次独立编译，每次30min上限；不增加模型或特征缓存。过门失败关闭对应beta，不追加隐藏搜索。

Ruling：最新授权扩大的是试测风险容忍度，不是数据泄漏或结果措辞——允许送明确标记的召回探索程序，但正式发布不动；代价是新EXE官方F1仍可能下降，因此提供精确版本/hash与本地FP成本，允许官方对照，不伪称已提高。
