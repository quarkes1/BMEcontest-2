# NEGATIVE-CLEAN-V1 — 第二个独立送测版实施计划

> 执行方式：superpowers:executing-plans，直接实现、TDD、最后整体只读审查。用户2026-10-21已批准路线并授权自主推进，不等待重复审批。

**Goal:** 在官方反馈F1=.521277的一致性重训合同上，找到可复核的第二项改善并尽快导出独立EXE；不保证隐藏集提升。
**Architecture:** 先只比较负监督去污染的两个单因素，再对通过者做一次同规模开发验证；未通过则收束并进入目标域全天弱监督定位，后者训练前另行锁定结构与预算。
**Tech Stack:** 原HGB/LGBM＋LR/LGBM、63/47/116维、已有缓存；数值线程8，不需要GPU。
**Spec:** 本文的“实验合同”；承接已批准聊天方案、official_recovery_consistent_refit_validation_v1.md。

## 新官方反馈与边界

用户报告49TP/27FP/63FN，F1=.521276596；前次42/26/70、.466666667，均112 GT，净TP+7/FP+1。按对话关联上一独立版，但未独立核对官方收到的EXE SHA/隐藏主体是否完全相同，不声称统计显著或唯一因果。
上一轮本地128FN只读诊断为83评分不足/43无IoU合格候选/2控制或竞争；不得当官方63FN的逐餐归因。

## 实验合同（训练前冻结）

1. T24原三个16train/8health及各自三折policy分区，用原T1 manifest，V7/outer不参与本轮拟合、选择或评分。历史分区已反复开发，非全新验证集。
2. 完整候选来自五份val的唯一联合，再按主体裁剪。旧eligible事件监督沿用规范loader；全量窗口标签保持原缓存，基础A为修复后的“一次抽样＋窗口subject-OOF meta”。每个producer仅在其当前训练主体内采样一次，seed20260901，session负窗<=max(3×正窗,1)，不拼fold_train。
3. W仅窗口负去污染：原label==0且按完整GT重叠已达到现有正窗规则（macro >50%、micro >=50%）的行改为-1；原1、原-1及其他0不改。先去污染、再采样。不要把灰区全扩为正例。候选评分范围不删行。
4. E仅meta负去污染：A的同一OOF候选、特征及micro阈值；原candidate label==0且与当前训练主体完整GT严格IoU>.25匹配的行不参与verifier fit。保留所有旧正例和其余负例，不增加新正例。E不得再去污染窗口；W不得同时改meta监督。三臂A/W/E，不加组合臂或调去污染强度。
5. 固定原工厂、C=.1、blend=.25、116D Context-v1。宏/微最终与OOF seed为base+2/+12，verifier为base+3/+23。micro阈值仍在当前train OOF按原网格选；A/E共用完全相同producer。W仅允许标签变动带来的后续变化。
6. 每个health训练T内，三折policy OOF的每个producer独立四折窗口OOF，再训练verifier；score主体不参与任何模型/imputer/去污染目标。各arm采用**相同完整GT策略目标**，原admission网格和事件阈值选择器；所有三个health的全部policy先冻结，再进行任何health refit。新A不是已送测EXE逐位复现，也不能沿用旧.54826等不同合同baseline数字。
7. health统一完整160餐，严格整数IoU>.25、一对一，缺候选仍在分母。报告TP/FP/FN、PPV/recall、三折/逐人变化、候选可达性、mask与采样计数/身份哈希、fit祖先图；只在三折完整结束后汇总。
8. gate：相对A pooled F1 >=+.02且至少2/3折严格提高；TP不得减少。若两臂均过，按F1、PPV、固定W先E后的顺序择一。阈值只在train OOF选择，不按health/官方反馈回调。本轮失败不微调mask规则或增加组合臂。
9. 预计240次窗口fit、72次meta fit；所有模型内存使用，不持久化中间权重或特征。<=2小时、证据<=128MiB、空间>=10GiB；顺序处理分区，A/E复用producer。输入缓存/CSV/代码与资格t_acc、imu_valid前后SHA；正式57项、上一独立版EXE及bundle全保护。

## 通过后的验证与交付

只对gate通过者，另行冻结一次39主体原五折paired验证方案；沿用同一一次采样/OOF/meta mask合同与完整254餐，每折policy只在outer-train三折OOF选。重复outer只称development，五折完整、F1>=+.02且>=3折提高、TP不降才支持新EXE；不为失败变体消费outer。
完整val联合按新合同一次抽样、subject-OOF训练meta、最终窗口refit；新模型和EXE新版本目录，原0.521277版和正式release不覆盖。官方无参数接口、EXE-only/中文路径/真实双文件Python一致/空错输入/全套回归/源SHA照旧验收。原官方例子缺失则明确缺口，不自动发送邮件或晋级正式材料。
若两个mask方向都失败，保存失败证据并转向已批准的全天弱监督定位；先写具体token、bag与负例合同，不把candidate Transformer旧实验改名重跑。

## Review Focus

- 去污染只影响train role，未删除score候选/GT；mask不制造正例，边界含等号差异可测试。
- 一个主体跨session的完整GT只传入其所在train；别名规范化/碰撞拒绝，不因缺训练行静默排除主体。
- 每层producer、imputer、meta和policy祖先都排除当前score人；A/E复用不等于跨health复用权重。
- 一次采样而非缓存训练折拼接，masked标签不进入任何fit；原数组不修改。
- 失败输出保留INCOMPLETE，不覆盖已有证据或模型；正式与第一独立EXE全量保护。

## Task 1 — masking与生产器适配

Files：新增`scripts/research/official_recovery/negative_clean.py`，新增`tests/research/test_official_recovery_negative_clean.py`。
Interfaces：`censor_window_batch(batch,truths,mapping,micro)`返回新batch及计数；`censor_candidate_labels(candidates,labels,truths,mapping)`返回新labels及计数；`prepare_train(universe,train,truths,arm)`返回仅train角色的一次采样数据。
- [ ] 测试先RED：跨文件GT、macro/micro50%边界、不改原1/-1和原数组、主体隔离、candidate严格.25边界、禁止未知ID与非法标签、完整候选保留。
- [ ] 最小实现复用corrected_labels/sample_batch/subject slice；GREEN后提交。

## Task 2 — 三臂嵌套执行与gate

Files：新增`scripts/research/official_recovery/negative_clean_probe.py`；Task1测试文件追加gate/空输出/身份异常覆盖。
Interfaces：消费Task1，复用build_candidate_ledgers与完整GT policy选择器；`execute(root,output)`输出全新证据目录。
- [ ] gate/缺折/GT分母不等/失败preflight测试RED→GREEN；新路径、git干净、保护及输入身份先封存。
- [ ] 先完整policy生成并提交seal，后health；真实三折结束后判断门；更新README/结果报告，提交。

## Task 3 — 条件推进

- [ ] 若通过：五折方案预注册→执行→通过才独立重训/EXE，结果未确定前不改正式包。
- [ ] 若失败：关闭去污染参数面、具体化全天弱监督定位任务并执行；不伪造第二个“有提升”的EXE。
- [ ] 一次Luna/medium整体只读审查、Important修复RED→GREEN、最终完整pytest、文档/commit/清理。删除若被平台拒绝，报告并不绕过。

## 执行账本

- 2026-10-21：本轮用户授权持续自主推进；在已授权main原地执行，本文合并spec/plan/ledger，不新增worktree或零散脚本目录。
- Pre-flight：mask→采样→producer→meta→policy→health共用精确主体集合；baseline/arm共用完整评价分母；A/E同producer而W分开；prototype无交付权限。
- Ruling: 第一轮只遮蔽达到正标签标准的冲突负窗，不遮蔽所有临近餐次灰窗——保持最小假设面；代价是可能留下部分弱污染，本轮不根据结果扩大规则。
- Ruling: 所有臂重新以完整GT选择train-only策略，固定blend=.25——与新官方目标一致；代价是A不再是旧T24数字，必须重跑配对基线。
- masking/边界/主体隔离及gate/preflight共9项新增测试已观察RED→GREEN；零候选主体仍保留GT和分区覆盖，不能以缺candidate为理由删人。实现未修改任何canonical算法文件。
