# A3-S13 — 完整5秒动作感受野：源域先行设计

状态：`IMPLEMENTED_READY_FOR_SINGLE_RUN`。用户要求继续下一阶段；未训练、
未读取新目标health。用户自主探索授权覆盖设计与后续有界研究；不请求重复
批准。正式outer和dist审批门保持不变。本文件先锁定研究设计，实施前补具体TDD计划。

## 为什么换这个问题

S11优化修复未胜冻结表示，S12概率形态+2TP却+3FP，也未胜冻结表示；
因此关闭这些固定配置，不以换融合权重/形态阈值继续追逐相同T24 health。
这不证明“现有表示完全无用”，也不证明“更大模型必然有效”。

三个可选方向：

- **扩大源encoder的完整动作感受野**：保留同样unit-free输入与16D输出，先检查
  源域学习，之后才决定是否支付昂贵目标nested验证。当前选此方向。
- 外部foundation encoder：数据规模更大，但输入物理单位与许可门尚未过，详见
  `challenge_070_external_pretrained_review.md`；不把未知ADC当已校准数据。
- 直接扩大事件TCN/Transformer：长程能力可能有用，但目标人数少，且此前小型
  顺序头未证明收益；先增强局部动作表征，暂不同时改变local/event两层。

Ruling：选源域先行，不是限制复杂度，而是隔离“更完整动作建模能否学到更多”
这一假设。误判代价是错过其他架构，所以负结果只关闭该固定配方，不否定复杂模型。

## 固定S13源域实验合同

ID `RV2-A3-S13-001`。只读取已有FD-I右腕数据，不读目标raw、目标GT、policy或health。
复用S0源数据fingerprint、seed20260926的27/7人划分、每人每class最多256个token、
中心标签/negative排除、10Hz过滤/25Hz重采样与unit-free处理。逐token起点/标签与
人口计数必须复核S0封存validation/population；不得为新模型补抽更容易的样本。
重用source validation属于开发证据，不称独立source测试。

只改变encoder架构：

```text
6×125输入 + sample_mask
  → Conv1d(6,64,kernel5,padding2) + ReLU，mask
  → 5个现有ResidualBlock(64,dilation=1/2/4/8/16)，每block后mask
  → 有效sample masked mean
  → Linear(64,16)
  → 独立Linear(16,1)源分类头
```

encoder应为126544参数（含bias），源head17参数。理论局部感受野
`5 + 4*(1+2+4+8+16) = 129`采样点，约5.16秒；边缘有padding，不能声称每个
位置都看到了完整真实5秒。原encoder为11656参数、33点约1.32秒。
输出仍16D，不同时增加目标verifier维数。宽度和深度一起改变，因此结论只能归于
该更强架构，不能单独归因于RF或参数量。

固定5000 steps、batch128、AdamW lr1e-3/wd1e-4、clip1、同样subject→available
class→token抽样、ACC/GYRO同一SO(3)旋转。无early stopping、无checkpoint挑选，
只保存初始encoder与final encoder/head；每100步保存训练/梯度/时间。
需记录模型初始化造成的随机数消耗差异，不能声称两架构随机实现逐位相同。

源域gate（全部满足）：

1. 所有finite、主体隔离、实际5000steps、资源和保护检查通过；
2. source-val AP至少比冻结S0的`.8440475046760654`高`.01`；
3. source-val raw BCE≤S0`.25704393608246073 + .02`；
4. 仍满足S0的AP≥阳性率+.10及不劣train sampler prior BCE要求。

这些门槛在新模型训练前固定。逐人AP/BCE、median/IQR/p10和方向必须报告，但不是
暗中新增gate。通过只说明值得注册目标迁移探针；失败关闭此架构/预算，不挑seed、
训练步数或source-val子群重跑。成功亦不能直接跑outer或改变dist。

## 工程与验证设计

新增研究模块 `scripts/research/representation_v2/wide_source.py`；新增
`tests/research/test_representation_v2_wide_source.py`。不修改现有S0模型、loader或
已封存运行路径，不让旧source合同接受新架构。复用unitfree的纯预处理/采样函数，
新artifact使用独立schema及architecture字段，CPU安全state-dict加载、严格shape与SHA。

输出 `outputs/challenger_70/representation_v2/amendment_a3_wide_source/`：
plan/population/training_history/validation/subject_metrics/summary/state/report及一个
final checkpoint（内含初始随机对照）。source文件只引用，不复制；token bank只在内存。
不得覆盖已有attempt；异常记incomplete，不据残缺轨迹宣布失败。

实施TDD必须覆盖：126544参数/RF计算、有限16D输出、mask外污染不影响结果、
真实梯度与optimizer更新、相同旋转应用两个triad、source-val扰动不改变train采样、
gate四项边界/缺失拒绝、checkpoint安全roundtrip及错误schema/shape/SHA拒绝、
tiny真实训练+隔离+原子保存；必须实际看到RED→GREEN，并跑相关与全仓测试。
预训练前固定代码/计划commit，结束后报告、README、registry与结果commit。

单实验上限2GPUh/4wallh，CPU4、CUDA优先；输出≤16MiB、磁盘余量≥20GiB、
源文件解码估算≤可用RAM80%、CUDA peak≤启动空闲80%。预算是停止条件，不是
用户硬件/模型复杂度的全局上限。报告实际步数、参数、显存、吞吐、wall time，
不能只报GPU训练时间而隐藏解码/预处理开销。

## 后续边界

S13通过后，再单独注册同架构随机/预训练对照的目标probe，target归一化/readout/
verifier均fold-contained。是否沿用固定四段16D聚合，由该目标规范在读取新目标结果
之前锁定；当前文件不授权同时试多种pooling或融合。源域提升不是目标事件提升。
S13失败则重新评估物理单位可解性与多尺度raw路线，不重新激活S12阈值搜索。
正式目标仍F1≥.70，正式可比提升>.03先交人工审批，批准后才更新dist。

## 实施计划与台账（同文件收口）

以本文件为binding spec；沿用既有research分支，不新建数据/模型副本。
计划/台账合并到本文件，避免额外过程md与临时workspace。原有source loader只读复用，
S13新schema独立，不影响任何旧运行。各任务接口如下：

- [x] Task1：TDD实现`WideTokenEncoder`、`source_gate(metrics,baseline_bce,checks)`；
  测试126544参数、真实长程梯度、mask污染、CPU/CUDA与完整gate边界。
  先运行`pytest tests/research/test_representation_v2_wide_source.py -q -p no:cacheprovider`
  观察模块/符号缺失RED，再实现到GREEN。
- [x] Task2：`train_bank(bank,train,validation,device,steps=5000,batch_size=128,observe=None,
  budget=None)`返回payload、validation logits、训练轨迹/抽样计数；
  `save_artifact(path,payload)`与`load_artifact(path,expected_sha)`严格安全roundtrip。
  tiny真实训练测试source-val扰动不影响最终权重、val排除、原子保存/schema/shape/SHA拒绝。
- [x] Task3：实现`run(root)`，只在clean git与S12完整失败后进入；
  将S0 plan/population/validation通过git show绑定29a625e，复核source fingerprint、
  split/population与validation的participant/start/label；复算S0冻结logits验证输入合同。
  所有最终输出完成后校验保护/资源，再给出完成状态。独立Luna/medium只读审查。
- [ ] Task4：相关研究测试、diff检查、代码/规范commit；随后单次运行S13与全仓回归。
  实验CLI无可调训练参数，不按中途source-val选择终点。
- [ ] Task5：复算source gate/逐人指标、更新README/registry/report，清理本轮pyc/tmp；
  结果commit并核对clean tree。通过只登记下一目标probe，失败关闭固定架构/预算。

Pre-flight：Task1模型16D与Task2 head16→1一致；Task2训练函数允许tiny测试预算，
但Task3固定5000/128且artifact记录实际步数；generic artifact loader只做结构验证，
未来目标用途loader必须额外要求5000与完整passed结果/plan身份，不能用tiny artifact准入；
Task3原子checkpoint与Task5 summary SHA一一绑定。无跨任务隐式width变化。
实现裁决：逐人单类人口的AP/AUC按现有metric定义为null；聚合统计仅用有限值，
记录有效人数，不把未定义AUC补成0或引入gate。目标raw访问始终为false。
身份检查同时绑定unitfree/fd_source预处理代码到29a625e；只读取S12封存state作为
上阶段关闭条件，不读取任何S12逐行health/candidate数据来训练或选择S13。

Task1/2/3证据：模块不存在RED→14项GREEN；reference/run缺失RED→17项GREEN；
指标范围和预处理身份缺失RED→21项GREEN。原有表示研究179 passed（增加4项前），
下一轮含最终测试重新验证。独立Luna/medium代码审查无Critical/Important；
预算末步越界原评Minor，按严格停止语义提升为Important并修复：真实训练+受控时钟
测试先失败，训练结束后追加budget检查，最终resource也显式验证training_seconds≤7200。
只读审查未运行测试/未来target准入不在本阶段，分别由主执行者回归与未来目标规范覆盖。
最终22项新增测试通过，相关表示研究184 passed、20条既有/预期缺失列警告，12.99s。
本次未修改任何旧源模型、预处理函数或生产文件。代码提交后才单次训练；全仓回归
以本次新代码另跑，不冒用S12的721项证明。
