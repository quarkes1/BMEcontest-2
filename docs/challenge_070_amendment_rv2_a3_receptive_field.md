# A3-S13 — 完整5秒动作感受野：源域先行设计

状态：`DESIGN_REGISTERED_NOT_IMPLEMENTED`。S12完整失败后登记的新问题；未写代码、
未训练、未读取新目标health。用户自主探索授权覆盖设计与后续有界研究；不请求重复
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
