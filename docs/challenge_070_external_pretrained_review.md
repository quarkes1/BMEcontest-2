# 0.70+ 外部预训练表示核查

核查日期：2026-10-10。性质：下一路线的资料与输入合同审查，**不是实验结果或实施注册**。
不改变正在运行的S12，不下载数据/权重，不改变发布。正式F1仍为0.6514285714。

## 研究动机

现有FD表示有排序增量，但若干严格inner事件实验未通过完整门槛；S11尤其说明
降低训练BCE不等于超过冻结表示对照。因此值得考虑更大规模预训练及更长感受野，
但不能从HAR论文成绩推断本项目的事件F1，也不能把所有失败归因于模型太小。
S12先检验冻结手势概率的时序信息，结束前不调整其schema或选择策略。

## 候选比较与实际阻碍

| 候选 | 已核实的入口 | 与本项目的关系 | 暂不能直接接入的原因 |
|---|---|---|---|
| Oxford HARNET5 | 官方hub定义5秒、30Hz、3轴ACC，提供预训练权重路径 | 腕部来源、窗口长度接近，是有价值的冻结编码器候选 | 物理单位/重力处理尚未对齐；Academic Use Licence需核实交付条件；分类头不是预训练进食头 |
| LIMU-BERT-X | 官方README链接checkpoint；配置为6轴、20时间步 | ACC+GYRO兼容，但预训练主要来自手机而非手腕 | 采样率与时间跨度仍须确认；上游归一化依赖ACC物理尺度；不能任意把5秒压成20点 |
| SensorLM | 官方代码提供训练与特征管线 | 多模态可穿戴大规模学习值得参考 | 已查README未找到可直接用的预训练checkpoint；输入含工程化生理/活动特征，不是当前6轴raw接口 |
| Inertia-1 | 官方实现与论文，腕部ACC预训练路线 | 数据/模型规模和时间建模研究相关 | 已查README未定位现成checkpoint；有请求权重的公开issue，不能将开源训练代码等同于可用权重 |
| Oxford Sensori | 官方README链接模型仓库，支持分钟级HAR表示 | 更长上下文路线的参考 | 物理校准和整日预处理要求较强，Python版本也不同；同样存在Academic Use Licence核查事项 |

“未找到”仅指本次查阅范围，不是断言项目永远不提供。下载前必须重新查实际文件、
版本、许可与输入说明。模型名称相似也不代表checkpoint合同相同。

HARNET依据：[官方hub代码](https://raw.githubusercontent.com/OxWearables/ssl-wearables/main/hubconf.py)、
[官方项目](https://github.com/OxWearables/ssl-wearables)、
[许可原文](https://raw.githubusercontent.com/OxWearables/ssl-wearables/main/LICENSE.md)。
hub加载器排除classifier权重；不能直接将其分类输出称为进食概率。

LIMU依据：[官方说明及权重链接](https://github.com/WANDS-HKUST/LIMU-BERT_Experience)、
[配置](https://raw.githubusercontent.com/WANDS-HKUST/LIMU-BERT_Experience/main/config/limu_bert.json)、
[预处理实现](https://raw.githubusercontent.com/WANDS-HKUST/LIMU-BERT_Experience/main/utils.py)、
[MIT许可](https://raw.githubusercontent.com/WANDS-HKUST/LIMU-BERT_Experience/main/LICENSE)。
`Preprocess4Normalization`将ACC前三轴除以9.8，说明当前unit-free token不是自动兼容输入。
上游随机样本划分不能照搬为本项目的跨人评估。

SensorLM依据：[官方说明](https://github.com/Google-Health/consumer-health-research/tree/main/sensorlm)、
[输入特征定义](https://raw.githubusercontent.com/Google-Health/consumer-health-research/main/sensorlm/constants.py)。
Inertia依据：[官方项目](https://github.com/yang-ai-lab/Inertia-1)、
[论文](https://arxiv.org/abs/2607.06617)、
[权重请求issue](https://github.com/yang-ai-lab/Inertia-1/issues/1)。
Sensori依据：[官方说明](https://github.com/OxWearables/Sensori)、
[链接的模型仓库](https://huggingface.co/light156/Sensori)。

## 首要门：输入物理合同，而不是下载速度

本仓库原始IMU为raw ADC；`src/pipeline/inference/local_server.py`明确保留原值。
现有A3 unit-free分支绕过了未知物理尺度，并未证明原始计数能转换为g、m/s²或rad/s。
目前没有经过验证的设备量程/灵敏度换算系数，不能凭常见16384等常数猜测。

接入前至少核实：轴顺序与方向、ACC/GYRO单位、采样率、重力是否保留、滤波、
缺口/mask、佩戴位置和侧别。25→30Hz重采样只能解决几何问题，不能解决单位问题。
不得用全部受试者或health统计量反推归一化，再声称无泄漏。

如设备元数据无法取得，下一实验应明确叫“unit-free域适配”，用partition内的
无标签目标信号重新适配，并与相同输入处理的随机初始化架构比较；不能声称已按
官方物理合同使用预训练模型。源域和目标域也不能各自随意选一套归一化。

## 许可与安全门

竞赛允许外部数据不等于任意模型许可均可打包。Oxford学术许可不是MIT，涉及使用、
分发及修改条件；进入dist前需逐项核实本队用途和交付方式，必要时提交用户确认。
LIMU仓库MIT也需保存版权与许可，并确认具体checkpoint没有另附限制。

只取固定commit、明确checkpoint文件和SHA；优先安全的state-dict加载。
不直接执行远程`torch.hub`代码，不为载入未知pickle静默放开安全选项。
先审查文件结构/体积/许可，再下载单一所需文件；不克隆多份数据或保留无用缓存。

## 后续实验应如何避免重复低收益尝试

以下是设计建议，尚未授权为一个可以直接运行的实验配置：

1. 根据S12结果选择下一条路线。若形态仍无稳定增益，关闭24D schema，优先研究
   更完整动作感受野或多尺度表示，不继续在同一health上手调形态阈值。
2. 在物理/许可门通过的情况下，优先比较HARNET腕部ACC或LIMU六轴表示中的
   **一个固定模型**，而非按health选择多个外部模型。前者域更接近、后者许可更宽松；
   这些都不是事件性能证据，不能据此宣布谁更强。
3. 对照至少包括当前116D、相同架构冻结随机表示、冻结预训练表示。
   若需要降维，第一版固定在window层train-fold投影32D，再进行固定事件聚合；
   不把大量candidate级384D新变量直接塞给小样本verifier。
4. 所有目标归一化/PCA/适配/监督训练及选择均包含在现有subject-disjoint nested
   分区中。inner-val的raw信号不得参加生成自身表示的目标SSL；最终refit只能在
   inner policy冻结后进行。固定外部源checkpoint的祖先另行标识并核对主体重叠。
5. 先固定预算与完整事件gate，再运行；比较TP/FP、逐人分布、资源和随机对照，
   不因训练loss或AP改善就扩大outer。重复T24只能称开发健康检查，不是独立验证。
6. 模型通过更广inner验证后才申请下一次完整outer；正式可比提升>.03时提交审批，
   批准后才生成dist。许可证未过不能因F1高而绕过交付审核。

结论：外部大规模表示值得认真尝试，但当前最先需要解决的是可验证的输入合同，
而不是盲目增加参数。若该门暂时无法通过，本项目自己的unit-free多尺度encoder
仍是合法研究路线，只需按同样严格的分区和固定预算证明事件增益。
