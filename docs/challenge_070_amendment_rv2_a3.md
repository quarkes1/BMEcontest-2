# RV2 Amendment A3 — 无量纲外部手势预训练探针

2026-09-26，用户授权自主寻找0.70路径。A2既定三折完成后才运行本独立假设；不修改A2参数、不合并成绩、不消费outer、不发布。

## 要回答的问题

候选条件下的目标SSL若不优于随机表示，下一步直接利用FD-I的手势标签，检验外部监督是否提供不同的运动信息。未知物理单位仍未解决；本轮不实施任何raw→物理换算，而使用明确定义的无量纲输入。原FD raw-unit gate仍然存在，新实验不能记为原Q2/Q3通过。

每5s、每ACC/GYRO三轴组分别在有效时间点去均值，然后除以去均值三轴向量的RMS。对非退化数据，若X'=aX+b（每triad共同a>0，固定3轴b），则Z'=Z。任一triad RMS=0、非有限或有效采样不足80%时token无效。浮点误差、各轴不同增益、串扰、饱和、量化、左右腕镜像和不同安装轴不在严格保证内；绝对幅度被主动丢弃。

目标预处理使用A1时序/间隙/filter合同，但 `gravity_align=False`；归一化在token内完成、不拟合全体target统计。source同样不gravity-align。增强为同一SO(3)旋转同时作用ACC/GYRO，不称为已校准或严格旋转不变的encoder。

## S0：先验证source标签可学习

ID `RV2-A3-S0-001`。只读FD-I右腕X_R/Y_R；已有官方文件本地加载，不下载、不读取target信号。34人按已有FDIAdapter的seed20260926拆27 train/7 validation，validation不参与encoder训练。

- 5s原始窗口stride5s、64Hz；标签沿用adapter：中心±0.5s有eating为positive；negative的窗口及前后2.5s不得有eating/drinking；其余跳过。
- 每participant每class最多256个窗口，固定seed均匀无放回抽取，再做过滤/归一化。记录每class/participant有效数，不为补齐失败类别再抽样。
- 先对每5s原始窗口做4阶10Hz低通，再 `resample_poly(25,64,padtype='line')` 到125点，避免零填充给静态信号制造边缘运动。
  为避免polyphase的微小DC增益波纹在RMS归一化后被放大，滤波/重采样作用于窗口去均值信号，随后恢复原均值；严格常量输入必须保持常量。最终仍执行上述token内去均值/RMS。
- encoder沿用11,656参数的16D局部TCN；head=Linear(16,1)，最终迁移只使用encoder。对照初始encoder权重随checkpoint保存。
- 固定seed20260926，5000 optimizer steps、batch128、AdamW lr1e-3/wd1e-4、clip1；subject→available class→token均匀抽样，无early stopping；每100步保存loss/梯度/step/elapsed。
- 每个batch给token一个随机SO(3)旋转，ACC/GYRO同R；无噪声、幅度、mask等其他增强；输入为token内无量纲值。
- 仅终点计算source-validation raw BCE/AP/AUC及先验基线，绝不报target事件F1。
- 只有finite/隔离/预算合格，validation AP>=validation阳性率+.10，raw BCE不差于以训练采样人口阳性率预测的常数基线，才允许后续target三折探针；失败则此source配置终止，不追选step。

“训练采样人口阳性率”按subject→available class→token的期望计算（每subject均有两类时为0.5），不是未重采样token的自然阳性率。额外报告cap之前全部eligible训练token阳性率、cap/过滤后bank阳性率及对应基线；它们都只用train，且都在抽样后的validation人口上评价，不冒充自然人口测试。gate固定使用sampler基线。source head只是学习诊断，不声称其概率适用于自然生活先验。每participant单独以(seed,participant_id)初始化抽样RNG，避免validation候选数量改变train的随机选择。

S0额外上限1 GPUh/2 wallh，内存预算加载单腕X/Y不超过可用RAM80%，磁盘余量>=20GiB。预计3GB读取与<100MiB token bank；bank仅内存。全过程单GPU训练。source checkpoint只保存final，旧FD checkpoint不复用。

## S1：仅S0通过才进行，需先固化具体实施清单

后续目标是与A2相同的原inner T三折，对照“无量纲随机冻结encoder”和“无量纲FD冻结encoder”，保持相同4段mean/std及5D元信息、固定C=.1线性读出。不能把A2的有幅度输入R直接作为新F的唯一对照。FD encoder source-only可跨target分区共享；所有target readout及candidate producer仍各fold独立。

S0只证明source学得到手势，并不授权跳过target probe、严格OOF互补性、完整inner event gate或outer人工授权。S1在S0报告后再登记具体ID、gate、成本和输出，不预先消费target数据。本轮自主授权允许继续准备，S0未过则不自动跑S1。

## 文件与验证

研究模块 `scripts/research/representation_v2/unitfree.py`；测试覆盖共同正增益/固定偏置不变、mask/常量输入、source标签边界、source参与者隔离、旋转正交且两个triad一致。输出 `outputs/challenger_70/representation_v2/amendment_a3_source/`，JSON显式LF，保存source文件指纹、注册SHA/code SHA、split、采样counts、轨迹、validation logits与最终模型。

训练前代码/测试/注册提交git；完成后报告实际开销、保留负结果、更新README/registry并清理临时文件。全部成本累计到原24 GPUh / 48 wallh / 12GiB cache / 2GiB checkpoint，不以新ID重置总预算。
