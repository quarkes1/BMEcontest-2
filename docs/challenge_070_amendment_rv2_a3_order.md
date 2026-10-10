# A3-S7 — 冻结表示的小容量时间顺序对照

ID `RV2-A3-S7-001`，用户已授权自主探索0.70路线。目标是检查时间组织的增量，
不是再次调融合权重。当前正式outer F1仍为0.6514285714。

## 假设与边界

S1证明FD冻结表示有目标域candidate信息；S2/S4/S5没有稳定事件收益。
S5使用无序全局池化，其失败不能排除候选内部顺序的价值。选择小容量读出，
而非直接扩大Transformer或重新微调encoder，避免重演R0过拟合。

原inner T24内同一GroupKFold(3)，每折16 train/8 health；原V7和outer完全排除。
health仅为反复使用的开发证据。每折独立重建10个候选生产器并审计祖先。
标签沿用历史incumbent IoU>=.25，目的为与S1可比；竞赛文本严格>的差异不在本轮改动。
NO OUTER / NO POLICY SEARCH / NO DIST / NO RELEASE / NO PROMOTION。

## 冻结输入与三个配对臂

仅加载S0固定checkpoint，SHA256
`c6276837b1366117075e7e8a7504e084926cbfc5d80b98b863e442255995efc4`。
R=初始随机encoder；F=FD最终encoder；FS=同F但候选内确定性打乱有效token向量。
三者encoder全程冻结。源head不用；无target SSL/encoder训练。

沿用A1真实时间5s/25Hz token、200ms gap、candidate-local滤波；gravity_align=False，
逐token ACC/GYRO独立demean/RMS。FS仅打乱有效16D向量，seed来自candidate key SHA256，
不改变时间、缺失位置、sample mask或coverage；训练/评分都使用同一定义。
候选级全局mean/std/max不变，四段统计会变化。无效候选回退train有效候选subject均衡先验。

每encoder的16D mean/std只fit当前train：subject等权→candidate等权→有效token等权，
scale floor=1e-6。F与FS共享同一个normalizer；health不参与任何拟合。
随后按真实时间聚合64 bins，每bin mean16+max16+四项几何/质量metadata=36D；保留gap，
不裁切长candidate。空token候选直接全零并mask=False。

## 固定读出和训练预算

Conv1d(36,8,1)→ReLU/mask→Conv1d(8,8,3,padding1)→ReLU/mask→
masked mean8+max8→Linear(16,1)，513个可训练参数；不加残差、BN或dropout。
相同fold seed、相同初始化及sampling stream；subject-first→candidate，不按class重采样。
仅有效train候选参与训练，bias初始化为subject均衡先验logit。
AdamW lr=.001、weight_decay=.01、clip1、batch64、固定2000步，无early stopping/搜索。
9个head共18000步、30个基础模型、3个S1协议LR参照；无额外encoder fit。
每100步记录loss，最后400步若下降>10%则convergence unresolved，gate不得通过。
保存最终head JSON weights，不复制encoder，不持久缓存tokens。

L参照：同一F编码的原始16D，四段mean/std+5D的133D，固定S1 subject-weighted
StandardScaler/LR(C=.1)。重新计算而不声称逐位回放；health候选身份/标签/valid必须与已封存S1一致。

## 候选gate（全部满足；不是事件F1）

三折all-candidate等权macro AP：F>=L+.02，F>R，F>=FS+.01；
至少2/3折F>L且至少2/3折F>FS；macro raw BCE F<=L+.02且<=prior+.02。
provenance、finite、encoder frozen、优化完成/收敛、资源检查必须全部通过。
通过仅支持另行注册严格nested事件实验，绝不自动outer。
失败封存 `S7_NO_SUPPORTED_ORDER_READOUT`，不在本批health上搜索通道数/步数/seed。

## 资源、文件与执行台账

上限3wallh/1GPUh（编码+head训练保守墙钟），输出<=16MiB；空闲磁盘>=20GiB，
GPU初始空闲>=1280MiB且峰值<=初始空闲80%。CPU线程4，CUDA可用则使用。
入口 `python -m scripts.research.representation_v2.order_probe`；输出
`outputs/challenger_70/representation_v2/amendment_a3_order/`。
保存spec/code/source SHA、祖先图、分人指标、logits、train轨迹、模型/normalizer JSON、预算。
失败保留小型state以说明原因，清理仅本轮无用临时文件。不影响历史证据。

本文件同时作为实施台账，遵循用户整洁要求，不另建重复计划目录。

用户补充：最终目标是实际事件F1提升；模型复杂度没有全局硬上限，前提是泛化改善且不过拟合。
513参数仅是本轮固定消融设计，不限制后续TCN/Transformer规模，也不把candidate AP当最终收益。

- [x] Task 1：新增order_probe与回归测试；缺失模块RED→7项GREEN，补充冻结编码回归。测试mask/空输入、
  shuffle保持边际、train-only moments、冻结、可表达顺序、CPU/CUDA容差和gate。
- [x] Task 2：接入现有crossfit/source/protection；审查隔离、预算和输出身份；代码/规范先commit。
- [x] Task 3：运行三折，记录正负结果，更新README/registry/报告，相关回归后commit。

Pre-flight：TokenBag→编码→normalizer→bins→head接口必须保留真实时间及缺失；
F原始embedding另供L，不错误使用标准化后的embedding冒充S1。
Ruling：自主研究授权下直接执行；规格/台账合一，资源优先使用相关回归，完整套件可异步运行。
Review focus：health进入normalizer；shuffle改变coverage；空bag触发coverage越界；
只核对声明图而非实际producer图；GPU异步使耗时失真。

Luna/medium独立只读复核：无实质阻塞。既有实际producer图与手工head图结合使用；
输出上限在fold写完后复核（检测超额而非操作系统硬配额），小型JSON预计远低于16MiB。

执行完成：`S7_NO_SUPPORTED_ORDER_READOUT`。F/FS/L的macro AP为.463177/.484051/.540503，
F仅1/3折胜过L与FS；F raw BCE .370153，对照L .269564。收敛标记亦未通过，
不延长训练。370.69秒墙钟、67.19秒编码/head计时、约679KiB证据，无缓存/encoder副本。
129项相关测试通过，保护快照不变；结果见输出report.md。没有新事件F1，不更新dist。
下一方向改为受控目标域适配与事件级收益检验，不限制后续模型复杂度。
最新用户纪律：正式事件F1相对当前发布版提高>.03先请审批，通过后更新dist及接口，再迭代。

Final verification：全仓688 passed、5 skipped、15条既有warnings，1067.94秒。
Luna/medium二次只读核对结果与文档一致，无实质错误。清理本轮两份可重建pyc，
无临时脚本/持久token缓存。研究分支保留，不合并、不发布；实验固定版本以97023fe为准。
