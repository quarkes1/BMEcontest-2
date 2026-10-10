# S7 — 冻结表示的小容量时间顺序对照

结论：`S7_NO_SUPPORTED_ORDER_READOUT`。本配置没有可推广收益，不运行事件扩展或outer，
不改dist/release。正式F1仍为0.6514285714；本轮没有产生新事件F1。

## 执行与可比性

预注册/代码commit `97023fe13cc7eb47db39a83321e302024773ac08`，规范
`docs/challenge_070_amendment_rv2_a3_order.md`。三个16train/8health划分仅来自原inner T24。
30个候选生产器、9个513参数head、3个133D LR；18000个head优化步。
源encoder完全冻结，实际状态逐次核对；每个head与normalizer只fit当前train。
源checkpoint身份固定，实际producer祖先图经过heldout检查。794个health候选（662有效）
的顺序、身份、标签与有效性均精确匹配S1。L重算协议与S1一致，macro指标亦一致。

R=随机冻结encoder；F=FD冻结encoder；FS=F但有效token向量打乱，时间/缺失/coverage不动；
L=同F编码的原133D线性读出。各臂固定相同head初始化/sampling seed，无搜索或early stopping。
这里的candidate AP/BCE不是事件F1，且health为反复使用的开发人口，不是独立测试集。

## 结果

| 候选指标 | R | F（有序） | FS（打乱） | L（原线性） | prior |
|---|---:|---:|---:|---:|---:|
| 三折macro AP ↑ | .352890 | .463177 | .484051 | .540503 | .122538 |
| 三折macro raw BCE ↓ | .457648 | .370153 | .364736 | .269564 | .373467 |

| fold | F AP | FS AP | L AP | F raw BCE | L raw BCE |
|---:|---:|---:|---:|---:|---:|
| 0 | .561109 | .491188 | .438294 | .272083 | .271747 |
| 1 | .428760 | .486061 | .594704 | .336239 | .260990 |
| 2 | .399662 | .474903 | .588513 | .502138 | .275954 |

F相对L的macro AP差为−.077326，BCE差为+.100589；相对FS的AP差为−.020874。
F仅1/3折胜过L、1/3折胜过FS。23名AP可定义的受试者中，F对L为9改善/13下降/1平，
AP差中位数−.006667；24人BCE为8改善/16下降。不是少数fold的总体均值偶然掩盖一致改善。

固定2000步全部执行，但预注册末400步下降检查标记R三折、FS fold1、F fold2为
convergence unresolved，优化gate不通过。即使忽略这一项，AP和BCE gate也明确失败。
不延长训练或依据这三折搜索宽度/学习率/seed。

## 含义与下一方向

FD仍比随机encoder有候选信息，但本次非线性时序读出没有稳定超越已有线性池化，
打乱顺序反而在两折更好。训练loss降低而未见受试者评分恶化，提示过拟合/域偏移风险；
不是“顺序必然没用”，也不证明更复杂模型一定无效。64bin浅层局部卷积也不是全尺度建模。
首折的好结果不能作为模型推广依据。

关闭本配置，不继续在这批health上手调frozen-head。接下来值得独立注册的路线是
源域手势监督与目标域事件监督联合训练/受控fine-tuning，让encoder适配事件任务，
而不是继续用不同池化器包装同一冻结encoder。更大TCN/Transformer不受参数量硬限制，
但必须有训练分区内的正则/预算/消融与真实事件TP/FP检验；复杂度本身不是进步。
下一轮主收益指标应回到严格nested的inner事件F1，candidate AP仅作为辅助诊断。
这是假设与执行方向，不是本轮已经验证的结果，不宣称能够达到0.70。

## 成本、保护与验证

CUDA执行，CPU线程4；总墙钟370.69秒，编码+head计时67.19秒（同步墙钟，非精确GPU kernel时间）。
峰值CUDA分配38,749,184 bytes，summary前证据695,381 bytes；零持久token缓存、零encoder副本。
保护快照前后完全相同，outer_calls=0。Luna/medium独立复核无实质阻塞。
129项表示研究相关回归通过（4条已有全缺失列imputer warning）；全仓回归另行记录最终结果。
预注册规格hash以plan所锚定commit版本为准，规范文件后续台账追加不是重新注册。

用户发布纪律：正式可比事件F1较当前发布版提高超过.03时提请人工审批，批准后才更新dist的
可执行推理程序、模型与接口文档，然后继续迭代。当前首次触发线>0.6814285714，
candidate AP/inner F1/GT-aware诊断均不替代该成绩。
