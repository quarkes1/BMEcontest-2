# S9 — 固定训练人口的适配优化审计

结论：`S9_TRAIN_ONLY_AUDIT_COMPLETE`。发现固定train目标确实恶化，target梯度可达encoder，
不存在“完全没有反传”的故障；未证明source辅助项是原因。无新事件F1、无health/outer调用、无发布变化。
预注册代码/规范commit `dc92cd0`，输入锚定S8 sealed commit `8206cc6`。

固定S8 fold0_policy0的11名train，重新生成OOF候选：272个运动有效候选（41 positive），
另72个无效候选不进入辅助运动readout审计。初始scaler/LR完整重放通过。
8个既有候选producer OOF fit，1次initial LR重放fit，0 event verifier fit、0 encoder更新。
没有生成validation ledger或加载其他受试者的raw token。

| 同一固定train集合 | subject-weighted BCE ↓ | subject-weighted AP ↑ |
|---|---:|---:|
| initial encoder + initial head | .221839 | .764366 |
| initial encoder + final head | .226770 | .750027 |
| final encoder + initial head | .282551 | .762793 |
| final encoder + final head | .261428 | .748730 |

最终组合BCE增加 .039589，仅3/11人train BCE改善。head对encoder漂移做了部分补偿，
但不足以恢复初始目标。encoder相对L2权重漂移仅 .005942，标准化latent绝对漂移
median/.9/.99/max为 .443803/.990200/1.684131/3.621005，说明小权重变化也可明显移动固定readout的输入。
这是训练优化/任务耦合的证据，不是模型泛化上限，也不证明单一来源的因果关系。

四个固定诊断batch，初始及最终encoder各有3个target/source梯度余弦为负、1个为正；
加权source/target梯度范数比初始 .0287–.0991，最终 .0422–.6653。
**四个target batch恰好全部为negative**，因此不能推广到positive梯度或完整训练目标，
更不能据这四个batch直接认定应去掉source监督。所有target梯度有限且非零，权重未更新。

## 下一步自主研究决策

先做一个有界train-only优化对照：保留source loss、架构、初始readout和步数，分别检验
取消global gradient clipping与扩大target batch是否修复固定train BCE。两者是独立单因素臂，
不组合搜索、不看health选参数。优先级和进入事件验证的train gate在下一份规范中固定。
若两者都失败，关闭本优化修复路线，转向新的多尺度/细粒度时序表示，而非继续扩大调参网格。
小batch+非线性裁剪可能改变优化行为只是待检验假设，本审计没有证明它。

## 保护、成本与验证

66.34秒，CUDA/CPU4线程；输出约10KiB、0 token缓存、0 encoder步骤。
源原始文件身份核对，S8输入前后hash一致，保护发布快照前后相同，outer_calls=0。
139项表示研究测试通过（4条既有imputer warning）；本阶段全仓回归待后续代码冻结后统一执行，
不把S8的694通过当作本阶段的新全仓证明。Luna/medium代码审查无实质阻塞。
