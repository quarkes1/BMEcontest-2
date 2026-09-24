# Representation v2 状态

当前终态：`P2_NO_SUPPORTED_BRANCH`。P0/P1 已完成；P2 目标数据 Q0/Q1 的 5 个 outer-train wrapper ×4 个 inner 分区已全部完成。Q1 scratch 的五组宏平均 candidate AP 为 **0.26439**，低于 Q0 frozen-random 的 **0.28847**；仅 1/5 wrapper 正向，log loss 也更差。按预注册 gate **停止 P3**。这些数字不是 event F1；没有运行 outer、没有新正式 F1、没有 dist/release 变更。完整结果见 [`p2_probe/report.md`](p2_probe/report.md) 与 `p2_probe/aggregate_metrics.json`。

FD-I 外部数据适配器已就绪，但源/目标 ACC/GYRO 物理单位与换算仍未证实。Q2/Q3 因此标记 `FD_DATA_REVIEW_REQUIRED`，不是性能失败；未使用旧 FD checkpoint，也未启动 source 预训练。若要再尝试迁移，需先补足单位证据并重新审阅研究计划。

P0 锁定了旧正式发布、保护快照、来源白名单、20 个 T/V 分区及资源上限。P1 实现 packet timestamp 展开、session gap/mask、5 秒 25 Hz 六通道 token、64-bin 有序小模型与训练/安全 checkpoint 测试；20 位 fold-0 训练受试者的首 5 秒输入抽检中，17 位通过固定 80% 覆盖率，三位按既定无效输入规则处理。目标缓存的 ACC/GYRO 是 raw counts；FD-I 样本 ACC 为物理量级，陀螺仪转换未知。

P2 训练前成本 dry-run 记载 20 个 T/V 分区、共享的 macro/micro 候选生产器 200 次正式基础拟合，以及 Q0/Q1 40 次目标分支拟合、80,000 optimizer steps；两次候选 pilot 另用了 20 次基础拟合。首分区完整候选/token pilot 耗时 103.5 秒，含 772 T /223 V candidates 与 148,183 T /41,608 V token。40 步合成 CUDA 吞吐校准约 0.047–0.049 秒/步，峰值显存 0.24–0.57 GiB。正式 Q0/Q1 两次命令合计墙钟约 3,527 秒、CUDA 训练累计 1,561 秒；40 份 checkpoint 约 5.7 MB，未生成持久 token cache。

P2 收口发现 checkpoint provenance 的 NumPy 字符串影响 `weights_only=True`。已对 40 份本轮工件作受限的仅元数据修复，验证权重逐项不变并留存前后哈希；未重训、未重算分数。旧训练代码 SHA 与修复代码 SHA 分别记录于 `p2_probe/report.md`。当前没有进入 P4/P5 的权限或条件。
