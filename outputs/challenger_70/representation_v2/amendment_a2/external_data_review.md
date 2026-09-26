# FD 迁移单位复核（只读，2026-09-26）

结论：`FD_SOURCE_UNIT_REVIEW_REQUIRED` 仍成立，本轮没有运行FD迁移。

本地官方 `FDdatasets/doi-10.48804-CN8VBB/v1.0/00_ReadMe.txt` 明确Shimmer3、双腕、6DoF、64Hz，以及 `[ax,ay,az,gx,gy,gz]` 通道顺序；未提供可直接核验的ACC/GYRO物理单位、量程/LSB和完整坐标映射。左腕为hand-mirroring前数据的说明不能替代坐标约定。

目标侧 `src/pipeline/io/raw_session.py` 按原始列读取ACC/GYRO，并未给出counts/g或counts/(deg/s)换算。`scripts/research/representation_v2/fd_source.py` 已正确保留 `undocumented_raw`，不猜测source→target转换。

历史P1 evidence中的“FD ACC observed m/s²”来自样本量级观察，不是数据集/设备规格证据，应降格为未证实假设；旧证据文件保持原样，本文件追加解释边界。不能凭gyro/acc量级自行确定单位。

同组原本的Downloads本地仓库路径在当前机器上不存在，未从该路径取得新证据。本轮没有下载替代副本或对外联系数据作者。

最小下一步：取得采集设备/固件导出说明（型号、full-scale、比例系数、轴与佩戴方向），以及FD的Shimmer3配置/导出单位。已向用户非阻塞询问；目标数据实验不等待这份资料。

即使未来解决单位，仍需区分FD的intake gesture标签与本任务的meal episode标签，不能直接把两种阳性定义合并。
