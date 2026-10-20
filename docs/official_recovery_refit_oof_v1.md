# REFIT-OOF-V1：最终复核训练特征的单因素对照

2026-10-20。代码检查已在REFIT-SAMPLING之前发现最终verifier用训练内窗口分数、CV用OOF分数。本项只隔离这一差别，不调参。

基准A固定REFIT-SAMPLING-V1的sampled臂：完整160餐上47TP/13FP/113FN、F1=.4272727273。B保留相同原CV窗口训练样本、标签、特征、模型、seed、完整候选池；只将verifier训练候选/116D特征改为当前16train内部四折窗口OOF生成。两臂verifier都用旧eligible事件标签、LR/LGBM各.5，seed+3/+23，策略及micro阈值逐折沿用同train分区原固定值。

B窗口crossfit的seed设为原seed+2，从而其最终macro refit seed等于A的seed+2，micro为seed+12。每折health候选几何及两路支持必须精确回放A；任何差异中止，不以新health候选混入归因。B训练候选可以改变，因为它们来自OOF，这是预期干预。

仅T24三个16train/8health。新窗口crossfit、imputer及verifier仅fit当前train，保留祖先图，V7及outer均排除。所有策略在开始前已冻结，不访问health标签选参数；元数据可能来自共享文件，立即按身份裁剪。两臂统一完整GT、严格IoU>0.25，缺餐保留分母，A计数必须逐项复现。

门仍为B-A pooled F1>=.01且至少2/3折提高；报告TP/FP/FN和逐人收益。不称独立验证，不与正式.651429或官方.466667直接比较；通过只支持部署训练一致性修复的更广验证，不直接更新dist。

预计30次窗口fit和6次verifier fit，无参数搜索、无神经网络/原始IMU读取，CPU4线程、<=30分钟、输出<=64MiB、磁盘剩余>=10GiB。源账本、输入缓存、session eligibility数组前后指纹，正式模型/release/EXE保护不变。失败INCOMPLETE，不保存新模型权重或特征缓存。
