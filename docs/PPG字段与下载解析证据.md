# PPG 字段与下载解析证据

核查日期：2026-10-13。范围：用户提供的下载附件、两个组内仓库和本项目已完成的T24抽样。仅只读核查与文档更新，未运行外部脚本、未训练、未改dist。结论：约25Hz包内波形获得组内实现的交叉支持；PPG3/PPG5有明确绿光映射声明，但尚未获得厂家字段表或逐值核验原始证据。

## 1. 下载附件能确认什么

来源：`C:\Users\19444\Downloads\1ae055b2ff8a4ffbaa7d7187476f01c3_附件\附件`。

核对了`数据下载脚本使用方法.txt`、`DataTable.txt`及`HuaweiResearch_DataDownlod - 外发.py`的下载相关代码。SDK wheel仅做静态源码关键字检索，未安装或执行；未发现PPG_TIME、PPG3、greenLight、sampleRate或samplingRate定义。SDK使用指导Word文档未在本次作完整内容审阅，不将其列为已核实的波形协议来源。

DataTable配置明确分为两层：

| 数据表 | 配置字段 | 本项目含义与边界 |
| --- | --- | --- |
| sensororiginaldata_system | externalid, sensorData, timeStamp.startTime, timeStamp.endTime | 受试者标识、附件路径、会话起止元数据；不是PPG包内定义 |
| mealinfo_puadqog7 | externalid, dietaryType, beforeTime, afterTime, drinkingVolume, satiety, foodName, tablewareType, dietaryHand, wearHand | 餐次标签及手侧、餐具等元数据；下载配置不提供单位/枚举完整解释 |

脚本`downloadAudio_Sensoe`（约109–134行）通过`BatchGetFileRequest`下载附件；ZIP解压代码处于注释状态，没有将PPG1..44解码或重排列的步骤。`get_attachments`负责取sensorData附件。脚本没有给出绿/红/红外映射、ADC单位、LED增益、采样率或包时间戳锚点。下载说明中的“连续脉搏波”是表用途说明，不是每列传感器协议。

安全：源脚本包含明文访问凭证。未使用凭证、未复制脚本到仓库；建议持有者在平台撤销并重新生成，文档不得记录其值。无需提供新凭证给本项目。

## 2. 组内 BME-CONTEST 的明确解析信息

用户链接重定向到`ghjs2007/BME-CONTEST`，本次固定快照`a07be2b35d1bd3c5627afe7038c55737d2158b8e`。

- [绿光验证代码](https://github.com/ghjs2007/BME-CONTEST/blob/a07be2b35d1bd3c5627afe7038c55737d2158b8e/code/validate_green_ppg_meal_response.py)：第19行设PPG_HZ=25；读取PPG_TIME、PPG3、PPG5及ACC。第45行稳定排序包时间戳，保留同包行顺序；每路独立滤波与Welch频谱估计。第215行输出映射`greenLight_1=-PPG3`、`greenLight_2=-PPG5`。实际计算读取正号原列；谱功率和以绝对DC计算的AC/DC对整体符号不敏感，但不能把该实现当作已验证的带符号波形重构。
- [README PPG部分](https://github.com/ghjs2007/BME-CONTEST/blob/a07be2b35d1bd3c5627afe7038c55737d2158b8e/README.md#ppg)：作者声明已逐值确认PPG3/PPG5为两路绿光。本次找到声明和用法，未获得其逐值对照文件或厂家原始映射表。因此记为“组内明确映射、待本项目独立核实”，不是官方事实。
- [旧生理特征代码](https://github.com/ghjs2007/BME-CONTEST/blob/a07be2b35d1bd3c5627afe7038c55737d2158b8e/code/build_candidate_ppg_physiology.py)明确说明同包PPG_TIME相同，行顺序承载约25Hz波形。但旧实现标准化多列后取中位形成共识波形；README已撤回混合光路、包时间和运动处理不足的旧HR/PRV结论，不能照搬。
- [15秒统计缓存](https://github.com/ghjs2007/BME-CONTEST/blob/a07be2b35d1bd3c5627afe7038c55737d2158b8e/code/preprocess_ppg.py)只保留每列count/sum/squared-sum。适合均值、方差和覆盖率，不保留样本顺序，不能据此恢复脉搏波或PRV。

可借鉴的是两路独立处理、覆盖率/包缺口/双路频率一致性、结合ACC低运动质量门控，以及缺失时不否决候选。其代码使用30秒bin、覆盖率>=.8、最大包间隔<=1.2秒、双路频率差<=10bpm等规则；这些只是该实验设置，不是本项目已验证阈值。稳定排序后连续滤波仍不能代替逐包缺口恢复，也不能自动保证生理可靠性。

作者报告低运动餐前/餐后HR变化及拒绝候选的条件区分信号，但没有据此发布PPG模型。我们没有重跑其数据，不能移植其指标为本项目成绩。餐后+45分钟证据仅适用于明确允许延迟的离线接口；餐前/餐后窗口必须以模型候选定位，不能在实际推理偷用GT餐起点。

## 3. NormanAC 的解析与限制

固定快照`b134aba6c8cae82abc43da66014122f99ab4c20c`。

[06_process_all.py](https://github.com/NormanAC/Eating-Detaction/blob/b134aba6c8cae82abc43da66014122f99ab4c20c/06_process_all.py)第50–73行：

1. 每种传感器以自己的TIME>0解复用；PPG列为PPG1..44。
2. 删除完全相同的行，按包时间排序，保留本会话非全零列。
3. 用不同包时间的中位间隔估计period_ms，用包行数众数估计batch_size。
4. 分配`t_ms = packet_time + row_index_in_packet * period_ms / batch_size`。

这为“包时间戳不是逐样本时间戳”提供另一份独立实现线索。但它默认包时间为首样本锚点；排序未显式要求stable，去重可能误删真实相同量化值，众数间隔也不能解决丢包和时钟倒退。因此不得直接照搬到脉搏时序恢复。

第185–188行PPG窗口特征仅每活跃列mean/std，未见波长映射或HR恢复。其PPG特征失败不等于正确恢复后的绿光生理信息无效。

[13_ppg_wrist.py](https://github.com/NormanAC/Eating-Detaction/blob/b134aba6c8cae82abc43da66014122f99ab4c20c/archive/early/13_ppg_wrist.py)只在已标注进食窗口上做同手/异手分类。它不能证明可以在全天未知餐次时路由；不能把餐次手侧元数据作为可部署输入。

## 4. 与本项目实测合并后的证据等级

| 问题 | 当前支持的结论 | 尚不能声称 |
| --- | --- | --- |
| 包与行 | T24固定24段抽样中完整包15个不同向量，包间隔中位602ms；两仓库均按包内序列理解 | 包内每行已得到官方时间标注 |
| 采样率 | 15/.602约24.92行/秒，与组内25Hz实现一致 | 2Hz波形或44列拼接成1kHz波形 |
| 有效列 | 本项目抽样前20列活跃、后24列全零 | 全数据永远只有20个通道 |
| 绿光 | BME-CONTEST明确使用PPG3/PPG5并声明逐值核验 | 厂家官方映射已获取，所有设备/固件通用 |
| 零值 | TIME=0需解复用，列可能未启用 | 所有零值都是调度而非缺失/饱和/无效 |
| 心率/PRV | 可开展约25Hz双列波形可行性验证 | 当前已可靠测得HRV，或PPG必然提高事件F1 |

本项目实测和分层数字见[初步审计](../outputs/challenger_70/representation_v2/hand_ppg_audit/report.md)。不修改历史预注册代码或合同；本文件是后续解释的补充。

## 5. 需要补充的资料与可继续的工作

优先向BME-CONTEST成员索取其“逐值确认”的原始依据：具名greenLight字段与PPG3/PPG5同一小段数据的对照、转换脚本或截图（去除凭证与个人信息）。若存在设备采集端导出代码，优先提供。

厂家/主办方字段表最好包含：PPG1..44的光路/波长与单位；具体手表型号和固件/采集应用版本；25Hz是否为配置采样率；15行包的行顺序；PPG_TIME标记首样本、末样本还是回调到达；零值、丢包、增益切换及无效值语义。

没有上述资料仍可在T24继续做带有明确假设的双列连续性、频谱一致性、运动污染和缺口审计；暂称PPG3/PPG5，不把它们升级为官方确认的生理量。先不做HRV、不拼接不同光路、不启动正式多模态训练或重开V7。精确波形时间恢复和后续训练方案需根据核实结果固定。
