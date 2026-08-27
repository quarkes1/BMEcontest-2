# 02 — Deep Learning for Intake Gesture Detection From Wrist-Worn Inertial Sensors

## 引用信息
- **作者**：Hamid Heydarian, Philip Rosenthal, ... Megan E. Rollo
- **出处**：IEEE Access, 2020, 8: 164936–164949. DOI: 10.1109/ACCESS.2020.3022042
- **链接**：https://ieeexplore.ieee.org/document/9187203 （Newcastle 开放获取版本：https://openresearch.newcastle.edu.au/articles/journal_contribution/Deep_learning_for_intake_gesture_detection_from_wrist-worn_inertial_sensors_the_effects_of_data_preprocessing_sensor_modalities_and_sensor_positions/29035184）
- **类型**：期刊论文

## 核心发现
1. 100 名参与者的腕部 IMU 数据（双手），半受控进食场景，逐帧"进食手势/非进食"标注
2. 系统比较了深度学习架构、传感器模态组合（加速度计/陀螺仪/两者）、佩戴位置（惯用手/非惯用手）
3. **CNN-LSTM 是最优架构**（F1=0.778）：CNN 提空间/手势特征 + LSTM 建模"接近-摄取-撤回"时序
4. **预处理结论（重要）**：
   - 连续应用 镜像（mirroring）→ 重力分离（gravity removal）→ 标准化（standardization）**有效**
   - **平滑（smoothing）有害**——会抹掉对识别有用的高频细节
5. 后续研究（FIC 数据集）：CNN 概率 → LSTM 时序两阶段，F1 可达 0.913（留一交叉验证）

## 对本项目的启示
- **L3a 深度可分离时序 CNN + 时序建模**与"CNN-LSTM 最优"的文献结论一致——我们的架构选择有文献支撑
- **预处理铁律**：镜像增强✓、重力分离✓（副产品给姿态角用）、标准化✓、**不平滑**✗（已在方案中固化）
- 其"非惯用手位置性能更低"的结论支持**分场景建模**的必要性（我们 L2 场景门控的依据之一）
- 报告可引用其模态消融结论：加速度计+陀螺仪组合优于单模态
