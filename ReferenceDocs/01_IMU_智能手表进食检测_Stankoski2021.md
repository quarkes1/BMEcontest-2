# 01 — Smartwatch-Based Eating Detection: Data Selection for Machine Learning from Imbalanced Data with Imperfect Labels

## 引用信息
- **作者**：Simon Stankoski, Marko Jordan, Hristijan Gjoreski, Mitja Luštrek
- **出处**：Sensors (Basel), 2021, 21(5): 1902. DOI: 10.3390/s21051902
- **链接**：https://www.mdpi.com/1424-8220/21/5/1902
- **类型**：期刊论文（Open Access）

## 核心发现
1. 12 名受试者**自由生活（in the wild）**数据，对食物、餐具、地点无限制；手表加速度计+陀螺仪
2. **新颖的数据选择方法**：从不平衡+标签不完美的数据中构建训练集——专门挑"难以区分的实例"训练可同时提升 precision 和 recall
3. **深度特征 + 经典特征 + HMM 时序建模**的融合框架；HMM 对连续分类的时间依赖建模显著改善进食片段识别
4. 人物无关（person-independent）评估：**precision 0.85 / recall 0.81 / F1 0.82**，对易与进食混淆的活动鲁棒
5. 显式检验了**不同餐具类型（cutlery）的泛化能力**——与本竞赛"筷子/刀叉/徒手动作差异大"的关切直接对应

## 对本项目的启示
- **L4 HMM 解码有直接背书**：先验参数、状态转移惩罚可参考其设置思路
- **L3a 训练集构建**：参考其"数据选择"思想——负样本中多采样"接近进食的难例"（如喝水、刷手机、整理餐具等混淆动作时段）
- **餐具鲁棒性**：按 tablewareType 分层评估 + 训练集平衡各餐具类型（我们已采纳多任务辅助头）
- **报告对照实验素材**：可作为 baseline 对比（其 F1=0.82 是自由生活场景下的参考线，注意其评估口径与我们 IoU>0.25 不同，引用时需注明口径差异）
