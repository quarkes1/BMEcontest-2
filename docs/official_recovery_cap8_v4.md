# CAP8-V4：最终送检前的固定候选预算对照

2026-10-26。用户要求保留官方最佳基线并产出下一份可送检EXE，补充说明10月28日、最晚29日最终提交；本轮不扩大参数搜索、不开展新架构训练。用户已授权自主执行/main提交，正式包及前端不自动替换。

## 基线与唯一改动

官方最佳主试测official_deployment_f1_policy_v3：用户反馈56TP/42FP/56FN、112GT、F1=0.5333333333，原EXE SHA fa6987b6244fe73c76ee741ea352fcc74e8e4de06de332812e99b1f1d963aa44。先前49/27/63、.5212765957保留历史。收件SHA及同一隐藏主体未独立核验，不宣称统计显著性。

固定cap6→cap8，四模型、特征、原始分数、admission .2、NMS .3、blend .25、最终阈值均不变。实际新EXE继续使用.4582664763345077；折内比较继续用R3各fold已由train-only选出的主档阈值。不重新优化阈值、不重算组件分数、不按测试集饭点/ID或输出碎片设计规则。

官方CSV一半主体输出6条、部分跨多天，支持核查总cap，但不证明额外候选为真。以前DAY-BUDGET、E03.5反事实的失败不能忽略；本轮用一致性修复后的完整候选分数，与已送主档配对，不混旧D或抽样池对照。只跑一个cap=8，不接着搜索7/9/10或改成按天。若失败，不包装为改善；是否仍有必要送风险对照应明确报告，而不是浪费最后机会。

## 评估与技术门

先T24三组160餐，再通过档39人五折254餐。保持R3主档相同分母、原分数与训练侧阈值；原事件与原报告逐个一致，否则停。各阶段新policy在读留出候选前写入seal。源码/全部源账本/75保护文件前后SHA核对：原66加主试测8bundle+EXE。标记repeated development policy diagnostic、actual_exe_cv=False；不以此替换官方分数或历史eligible F1。

最后机会门（在评分前锁定）：aggregate F1至少+.005、TP不降，至少2/3或3/5折F1不差；新增FP不超过2倍新增TP（TP不增时不得增加FP）。T24不通过不消费新五折；五折不通过不推荐最终替换。不要求达到.70，也不降低门来保证有新EXE。记录净新增候选/事件、逐折TP/FP/FN、逐人变化。

通过时仅复制主试测模型，policy只改cap，版本official_cap8_policy_v4，dist/official_trials/cap8_policy_v4/EatingDetection.exe；不改旧bundle/EXE、正式submission或前端。沿用已验证单文件CPU打包叶函数，精确路径白名单、拒覆盖、DLL依赖内置、neutralcwd自检、中文路径EXE-only真实双文件CSV逐字节验收。冻结源码后最新完整pytest。新scratch自动清理，不绕过之前删除拒绝。

## 用户工作树边界

本轮开始用户删除dist/official_trials/deployment_f1_policy_v3/README.md，新增dist/official_trials/submission.zip。两者保持原样，不恢复、不删除、不stage。新阶段的源码前置检查仅允许这两个精确路径的既有状态，并记缺失/ZIP SHA；任何其他未提交源码/产物先提交本轮自己的文件。不能因旧builder的clean-tree门暂存用户文件或修改旧门。

## 预算及截止

零训练，probe≤10分钟/RSS12GiB/每阶段证据16MiB；编译≤30分钟、验收子进程≤300秒、磁盘≥10GiB。优先当天完成实验判断与交付；最终材料操作另行进行，用户送邮件/正式提交，不代发。若新方案无证据优势，以.533333原版回退，不牺牲可靠性换取“又一个版本”。

## 执行ledger

- Ruling：main原位执行，用户既有修改隔离保留——已明确授权main且截止紧，代价是不能宣称整个工作树干净；只声明本轮改动已提交。
- Pre-flight：probe给条件交付提供COMPLETE/过门证据和原模型身份；后续包装只能消费这些证据，不能通过改阈值补救失败。
