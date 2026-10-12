# A6 T0/T1 Transformer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在严格隔离的 T24 开发协议下完成数据审计与四臂事件复核实验，判断 Transformer 是否值得继续扩大验证；长期正式事件 F1 目标为 0.70+。

**Architecture:** 冻结既有 FD-I 局部动作 encoder，对全天有效时间提取 5s/16D tokens。候选内部、前后各 10min 上下文分别保留，使用相同输入比较池化、TCN、Transformer，再与固定 incumbent 概率融合；proposal 与事件匹配合同不变。

**Tech Stack:** 现有 bme Python 环境、NumPy、PyTorch、scikit-learn、LightGBM、pytest；不安装新依赖。

**Spec:** [A6 双任务设计](../../challenge_070_amendment_a6_transformer_design.md)。本计划仅实现其中 T0/T1；FS1、T2、T3 分阶段另行注册，不能在本次结果出来后静默加入。

## Global Constraints

- 本计划状态：待用户审阅；审阅前不实现、不训练。建议由主代理原生执行，减少共享接口的跨代理开销。
- 原分支原地执行；A5 保持暂停，不创建复制原始数据的工作树。
- 正式 F1 仍 0.6514285714，TP/FP/FN=114/83/39；不改 dist、release 或旧实验结论。
- 初始开发只使用原 T24；V7、outer raw/labels 不进入训练、采样、归一化拟合、预训练、阈值选择或早停。
- 首轮持久派生缓存上限 **2GiB**；不删除原始 FD、session cache、历史不可替代证据。
- 5s、25Hz、125 samples/token；时间缺口按现有绝对 200ms 规则处理，不插值跨缺口。
- FD source checkpoint SHA256：`c6276837b1366117075e7e8a7504e084926cbfc5d80b98b863e442255995efc4`；27/7 源域身份沿用原 manifest，无 FD 新训练。
- 任何身份冲突、缓存错配、非法标签、非有限 loss 或预算越界均停止；不能换划分或自动缩减实验臂补成绩。
- 每个实现任务通过针对性测试后及时提交；训练前全仓回归完整通过并归档，不以 A5 的中断记录代替。

## Review Focus

1. 首尾短时段与非 5s 整数长度：保留真实 coverage，不用 padding 伪造观测（任务 2）。
2. 超长餐段与餐中采集缺口：每个有效 token 恰好被聚合一次，跨缺口不做 attention/卷积（任务 2/3）。
3. 全无效候选与单类训练：候选不删除；前者回退到 incumbent，后者明确停止（任务 3/4）。
4. 政策选择前出现 health 文件或上次半成品：拒绝读取/覆盖，失败状态保留（任务 5）。
5. 受试者没有 GT 或没有预测：主指标沿用 canonical matcher；无 GT 个体 FP 单列（任务 6）。

## 锁定的实验合同

### A. 拓扑与四臂

ID：`RV2-A6-T1-001`。输出根：`outputs/challenger_70/representation_v2/amendment_a6_transformer/t1/`。

沿用 `registered_e03_config(0, workers=4)`，从 `FilesystemDataSource.load_outer_train_only` 得到旧 outer-train 特征，取 `inner_subject_partitions(data,n_splits=4)[0]` 的 train 为原 T24、validation 为 V7，立刻裁到 T24。特征载入的物理反序列化范围如实披露，不声称从未读过其他人的特征文件字节。

T24 用现有 `inner_subject_partitions(...,n_splits=3)` 形成三个 wrapper。每个 wrapper-train 再三折 policy OOF，最后 train→health refit；共 **12 个 split**。不重新随机分折，不采用 A5 的两个新盐值。

| 臂 | 新读出 | 最终候选 score |
| --- | --- | --- |
| b | 无 | incumbent LR/LGBM 116D 等权概率 |
| p | 固定局部 token 的 pooled head | .75 × incumbent + .25 × pooled probability |
| c | 固定局部 token 的 TCN head | .75 × incumbent + .25 × TCN probability |
| t | 固定局部 token 的 Transformer head | .75 × incumbent + .25 × Transformer probability |

固定融合沿用项目已有 .75/.25 对照思路，不搜索融合权重，也不额外跑 pure head 正式臂。pure head AP/BCE 可记录作诊断。p/c/t 的 proposal、监督、归一化、上下文、采样序列与训练预算相同。

每 split 调用一次 `build_candidate_ledgers`，共享 b/p/c/t 的候选与 116D base features。训练 ledger 的 macro/micro 是训练主体 OOF 输出；score ledger 是 train-only refit 输出。score 主体及 wrapper-health/V7/outer 不进入任何拟合祖先。

候选标签沿用 `_candidate_labels`：最大同 sid IoU≥.25 为 1，其余 0，不发明新的歧义过滤。116D 的 NaN 保持既有 verifier 处理。只对 p/c/t 的数值 token 表示做 train-only 标准化。

### B. 输入与 chunk 聚合

连续 session grid 以该 session 首个观测时间为锚点，每 5000ms 一窗，末窗裁到真实 session 末端。`prepare_session` 给出观测 spans；一个 token 跨越两个 span 时，不拼接两个 span 内容：只保留覆盖最长的单 span（同长取较早 span），coverage 分母仍为真实 token 的 125 个采样位置。少于 .8 覆盖或 row_rate≤20 的 token 标无效，保持时间索引。

使用 `build_token_bag` 的重采样/滤波合同和 `normalize_triads`；批量送冻结 encoder。不能一次把全天所有 6×125 原始 token 常驻内存：每 session 分块处理，原始块最大 360 tokens；滤波需从同一真实 span 取完整上下文，不把计算块边缘当物理缺口。编码前后比较 encoder state hash。

为每个候选构造三个区域：`[start-600s,start)`、`[start,end)`、`[end,end+600s)`，裁到 session。按 token center 的半开区间归属，三个区域不重叠；无中心落入的极短候选，选与其重叠最大的有效 token 给内部区域，并从外围移除该 token，不能删除候选。

每区域按 span identity 分开，再按时间顺序分成最多 360 token 的不重叠 chunk；保留所有 chunk，无最大候选时长截断。attention/TCN 只在同一 chunk 内传播；chunk 汇聚后再按区域合并，不跨缺口生成邻接关系。

先对 train ledger 中出现的**去重 session token**拟合 16D mean/std，受试者等权、每人 token 等权，scale floor=1e-6；score token 不参与。外围和内部不重复计入同一 token。coverage 作为额外标量，保持 [0,1]，不拟合归一化。

三个 head 共享结构规则而非权重：`Linear(17,64) → LayerNorm(64) → GELU`，加固定的 64D sinusoidal 实际相对秒数编码；时间原点为候选 start，以 300s 为单位，不用 token 下标或绝对时钟。p 没有时间传播层；c 使用两个 kernel=3、dilation=1/2 的残差 Conv1d block，64 channels、GELU、dropout .1；t 使用两层 pre-LN Transformer、64D/4 heads/FFN128/dropout .1。

所有 head 对 chunk 内有效输出累积 sum、sum-of-squares、count，再跨 chunk 合并，得到各区域 mean/std（population variance、floor=0）。三个区域各128D，按 before/inside/after 拼为384D，加三区域 valid/total coverage 三标量，`Linear(387,1)` 输出 logit。空区域全零；三个区域全无效时不送 attention，p/c/t 最终 score 直接等于 incumbent。

定位路径的 30min/15min responsibility ownership 在 T0 建立纯几何函数与测试，不启动 dense head：责任区 `[a+900j,a+900(j+1))`，上下文责任区前后各450s、裁到当前连续 span；每 token center 恰属一个责任区。

### C. 训练与计算预算

- 每 split 每新 head **1000 optimizer steps**；无早停、无 best epoch 选择，使用最终 checkpoint。
- AdamW，lr=.001、weight_decay=.01、clip_grad_norm=1；BCEWithLogitsLoss，无额外 class weight。
- 有效 batch=16 候选，先均匀抽 subject、再均匀抽该 subject 候选，有放回；三臂用完全相同的采样索引，seed=`20261012+split_id`，split_id=`wrapper*4+policy_index`，health-refit 的 policy_index=3。
- 初版 FP32，避免把混合精度引入首轮数值差异。CUDA 可用时使用 CUDA，否则 CPU；torch/BLAS 限4线程，DataLoader 不做嵌套进程并行。
- 可按候选 microbatch=1 累积梯度，每个有效 batch 一次更新；超长候选按 chunk 重计算/gradient checkpointing，不能截断。OOM 只允许保持有效 batch/steps/科学配置的执行修复。
- `config.inner_splits` 必须为4：每 split macro/micro 各4 OOF+1 refit，即10 producer fits；12 split 共 **120 producer fits**。
- base LR/LGBM 每 split 各1 fit，共 **24 base-verifier fits**；三新臂各12 fit，共 **36 neural fits / 36000 optimizer steps**。共180次模型 fit，source encoder 0 fit。
- 单新 head fit 上限20min GPU活跃时间；累计 neural GPU活跃≤12h，累计其它 CUDA 编码≤2h；总墙钟≤36h，CPU producer/验证部分累计≤12h。每 step/fit 边界检查预算并记录，超时停止，不输出完整 gate。
- 派生持久缓存≤2GiB，其他小证据≤256MiB，磁盘剩余≥20GiB；GPU峰值≤启动空闲显存80%。不足时停止，不挤占系统保留量。

### D. policy、gate 与报告

每个 wrapper 的四臂分别使用 `choose_policy` 原有 admission grid、minimum recall、event threshold selector 与 tie policy，不扩大网格。所有 wrapper 的 policy 文件都生成并提交 hash freeze marker 后，才可调用任何 health 推理/指标函数。policy 阶段不得打开 health raw，也不能通过日志打印 health label。

每臂完整三 wrapper aggregate F1 相对 b≥.01 且至少2/3 wrapper F1非下降，才可标为 `SUPPORTS_WIDER_VALIDATION`。不加隐含“TP绝不掉”等额外 gate；TP/FP、短餐和尾部损失必须报告。t 与 p/c 的差值决定是否有证据支持 attention 的额外价值，不把 t 相对 b 的提升自动归因于 attention。

若 t 未胜 p/c，仍如实报告最有用的输入/简单读出信号；不事后以 p/c 最好为由自动发布。无完整三折、预算越界或 integrity 失败时状态 `INCOMPLETE`，不得计算 promotion gate。

必交付：`report.md`、`manifest.json`、`registered_plan.json`、`state.json`、`policy_phase_frozen.json`、逐 split score/provenance、逐 wrapper/subject counts、`summary.json`、resource/fits ledger。完整成绩是 inner development evidence，不生成新的 outer headline；不调用 outer，不更新 dist。

## 文件职责与复用点

新包：`scripts/research/representation_v2/transformer/`。

| 文件 | 单一职责 |
| --- | --- |
| `__init__.py` | 包声明，不在 import 时读取数据 |
| `contract.py` | 不可变配置、身份/缓存 key、预算计数 |
| `timeline.py` | 连续 span tokens、区域/chunk ownership、时间位置 |
| `cache.py` | 单 session 编码、原子缓存、容量限制 |
| `heads.py` | p/c/t 模型与无效区域安全汇聚 |
| `training.py` | train-only moments、配对采样、固定步数拟合 |
| `probe.py` | manifest/audit/policy/health 状态机与命令行 |
| `reporting.py` | 四臂 canonical counts、subject/short/coverage 诊断 |

测试均为 `tests/research/test_representation_v2_transformer_{contract,timeline,cache,heads,training,probe,reporting}.py`。不修改 A5 或旧实验实现；复用 `crossfit.py`、`event_probe.py` 的现有函数而非复制整套 runner。

## Task 1: 固定合同、身份与 fit ledger

**Files:** Create `transformer/__init__.py`、`transformer/contract.py`；Test `test_representation_v2_transformer_contract.py`；路径前缀见上表。

**Interfaces:** `T1Config` frozen dataclass；`validate_identities(identities: dict[str,list[str]]) -> None`；`cache_key(meta: dict) -> str`；`validate_fit_counts(counts: dict[str,int]) -> None`。

- [ ] 写失败测试，覆盖空/重复身份、T24不是24、source27/7不符、目标/source外部身份映射尚未确认、缓存encoder/归一化/时间身份变化。

```python
def test_registered_counts_reject_missing_refit():
    from scripts.research.representation_v2.transformer.contract import validate_fit_counts
    import pytest
    expected = dict(producer_fits=120, base_fits=24, neural_fits=36, steps=36000)
    validate_fit_counts(expected)
    with pytest.raises(ValueError):
        validate_fit_counts({**expected, 'producer_fits': 119})
```

- [ ] 用 `D:/Anaconda3/envs/bme/python.exe -m pytest tests/research/test_representation_v2_transformer_contract.py -q` 确认失败来自缺失实现。
- [ ] 实现常量合同与严格 key 集合验证；身份校验不能仅凭 `FD-R-*` 命名证明跨数据身份无重叠：manifest注明当前外部身份的证据与剩余限制，未确认重叠风险则T0停止训练。

```python
def validate_fit_counts(counts):
    expected = dict(producer_fits=120, base_fits=24, neural_fits=36, steps=36000)
    if counts != expected:
        raise ValueError('registered fit budget differs')
```

- [ ] 跑同一测试至通过；显式 git add 两个实现文件和测试，commit `feat: register A6 T1 experiment contract`。

## Task 2: 时间轴、全长候选与定位 ownership

**Files:** Create `transformer/timeline.py`；Test `test_representation_v2_transformer_timeline.py`。

**Interfaces:** `SessionTokens(sid, center_ms, span_id, coverage, valid, embeddings)`，embeddings=N×16；`owned_cells(start_ms:int,end_ms:int,centers:np.ndarray)->np.ndarray`；`candidate_chunks(tokens:SessionTokens,start_ms:int,end_ms:int)->list[RegionChunk]`。`RegionChunk` 包含 region、token_indices、span_id，三 region 编号0/1/2。

- [ ] 写失败测试：首尾、空区、极短候选、跨span候选、>30min餐段、每token不重复、缺口不产生chunk邻接。

```python
def test_responsibility_is_unique_at_boundaries():
    from scripts.research.representation_v2.transformer.timeline import owned_cells
    import numpy as np
    np.testing.assert_array_equal(
        owned_cells(0, 1800001, np.array([0, 899999, 900000, 1800000])),
        [0, 0, 1, 2])
```

- [ ] 运行该文件确认红灯；实现半开区间、span-local分块和短候选重叠fallback。

```python
def owned_cells(start_ms, end_ms, centers):
    centers = np.asarray(centers, dtype=np.int64)
    if end_ms <= start_ms or np.any((centers < start_ms) | (centers >= end_ms)):
        raise ValueError('centers outside owned span')
    return (centers - start_ms) // 900000
```

- [ ] 运行测试至通过；对45min候选断言全部内部有效token进入恰好一个chunk，不能只检查shape；提交 `feat: preserve full candidate timelines for A6`。

## Task 3: 冻结编码与容量受限缓存

**Files:** Create `transformer/cache.py`；Test `test_representation_v2_transformer_cache.py`。

**Interfaces:** `encode_session(root:Path,sid:str,allowed_sids:set[str],encoder,device:str)->SessionTokens`；`load_or_encode(..., cache_root:Path, metadata:dict)->SessionTokens`；`audit_data(root:Path,manifest:dict)->dict`。复用 prepare_session/build_token_bag/normalize_triads，明确 span 与滤波块边界实现。

- [ ] 写测试，受保护sid必须在 `np.load` 前被拒绝；伪造encoder哈希/文件校验和拒绝；小容量参数测试quota；源encoder状态保持不变。

```python
def test_disallowed_session_is_rejected_before_io(tmp_path):
    import pytest
    from scripts.research.representation_v2.transformer.cache import encode_session
    with pytest.raises(ValueError, match='not allowed'):
        encode_session(tmp_path, 'heldout', {'train'}, None, 'cpu')
```

- [ ] 确认测试失败；实现sid门禁在首行、session逐块读取/编码、无pickle的新缓存、原子tmp→replace与finally清理。

```python
if sid not in allowed_sids:
    raise ValueError('session not allowed')
# 在任何 raw/cache 读取之前执行；缓存身份包括 raw checksum、配置和 encoder hash。
```

- [ ] 扩展合成packet/200ms gap/末端不足5s测试，证明不会把两span相加达到coverage；跑测试通过后提交 `feat: add bounded frozen-token cache for A6`。

## Task 4: 三读出、严格 train-only 拟合

**Files:** Create `transformer/heads.py`、`transformer/training.py`；Test `test_representation_v2_transformer_heads.py`、`test_representation_v2_transformer_training.py`。

**Interfaces:** `build_head(arm:str)->nn.Module`；head.forward 输入 region chunks 与实际时间，输出每候选logit；`fit_head(arm,train_sequences,labels,groups,*,seed,device,steps=1000)->dict`；`predict_head(fitted,sequences)->np.ndarray`；`fuse_scores(base:np.ndarray,motion:np.ndarray,valid:np.ndarray)->np.ndarray`。

- [ ] 写失败测试：三臂有限logit、全空区不NaN、padding不影响、单类停止、同seed配对采样、moments不读取score主体、冻结encoder无梯度。

```python
def test_invalid_motion_preserves_incumbent_score():
    import numpy as np
    from scripts.research.representation_v2.transformer.training import fuse_scores
    np.testing.assert_allclose(
        fuse_scores(np.array([.8, .2]), np.array([.4, .9]), np.array([True, False])),
        [.7, .2])
```

- [ ] 跑两文件确认失败；实现合同 B 的三个head、有效sum/sumsq/count全长汇聚与合同 C 的更新循环；禁止 all-masked attention。

```python
def fuse_scores(base, motion, valid):
    base, motion, valid = np.asarray(base), np.asarray(motion), np.asarray(valid, dtype=bool)
    if base.ndim != 1 or motion.shape != base.shape or valid.shape != base.shape:
        raise ValueError('score alignment differs')
    if not np.isfinite(base).all() or not np.isfinite(motion).all():
        raise ValueError('nonfinite probability')
    if np.any((base < 0) | (base > 1) | (motion < 0) | (motion > 1)):
        raise ValueError('invalid probability')
    return np.where(valid, .75 * base + .25 * motion, base)
```

- [ ] 加入2-step合成训练smoke，确认 optimizer step数/clipnorm和采样索引审计，不能跑真实数据代替单测；CPU/CUDA可用时probability `atol=1e-5,rtol=1e-4`，阈值±1e-5外决策一致。
- [ ] 测试通过后提交 `feat: add paired pooling TCN transformer readouts`。

## Task 5: 可恢复但不覆盖的实验状态机

**Files:** Create `transformer/probe.py`；Test `test_representation_v2_transformer_probe.py`。

**Interfaces:** `build_manifest(root:Path)->dict`；`run_audit(root:Path)->None`；`run_policy(root:Path)->None`；`run_health(root:Path)->None`；模块CLI `--stage manifest|audit|policy|health|report --root D:/BMEtest`。

- [ ] 用依赖注入的合成callback写失败测试：所有policy冻结前health callback从未调用，现有attempt禁止覆盖，故障原子记录 `INCOMPLETE`，代码/manifest/spec hash改变拒绝恢复。

```python
def test_early_health_is_blocked(tmp_path):
    import pytest
    from scripts.research.representation_v2.transformer.probe import run_health
    with pytest.raises(RuntimeError, match='policies not frozen'):
        run_health(tmp_path)
```

- [ ] 确认红灯；实现 `MANIFEST_READY → AUDIT_PASSED → POLICIES_FROZEN → HEALTH_COMPLETE → COMPLETE`，任何阶段异常→`INCOMPLETE`。同一attempt阶段不可重跑；执行故障修复须新登记attempt并保留旧证据，不能删除后重试冒充首跑。
- [ ] 每split复用 build_candidate_ledgers 和 verifier_probabilities；图记录producer/source/moments/p/c/t/policy祖先，使用 assert_no_heldout_ancestors。只向cache传当前train/score允许sid；policy阶段score是policy-val不是health。

```python
from scripts.research.representation_v2.event_probe import choose_policy, decode_policy
# policy阶段收齐四臂三折subject-OOF分数，按原config选择；health阶段只读已冻结policy。
```

- [ ] fit ledger累计精确120/24/36/36000；每次拟合记录允许主体、实际主体、seed、elapsed和CUDA峰值，禁止训练函数接收health标签。
- [ ] 全部状态机测试通过后提交 `feat: orchestrate sealed A6 inner experiments`。

## Task 6: 四臂报告、执行与清理

**Files:** Create `transformer/reporting.py`；Test `test_representation_v2_transformer_reporting.py`；Modify `README.md`、A6设计状态；实验输出写入合同A目录。

**Interfaces:** `summarize(records:list[dict])->dict`，要求3唯一wrapper、24唯一health主体、四臂相同GT分母；`write_report(output:Path,summary:dict)->None`。复用 compute_event_metrics、event_gate所用counts公式，不调用旧三臂gate。

- [ ] 写失败测试：不完整fold拒绝汇总、no-GT FP不丢、匹配短餐沿canonical规则、1/3巨大增益不能替代多数方向、不能把inner改成outer字段。

```python
def test_incomplete_fold_set_has_no_gate():
    import pytest
    from scripts.research.representation_v2.transformer.reporting import summarize
    with pytest.raises(ValueError, match='three complete'):
        summarize([])
```

- [ ] 确认红灯；实现主指标和subject median/IQR/P10（有GT人群单列）、no-GT FP、改善/恶化人数、candidate可达率、<10min短餐recall、paired t-p/t-c、成本表。
- [ ] 跑新增7个文件及全部现有 representation_v2 测试；PowerShell 用 `rg --files tests/research` 取得实际文件数组，不把CLI位置参数写成字面 `*.py`。然后跑 `D:/Anaconda3/envs/bme/python.exe -m pytest -q` 完整全仓测试并记录最终摘要；中断不能记为通过。
- [ ] 更新文档为实现完成/尚未训练，检查diff、commit代码和协议；再从干净工作树运行manifest与audit，提交manifest/audit和code hash。

```powershell
& D:/Anaconda3/envs/bme/python.exe -m scripts.research.representation_v2.transformer.probe --stage manifest --root D:/BMEtest
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& D:/Anaconda3/envs/bme/python.exe -m scripts.research.representation_v2.transformer.probe --stage audit --root D:/BMEtest
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

- [ ] audit通过且证据提交后跑policy；冻结全部policy并提交marker后才跑health和report。命令均沿同一CLI，任何非零退出码即停止下一阶段。
- [ ] 完成后检查真实fit ledger、缓存容量、protected_snapshot不变、完整三折与24主体覆盖；只删除本attempt可重建缓存/已关闭临时文件，保留必要最终checkpoint和score/provenance证据，不操作宽泛目录。
- [ ] 更新README和A6状态、报告所有四臂F1及TP/FP/FN，commit结果；若失败报告停止位置和已消耗预算，不交付伪造完整成绩。用户再决定FS1/T2或扩大验证；本次不调用outer、不更新dist。

## 计划自审与交接

T0/T1覆盖设计的数据隔离、全长时间轴、四臂归因、缓存与计算边界；定位ownership仅做纯函数测试，不代表T2已经完成。FS1/T2/T3明确排除出本轮，而非未完成的实现占位。

本计划待用户确认实验合同与原生执行方式后开始。最终交付是完整三折inner结果或有证据的终止报告，不承诺必达0.70，也不把inner开发成绩当正式成绩。
