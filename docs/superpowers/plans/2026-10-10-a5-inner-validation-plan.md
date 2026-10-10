# A5 Partition-Expanded Inner Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Native execution is recommended; no subagents are dispatched by this plan without a separate explicit request.

**Goal:** 在两组固定T24划分中检验冻结FD和S11的事件收益及互补，不生成正式outer成绩。

**Architecture:** 新入口负责manifest、六wrapper的policy冻结屏障、health评分及开发判读。
复用现有split级候选/编码/verifier逻辑，通过显式回放与资源回调避免复制整套训练代码。
统计模块独立于训练入口，保留旧S5/S8/S11默认行为及历史结论。

**Tech Stack:** 现有Python、NumPy、PyTorch、scikit-learn、LightGBM、pytest；不安装依赖。

**Spec:** `docs/challenge_070_amendment_a5_inner_validation_design.md`（用户已审阅通过）。

## Global Constraints

- ID `RV2-A5-S15-001`；NO OUTER / NO DIST / NO RELEASE / NO PROMOTION。
- 原T24，不引入原V7/outer/source-val；24名独立受试者，两次划分不等于48人。
- 盐值20261012、20261013；三折8health/16train；policy OOF三折，原producer OOF不改。
- B116D；R冻结FD251D；F固定S11 batch32/norm1、1000steps、seed20261011。
- source SHA `c6276837b1366117075e7e8a7504e084926cbfc5d80b98b863e442255995efc4`。
- 预计24split/240producer/144verifier/24initial-readout/24000steps，实际计数须核对。
- 8wallh、4h同步adaptation+encoding、128MiB证据、20GiB磁盘保留；CPU4线程。
- CUDA初始空闲>=1280MiB、峰值<=初始空闲80%；每split前后及每100适配步检查。
- 输出仅 `outputs/challenger_70/representation_v2/amendment_a5_validation/`；禁止覆写attempt。
- 六wrapper全部policy冻结后才health；任何不完整结果不得选择winner。
- 每次各自聚合F1，再取两次算术平均；不拼接重复GT生成“新样本”统计。

## Review Focus

1. 同一划分仅fold编号置换仍应判等；两盐值不能静默替换。Task 1覆盖。
2. 旧S8强制replay不能被新入口全局绕过；默认调用仍拒绝旧证据不匹配。Task 2覆盖。
3. 磁盘/显存/时间运行中超限必须中断并留下INCOMPLETE，不能只末尾检查。Tasks 2/3覆盖。
4. 无candidate的session和无预测的subject仍保留完整GT分母；缺失运动保持原NaN策略。Task 4覆盖。
5. IoU并列时matching顺序、短餐统计及FP清单必须与canonical matcher完全对齐。Task 4覆盖。

## Task 1 — Manifest与固定划分（不训练）

**Files:** Create `scripts/research/representation_v2/validation_probe.py`；Create
`tests/research/test_representation_v2_validation_probe.py`。

**Interfaces:** `registered_partitions(subjects: tuple[str, ...], historical: tuple) -> tuple`
返回两个repeat，每个repeat三个`(train, health)`元组；`build_manifest(data, payload, config) -> dict`
记录T24/V7/outer/source身份、配置、历史及新划分、代码/spec/source hash。

- [ ] Step 1: 写分组确定性、重复subject/人数不符/等价划分拒绝的失败测试：

```python
def test_registered_partitions_are_subject_only_and_balanced():
    from scripts.research.representation_v2 import validation_probe as vp
    subjects = tuple(f"p{i:02d}" for i in range(24))
    historical = tuple((tuple(s for s in subjects if s not in subjects[k*8:(k+1)*8]),
                        subjects[k*8:(k+1)*8]) for k in range(3))
    repeats = vp.registered_partitions(subjects, historical)
    assert repeats == vp.registered_partitions(tuple(reversed(subjects)), historical)
    for folds in repeats:
        assert sorted(s for _, h in folds for s in h) == sorted(subjects)
        assert all(len(t) == 16 and len(h) == 8 and not set(t) & set(h) for t, h in folds)
```

- [ ] Step 2: 运行 `D:/Anaconda3/envs/bme/python.exe -m pytest tests/research/test_representation_v2_validation_probe.py -q`，观察缺失接口RED。
- [ ] Step 3: 用以下算法实现分组，输入必须先校验24个唯一非空字符串；不接收labels：

```python
order = sorted(subjects, key=lambda s: (hashlib.sha256(f"{salt}:{s}".encode()).hexdigest(), s))
health = tuple(sorted(order[k::3]))
train = tuple(sorted(set(subjects) - set(health)))
signature = tuple(sorted(tuple(sorted(h)) for _, h in folds))
```

`signature`用于检测两个repeat及历史组的等价；outer身份来自既有config的折清单，不读outer信号。
manifest保存原始身份和namespace，source训练人数27、验证人数7；hash采用排序JSON UTF-8。
`--stage manifest`要求clean git，载入既有特征后立即裁T24，创建唯一attempt并写plan/manifest/state，
不加载target raw或训练模型。冻结manifest需git提交后才能运行`--stage event`。
- [ ] Step 4: 增加manifest篡改、protected身份重叠及缺训练窗测试并运行至GREEN；检查无训练调用。
- [ ] Step 5: 提交该任务代码/测试，`git diff --cached --check`失败则不得commit。

## Task 2 — 显式split回放策略与适配预算检查

**Files:** Modify `scripts/research/representation_v2/adapt_probe.py`；Modify
`tests/research/test_representation_v2_adapt_probe.py`；Modify新validation模块/测试。

**Interfaces:** `adapt_encoder(..., resource_check=None)`；
`prepare(..., adaptation_options=None, replay=None, resource_check=None)`。
replay签名沿用`(root, name, evidence) -> str`，None仍调用`jp.baseline_replay`。
resource_check签名`() -> None`，正常返回或抛异常；不得改变优化器、采样或RNG序列。

- [ ] Step 1: 先为新可选回调写失败测试，真实两步训练必须调用检查且不改变无回调权重。

```python
def test_resource_failure_interrupts_before_adaptation(basic_adaptation_inputs):
    def stop():
        raise RuntimeError("registered budget exceeded")
    with pytest.raises(RuntimeError, match="budget"):
        ap.adapt_encoder(**basic_adaptation_inputs, resource_check=stop)
```

`basic_adaptation_inputs`在测试文件中用已有`bag()`构建8个双类别候选、两subject、两source组、
MotionBranch encoder与Linear head；kwargs使用steps=2、seed=2、device='cpu'。
- [ ] Step 2: 运行新增测试，确认因缺失参数失败；然后在适配开始、step%100==0和最终步检查预算。

```python
if resource_check is not None:
    resource_check()
# After opt.step(), without touching RNG or optimizer parameters:
if resource_check is not None and (step % 100 == 0 or step == steps):
    resource_check()
```

prepare将回调显式传入adapt_encoder，默认replay路径保留原证据字段；新replay使用
`baseline_validation_sha256`而非伪造历史source hash。新回放校验ledger/身份/有限B概率/图，
同split scorer等价通过测试中独立调用`ep.verifier_probabilities`确认，不在研究运行中增加B拟合。
- [ ] Step 3: 新replay不能使用git旧fold证据；旧None路径继续调用jp.baseline_replay，
用mock spy证明被调用并保留篡改拒绝。真实B及251D scorer测试，不以mock代替算法一致性。
- [ ] Step 4: 运行adapt/joint/optimization及validation测试至GREEN，验证全无效运动质量列不变。
- [ ] Step 5: 提交接口改动；不修改joint_probe.run或旧event_gate。

## Task 3 — 六wrapper执行与全policy冻结屏障

**Files:** Modify `scripts/research/representation_v2/validation_probe.py`及对应测试。

**Interfaces:** `run_event(root: Path) -> None`；`load_manifest(root: Path) -> dict`；
`freeze_policies(output: Path, manifest: dict) -> str`；
`load_all_policies(output: Path, marker_sha: str, manifest: dict) -> dict`；
`check_resources(root, output, started, work_seconds, initial_free_cuda) -> None`。

- [ ] Step 1: 失败测试覆盖缺一份policy、marker/manifest/文件hash篡改、health提前调用、
已存在event状态拒绝重跑；用调用记录fake split证明先18次policy再6次health。

```python
def test_policy_freeze_requires_all_six_wrappers(tmp_path):
    with pytest.raises(RuntimeError):
        vp.freeze_policies(tmp_path, {"wrapper_ids": list(range(6))})
```

- [ ] Step 2: 按既有joint引擎建立新独立组织器。每repeat/fold从manifest取train/health，
policy沿用原三折GroupKFold；三臂共享split ledger，使用Task 2显式回调，名字含repeat/fold。

```python
for wrapper in manifest["wrappers"]:
    for inner, (fit, valid) in enumerate(policy_partitions(wrapper)):
        ledger, scores, graph, seconds = splitter_for_wrapper(wrapper, inner, fit, valid)
        # Aggregate aligned OOF candidates/scores and choose existing policies per arm.
    save_policy_selection(wrapper)
marker_sha = freeze_policies(output, manifest)
policies = load_all_policies(output, marker_sha, manifest)
for wrapper in manifest["wrappers"]:
    score_health_with_fresh_refit(wrapper, policies)
```

其中`policy_partitions`为slice_outer_train+inner_subject_partitions(n_splits=3)；
`splitter_for_wrapper`只是ap.prepare返回的split在当前wrapper参数上的调用；
`save_policy_selection`写policy/图/hash；`score_health_with_fresh_refit`调用同split后decode_policy，
仅此阶段计算health指标。实现时这些局部步骤可以内联，不额外创建公共API。

所有祖先检查包括health、V7、outer及source-val；冻结marker含六wrapper ID、manifest/plan hash、
六policy文件hash及health_files_read=0。每次health前重新核验全部文件，不能只核验当前fold。
异常写S15_INCOMPLETE及completed split/wrapper计数，保护快照有变化也必须失败。

- [ ] Step 3: budget检查使用墙钟、同步work计时、shutil.disk_usage、输出文件字节及CUDA峰值。
训练过程回调读当前计时：已累计work+当前split计时，避免适配中work预算漏检。
在CPU、CUDA两路径测试边界与异常；预检失败也写INCOMPLETE，不留下伪成功state。
- [ ] Step 4: 实际producer数从fit graph的macro/micro OOF/refit节点核算；verifier按三个臂
各两次成功fit核算，readout和steps从实际适配report核算；结束校验240/144/24/24000。
测试重复节点、缺split、少steps拒绝完成，所有资源/完整性项以真实结果而非常量True记录。
- [ ] Step 5: 运行新测试至GREEN，提交可执行入口；此时仍不读取新health成绩。

## Task 4 — 汇总、互补与完整分母

**Files:** Create `scripts/research/representation_v2/validation_metrics.py`；Modify对应validation测试。

**Interfaces:** `summarize_repeats(records: list[dict]) -> dict`；
`matched_indices(predictions: tuple, truths: tuple) -> tuple[set[int], set[int]]`；
`complementarity(records: list[dict]) -> dict`。

- [ ] Step 1: 先写不等事件数的两repeat测试，验证mean(pooled F1)不是拼接F1；
阈值边界delta=0/.005、R/F差值=.001、只一repeat改善均覆盖。

```python
assert classify_gain([.006, .004]) == "SUPPORTED_NEXT_DESIGN"
assert classify_gain([.02, -.001]) == "UNSTABLE_GAIN"
assert classify_gain([0., 0.]) == "NO_SUPPORTED_GAIN"
```

`classify_gain(deltas: list[float]) -> str`定义在validation_metrics；必须恰好两项且有限。
- [ ] Step 2: 对完整truths复刻canonical greedy matching：按pred索引外层、GT索引内层
创建IoU>=.25且同sid的pair，稳定按-IoU排序后逐一占用，返回占用索引。
对并列IoU、重复GT、跨sid、空pred/空GT测试nTP与compute_event_metrics完全一致。
短餐仅在完整匹配后统计duration<600000的已占用GT，不重匹配短餐子集。
- [ ] Step 3: 汇总每repeat及每subject，subject来源完整subject_by_session，保留无candidate会话。
输出GT三臂命中布尔、独有命中及共同FN；各臂FP保存未匹配预测完整区间，不构造GT-aware模型。
候选AP/BCE遇单类/空集记录null与原因，不能删除该subject或折；event分母不受其影响。
- [ ] Step 4: 报告两次指标、逐人分位数/增减人数、24人delta均值及成本；无F1置信度或独立泛化宣称。
新增统计测试至GREEN后提交；不调用旧三fold复合event_gate。

## Task 5 — 锁定、运行、核验与收口

**Files:** Modify `README.md`、A5设计状态、append-only
`outputs/challenger_70/experiment_registry.jsonl`；Create A5输出证据。

- [ ] Step 1: 执行相关表示研究测试；共享adapt代码已变更，执行一次全仓回归。

```powershell
& D:/Anaconda3/envs/bme/python.exe -m pytest tests/research -q
& D:/Anaconda3/envs/bme/python.exe -m pytest -q
```

任何失败先诊断修复；每条命令分别检查exit code，不用末条成功掩盖前条失败。
保存测试证据，不在运行期间反复重跑全仓；代码后续变更则针对风险补验证。
- [ ] Step 2: 代码/spec/测试证据先commit，保持clean，再运行manifest阶段：

```powershell
& D:/Anaconda3/envs/bme/python.exe -m scripts.research.representation_v2.validation_probe --stage manifest
```

核查六组8/16和排除身份；commit plan/manifest后运行event阶段，不更换盐值或超参数：

```powershell
& D:/Anaconda3/envs/bme/python.exe -m scripts.research.representation_v2.validation_probe --stage event
```

event预检校验manifest提交及工作树clean，source/code/spec与plan一致；结果只能写新证据。
plan锚定manifest生成前的代码commit及受影响源码逐文件hash；manifest提交会自然改变HEAD，
因此校验代码commit仍为HEAD祖先且源码hash未变，不能要求HEAD等于旧代码commit。
spec也用生成manifest时的逐字节hash，不以之后更新的执行状态反向改变预注册规格。
长训练用后台会话，持续报告完成split/预算，不以半程成绩选臂或提早停掉较差折。
- [ ] Step 3: 六fold全部完成后，独立从保存pred/truth重算每repeat TP/FP/FN/F1，
核验summary、fit graph、freeze hash及protected snapshot；失败则标INCOMPLETE。
- [ ] Step 4: README与registry记录development_only、outer_calls=0、正式event_f1=null、
两次详细结果及臂的独立判读；不改历史失败行，不宣布正式F1提高。
- [ ] Step 5: 清理只限本阶段新增pyc、pytest临时文件和可重建缓存；先检查绝对路径及归属，
保留必要权重/评分/报告，不动其他任务文件。git diff --check通过后显式add/commit并检查clean。

## Self-review / handoff

此计划覆盖设计第1–10节；新模型、融合、outer和发布明确不在执行范围。
全部公共接口已在所属Task定义；边界测试对应Review Focus；无未定参数。
推荐Native：三个候选强共享split逻辑，由一个实现者连续维护接口可降低协调成本。
计划待用户审阅；批准后使用executing-plans按Task 1–5执行，不因设计已批准跳过计划审阅。
