# S14 — unit-free LIMU-BERT-X source transfer probe

## Spec / authority

2026-10-10。承接S13完整失败，执行用户“自主探索/继续下一阶段”授权。
目标仍为正式event F1≥.70；本阶段只回答固定外部预训练表示在FD-I源域是否有增量。
不读取新目标raw，不跑outer，不更新dist/release，不生成event F1。
这是unit-free输入桥接研究，**不是官方物理单位合同已经成立**。目标ADC换算仍未知；
HARNET留待物理/交付许可门解决。LIMU的公开工业预训练主体ID不可审计，不能宣称已
证明其与下游主体不重叠；后续研究必须披露这一限制。

## Fixed design

- 上游固定commit `bdfc3d0f734befaeb9375f5f7b8a7342f0ddfdcb`，MIT。
  checkpoint `weights/limu_bert_x.pt`，226783B；Git blob
  `06a6e26efb31ed25583ae1b2b5aa261ec336640c`，SHA256
  `65697e7017020f36429a221382253f1fc370fdfd493cfb71667f95846c18516c`。
  已在内存安全解析tensor state-dict确认shape；不得执行远程hub/code或放开pickle安全模式。
- 本地独立实现上游encoder计算：6输入，hidden72/FF144，4次共享attention/FF迭代，
  4heads，20位置，embedding双LayerNorm，epsilon1e-12；保留其非标准残差语义
  `h=attention(h); h=norm1(h+proj(h)); h=norm2(h+FF(h))`，不擅自改成标准Transformer。
  用固定上游models.py/config.py/config/limu_bert.json作出处，不import它们。
- 复用S0 FD-I右腕token、seed20260926、27/7人、8847/2274有效token及同一标签。
  原5秒/25Hz无量纲triad输入，resample_poly(4,5)→5秒/20Hz，分5个连续1秒×20点。
  20Hz是官方HHAR演示支持的工程选择，不是已验证的工业预训练采样率。
  保留原unitfree去均值/RMS，不再除9.8，不称为g或rad/s；域差异是本实验限制。
- 固定三臂：R frozen random LIMU（seed20260926）；F frozen pretrained LIMU；
  D frozen S0 FD encoder。所有神经权重冻结，0 optimizer steps。
- LIMU每1秒输出20×72，时间mean→72D。PCA72→32只拟合train的5×8847子窗口，
  svd_solver=full、不whiten；之后5子窗mean/std/max→96D。
  D沿用16D输出。三臂均使用已有subject-balanced scaler+LR C=.1/max_iter2000/tol1e-6，
  只拟合27个train主体。D只作强参考，不把其不同width解释为纯预训练因果对照。
- 记录逐人AP/BCE、median/IQR/p10、有效人数/方向，单类AP/AUC保持null。
  不搜索采样率、pooling、PCA维数、C、seed或checkpoint；不微调encoder。

## Gate (before new source scores)

完整性全部通过且F AP≥R AP+.01、F AP≥D AP+.01、F AP≥S0 .8440475046760654+.01；
F raw BCE≤R+.02、≤D+.02、≤S0 .25704393608246073+.02；AP≥prevalence+.10。
过门仅允许另外注册target probe；失败关闭这套无量纲桥接/池化/读出，不否定全部
foundation models，不以source AP充当event F1。状态S14_SUPPORTS_TARGET_PROBE或
S14_NO_SUPPORTED_FOUNDATION_TRANSFER；异常另记S14_INCOMPLETE。

## Implementation plan (inline, combined ledger)

> Use executing-plans + test-driven-development. Goal: implement the above source-only probe.
> Spec: this file. Architecture: isolated research module, reuse frozen source and linear readout helpers.
> Tech: current bme Python/PyTorch/SciPy/sklearn; no new dependencies.

Global constraints: existing research branch/workspace, no target raw/outer/dist writes; clean git before
run, no overwrite/retry, safe tensor loading, no persistent token/embedding cache. Training code and
spec committed before scores. Limits:2wallh,32MiB output,20GiB free disk,decoded source estimate≤80%
available RAM,CPU4,CUDAbatch512,peakVRAM≤80%initial free,CUDAinitialfree≥1.25GiB.

Review focus: exact nonstandard upstream graph; unknown checkpoint keys/dtype/nonfinite; temporal
resampling preserving duration; validation excluded from PCA/readout; network/resource failures
marked incomplete, never performance failures. Tests pin each boundary, no network in unit tests.

### Task1: encoder, input bridge, strict loader, source gate

Files `scripts/research/representation_v2/foundation_source.py`,
`tests/research/test_representation_v2_foundation_source.py`.
Produces `LimuEncoder`, `load_pretrained(path)`, `bridge_tokens(values)`, `foundation_gate(metrics,checks)`.
- [x] Write tests for manual independent graph parity, checkpoint roundtrip/wrong SHA/key/shape/dtype/
  nonfinite, valid N6×125→N5×20×6 bridge and duration, CPU/CUDA parity, gate boundaries.
- [x] Run `pytest tests/research/test_representation_v2_foundation_source.py -q -p no:cacheprovider`;
  expected missing-module RED. Implement and rerun; expected all GREEN. Commit.

### Task2: fold-contained linear probe and bounded runner

Same files. Consumes Task1 plus S0 source/reference; produces
`fit_source_probe(embeddings,labels,groups,train,validation,project)` with serialized PCA/readout+logits;
`run(root)` fixed CLI. Fit probe validates complete disjoint partitions; validation changes cannot
change fitted PCA or readout. Encoder states identical before/after encode.
- [x] Test real tiny probe, val perturbation invariance, invalid partitions/shapes and output preservation;
  no-overwrite before network/data, bounded download bytes/SHA, post-step/resource checks.
- [x] Observe missing-functions RED; implement to GREEN. Run all representation research tests and
  full `pytest -q -p no:cacheprovider` with torch CPU2; no new code during full regression. Commit
  before single source run. Expected finite outputs, identity replay and gate evaluated once.

### Task3: execute, review, report, clean

Output `outputs/challenger_70/representation_v2/amendment_a4_foundation_source/` contains fixed
plan, upstream LICENSE/code/config provenance, one checkpoint, three readouts/PCA, validation,
subject metrics, summary/state/report. No raw copies or token banks. Download only explicit small
files; no full clone/cache. Fixed snapshots of upstream code are reference text only, never executed.
- [x] Run module once, verify source fingerprints, population, baseline frozen-logit replay and output
  artifact identities. Review implementation+evidence, fix important findings with RED→GREEN.
- [ ] Update README and registry (outer_f1/event_f1 null), clear only this stage's rebuildable pyc/tmp,
  commit results and verify clean tree. Preserve small frozen weights/negative evidence for reproduction.

Ruling: use existing approved research workspace and combined spec/plan/ledger; no extra data copies.
Ruling: prior explicit autonomous-exploration authorization governs stage design/inline execution;
release approval and >.03 formal-gain escalation remain human-owned. No unregistered target probe.
Pre-flight: Task1 bridge yields5 subwindows, Task2 PCA fits only train and pools96D; D project=false
retains16D. Generic loader does not authorize a future target experiment. No shared production edits.

Task1 evidence: missing module RED, then16 passed (3.88s), including real CUDA parity and independent
functional graph. Metadata-only checkpoint inspection happened in memory before registration;
no source/target scores were read. Runtime dependency discovery found requests absent: use stdlib
urllib, do not install new dependencies. All native paths now resolved through rg inventory.
Task2 evidence:9 missing-function failures RED, then25 passed (6.04s). Real PCA/LR validation
perturbation leaves all fitted payloads bit-identical; full regression and independent review follow
before sealing. Checkbox marks implementation/related-test completion, not a claim of full-suite proof.
Final review: independent Luna/medium found one Important issue (resource preflight failures lacked
INCOMPLETE state), no Critical/Minor. Fixed in one pass: two RAM/GPU failure tests RED→GREEN;
all resource preflight rejection paths now preserve phase/error state, no summary. Existing attempt
and dirty-tree invocations remain refused without modifying an existing output. Initial full suite
was stopped before scores because it predated this fix; post-fix whole-suite regression runs afresh.
Declined-to-judge rulings: runtime scores/replay/report remain Task3, validated after execution;
target/release authority is explicitly out of scope, no target evaluation or dist change is allowed.
Ruling: overlap fixed source-only evaluation with post-fix full regression for efficiency; no code
changes while these run, and stage cannot be sealed until regression succeeds. If an integrity/test
failure emerges, preserve evidence and investigate rather than promote. No release risk is accepted.
Task3 execution: code12c33d0, complete negative. R/F/D AP .76428/.69747/.85144;
raw BCE .27218/.29392/.22677. All checks pass,0 neural steps,3 LR fits,29.26wall s,
peak CUDA约36.1MiB,output<1MiB,token cache0. Independent metric/gate/checkpoint replay agrees;
S0 max logit replay diff1.91e-6 within fixed tolerance. Full regression still pending; do not seal yet.

## Primary provenance

[Official README](https://github.com/WANDS-HKUST/LIMU-BERT_Experience/blob/bdfc3d0f734befaeb9375f5f7b8a7342f0ddfdcb/README.md),
[encoder](https://github.com/WANDS-HKUST/LIMU-BERT_Experience/blob/bdfc3d0f734befaeb9375f5f7b8a7342f0ddfdcb/models.py),
[config](https://github.com/WANDS-HKUST/LIMU-BERT_Experience/blob/bdfc3d0f734befaeb9375f5f7b8a7342f0ddfdcb/config.py),
[MIT licence](https://github.com/WANDS-HKUST/LIMU-BERT_Experience/blob/bdfc3d0f734befaeb9375f5f7b8a7342f0ddfdcb/LICENSE).
