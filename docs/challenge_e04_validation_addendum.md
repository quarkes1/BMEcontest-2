# E04 Validation Addendum — immutable installed artifact

This is a post-hoc validation record for the already-installed diagnostic-only
artifact at `outputs/challenger_70/e04_verifier_audit/`. It does not alter that
artifact, rerun E04, create F1, train, tune a policy, write a registry, or
change a release, model, or distribution. A future execution must create a new
F1 artifact rather than modify this one.

## Installed provenance

- Installed artifact code SHA: `9b29ac4c8a0594305d94eb3a7b524c986a09ac16` (from its immutable `manifest.json`).
- Independent post-hoc score-chain check: all 229 exported CSV rows from
  `reachable_fn_candidates.csv`, `tp_reference.csv`, and `fp_reference.csv`
  satisfy `blend_score = w * logistic_score + (1 - w) * lgbm_score` exactly
  under fold weights `{0: 0.50, 1: 0.25, 2: 0.25, 3: 0.00, 4: 0.25}`; maximum
  absolute error is `0`.

## Installed file inventory

All paths are relative to `outputs/challenger_70/e04_verifier_audit/`; hashes
are SHA-256 of the installed bytes.

| Path | SHA-256 |
|---|---|
| `candidate_examples/fold-0_1b22999e6c43b10f.json` | `37d5d52f43417e731c74ff0cd40023cefe074f19cc7fae2805caee63732f09ce` |
| `candidate_examples/fold-0_fe04a268fbe96475.json` | `3345cf8a677c275dffa6b2370c33bdb1bc4fce882791155b92529651ae2e7c0b` |
| `candidate_examples/fold-1_5be6324503072a32.json` | `a2dffd576876c81de61f22329dff2a14bdd7dcabc04fb6f3a7acef1e6032bdd5` |
| `candidate_examples/fold-2_0eaccfe1492a23eb.json` | `feaeaa99dd84963ca4d3d0abfb9380b8473842772908940045cb7173f1d4ad41` |
| `candidate_examples/fold-2_c645595bfed5d5a5.json` | `baea6b419cd556912191b7d61a325ea3e2445ca2d7711dcccbc91306527b6dfd` |
| `candidate_examples/fold-3_3d18ef78f948cf75.json` | `69792a7e46e208727a3bd509e7c39e64d56201ef5c10f8afaf8985981d9d662f` |
| `candidate_examples/fold-3_e0de7449d915fe7e.json` | `e01dd58d3097da5afaf228b985dedfbcc3126ecd2c683da453f07600e0f43623` |
| `candidate_examples/fold-4_307f439b027357f8.json` | `233c181f3a9ed922e0c3f3d4a8190a48fa6ad68cfc2b96d226ae1aec4aed10cb` |
| `candidate_examples/fold-4_c8d5b18af381ace5.json` | `42e69c7aa445fa0697b9321ebb8adb7566a92b330d714a1ba8861dc65b4c7916` |
| `context_contribution.json` | `eba203be37977dd6a267608e81c26fff2950546ee42cb1985cfe19fea562f40b` |
| `fp_reference.csv` | `9b0cd935b89f8b9e5a75766eb32bb0c9689883cadd9551eb51887f271a38326a` |
| `lr_lgbm_disagreement.json` | `cbc26a9200ad6090fee207cef81cb8b9c815bacfe67aecb6cbc2701127f082c9` |
| `manifest.json` | `0dbf50f172f52d59af43a4faab6fe0f5507b30feed82f07dc6c2c6d675b09419` |
| `reachable_fn_candidates.csv` | `3b05952a63e73a602f72cd0fab142343c4df14d94786688c26d732f8d97cc5d0` |
| `report.md` | `de96313c8a06961cb93af5a491b0bba4c81aacca046ff23845239ebe6feea6f8` |
| `score_margin_analysis.json` | `83411449bced140b5870ad3db2dd2c14986175241d8ddcc080efcbff88abd293` |
| `separability_by_fold.json` | `acad9dd7db9501aa31ec5942c70608182e48fd8b8b4937ac0dcb1f2e8ce9dc17` |
| `tp_reference.csv` | `e963fd43964c948f05f11c199923469200d4ab1eda433b473fdb5a405809cc41` |

## Preregistered legend for future E04 artifacts

The following flags are non-exclusive and descriptive: a witness can carry
more than one flag. A is frozen blend margin in `[-0.05, 0)`. B requires both
fitted models to have base contribution at least the same-fold matched-TP
median and Context contribution that is negative and at or below the
same-fold matched-TP first quartile. `B_model_disagreement` applies when
exactly one model meets B and both models' reference quantiles are defined. C
is `|LR-LGBM| >= 0.20`. D requires both probabilities below their same-fold
matched-TP first quartile and within their same-fold final-FP interquartile
range. `unsupported` is added when a required same-fold reference quantile is
undefined; `unresolved` applies when none of A/B/C/D or
`B_model_disagreement` is supported. These flags are explanatory labels, not
a prediction taxonomy, tuning rule, promotion rule, or challenger selection.
