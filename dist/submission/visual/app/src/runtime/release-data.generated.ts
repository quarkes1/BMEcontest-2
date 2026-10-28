// 由 tools/prepare-release.mjs 根据最终提交元数据及交叉拟合摘要生成。
// 请通过 npm run build 或开发前置步骤重新生成，不直接修改此文件。
export type ReleaseMetrics = { f1: number; ppv: number; sensitivity: number; n_tp: number; n_true: number; n_pred: number; fp?: number; fn?: number };
export type ReleaseMetadata = {
  run_key: string;
  official_metrics?: ReleaseMetrics;
  historical_reproduction?: { run_key: string; outer_metrics: ReleaseMetrics };
  outer_metrics?: ReleaseMetrics;
};

export const RELEASE_METADATA: ReleaseMetadata = {
  "run_key": "official_deployment_f1_policy_v3",
  "official_metrics": {
    "f1": 0.5333333333333333,
    "ppv": 0.5714285714285714,
    "sensitivity": 0.5,
    "n_tp": 56,
    "n_true": 112,
    "n_pred": 98,
    "fp": 42,
    "fn": 56
  },
  "historical_reproduction": {
    "run_key": "160afaf81debf1ee",
    "outer_metrics": {
      "f1": 0.6514285714285715,
      "n_pred": 197,
      "n_tp": 114,
      "n_true": 153,
      "ppv": 0.5786802030456852,
      "sensitivity": 0.7450980392156863
    }
  }
};
