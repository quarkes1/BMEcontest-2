import type { Prediction } from '../data/types';
import { RELEASE_METADATA } from '../runtime/release-data.generated';

type ReleaseMetrics = { f1: number; ppv: number; sensitivity: number; n_tp: number; n_true: number; n_pred: number; fp?: number; fn?: number };
export type ModelReleaseMetadata = {
  run_key: string;
  official_metrics?: ReleaseMetrics;
  historical_reproduction?: { run_key: string; outer_metrics: ReleaseMetrics };
  outer_metrics?: ReleaseMetrics;
};

/** 构建时从 prepare-release.mjs 统一生成的元数据内嵌发布证据，独立页面无需额外请求。 */
export default function ModelPage({ prediction, release = RELEASE_METADATA as ModelReleaseMetadata }: { prediction: Prediction; release?: ModelReleaseMetadata }) {
  const historical = release.historical_reproduction ?? (release.outer_metrics
    ? { run_key: release.run_key, outer_metrics: release.outer_metrics }
    : undefined);
  return (
    <section className="panel model-page">
      <div className="page-title"><div><h1>Model</h1><p>Official submission model and historical reproduction.</p></div></div>
      <div className="model-grid">
        <div>
          {release.official_metrics && <>
            <h2>Official submission model</h2>
            <p>Official test score.</p>
            <dl>
              <dt>Run key</dt><dd className="mono">{release.run_key}</dd>
              <dt>Official test F1</dt><dd>{release.official_metrics.f1.toFixed(3)}</dd>
              <dt>Precision / PPV</dt><dd>{release.official_metrics.ppv.toFixed(3)}</dd>
              <dt>Recall / sensitivity</dt><dd>{release.official_metrics.sensitivity.toFixed(3)}</dd>
              <dt>TP / true events / predictions</dt><dd>{`${release.official_metrics.n_tp} / ${release.official_metrics.n_true} / ${release.official_metrics.n_pred}`}</dd>
              <dt>FP / FN</dt><dd>{`${release.official_metrics.fp ?? '—'} / ${release.official_metrics.fn ?? '—'}`}</dd>
            </dl>
          </>}
          {historical && <>
            <h2>Historical local reproduction</h2>
            <p>Subject-disjoint five-fold cross-validation development evidence. This is a historical local reproduction score, not the official test score.</p>
            <dl>
              <dt>Run key</dt><dd className="mono">{historical.run_key}</dd>
              <dt>Historical F1</dt><dd>{historical.outer_metrics.f1.toFixed(3)}</dd>
              <dt>Precision / PPV</dt><dd>{historical.outer_metrics.ppv.toFixed(3)}</dd>
              <dt>Recall / sensitivity</dt><dd>{historical.outer_metrics.sensitivity.toFixed(3)}</dd>
              <dt>TP / eligible / predictions</dt><dd>{`${historical.outer_metrics.n_tp} / ${historical.outer_metrics.n_true} / ${historical.outer_metrics.n_pred}`}</dd>
            </dl>
          </>}
        </div>
        <div>
          <h2>Inference path</h2>
          <div className="pipeline">Raw wrist IMU <span>↓</span> Macro / Micro <span>↓</span> Candidate admission <span>↓</span> Context + event verifier <span>↓</span> Eating episodes</div>
          <p className="muted">Visualization only consumes the published prediction contract and optional motion telemetry. Canonical inference always runs in Python.</p>
        </div>
      </div>
    </section>
  );
}
