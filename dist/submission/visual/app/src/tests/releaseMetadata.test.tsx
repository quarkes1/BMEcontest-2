import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import ModelPage from '../pages/ModelPage';
import { demoPrediction } from '../data/demo';

describe('release metadata generation and presentation', () => {
  it('labels official test performance and historical reproduction separately', () => {
    const html = renderToStaticMarkup(<ModelPage prediction={demoPrediction} release={{
      run_key: 'official_deployment_f1_policy_v3',
      official_metrics: { f1: .5333333333333333, ppv: .5714285714285714, sensitivity: .5, n_tp: 56, n_true: 112, n_pred: 98, fp: 42, fn: 56 },
      historical_reproduction: { run_key: '160afaf81debf1ee', outer_metrics: { f1: .6514285714285715, ppv: .5786802030456852, sensitivity: .7450980392156863, n_tp: 114, n_true: 153, n_pred: 197 } },
    }} />);
    expect(html).toContain('Official submission model');
    expect(html).toContain('Official test score');
    expect(html).toContain('TP / true events / predictions');
    expect(html).toContain('TP / eligible / predictions');
    expect(html).toContain('0.533');
    expect(html).toContain('Historical local reproduction');
    expect(html).toContain('0.651');
    expect(html).toContain('160afaf81debf1ee');
    expect(html).not.toContain('Subject-disjoint five-fold cross-validation development evidence. Not an independent test-set score.');
  });

  it('does not label registry fallback crossfit metrics as official test results', () => {
    const html = renderToStaticMarkup(<ModelPage prediction={demoPrediction} release={{
      run_key: '160afaf81debf1ee',
      outer_metrics: { f1: .6514285714285715, ppv: .5786802030456852, sensitivity: .7450980392156863, n_tp: 114, n_true: 153, n_pred: 197 },
    }} />);
    expect(html).toContain('Historical local reproduction');
    expect(html).toContain('0.651');
    expect(html).not.toContain('Official test score');
    expect(html).not.toContain('Official test F1');
  });
});
