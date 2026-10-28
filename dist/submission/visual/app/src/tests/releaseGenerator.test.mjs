import { execFile } from 'node:child_process';
import { mkdtemp, mkdir, readFile, rm, writeFile, copyFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';
import { describe, expect, it } from 'vitest';

const execFileAsync = promisify(execFile);
const generator = fileURLToPath(new URL('../../../tools/prepare-release.mjs', import.meta.url));

async function fixture(finalSubmission) {
  const directory = await mkdtemp(join(tmpdir(), 'visual-release-'));
  const visual = join(directory, 'dist', 'visual');
  await mkdir(join(visual, 'app', 'src', 'runtime'), { recursive: true });
  await mkdir(join(visual, 'tools'), { recursive: true });
  await copyFile(generator, join(visual, 'tools', 'prepare-release.mjs'));
  await mkdir(join(directory, 'release'), { recursive: true });
  await mkdir(join(directory, 'outputs', 'crossfit'), { recursive: true });
  await writeFile(join(directory, 'release', 'event_stack_incumbent.json'), JSON.stringify({ run_key: '160afaf81debf1ee' }));
  await writeFile(join(directory, 'outputs', 'crossfit', 'summary_160afaf81debf1ee.json'), JSON.stringify({ outer_metrics: { f1: .6514285714285715, ppv: .5786802030456852, sensitivity: .7450980392156863, n_tp: 114, n_true: 153, n_pred: 197 } }));
  if (finalSubmission !== undefined) await writeFile(join(directory, 'release', 'final_submission.json'), JSON.stringify(finalSubmission));
  const { stdout } = await execFileAsync(process.execPath, [join(visual, 'tools', 'prepare-release.mjs')]);
  return { directory, visual, stdout };
}

describe('release metadata generation', () => {
  it('preserves incumbent-registry behavior when final submission metadata is absent', async () => {
    const { directory, visual } = await fixture();
    try {
      const metadata = JSON.parse(await readFile(join(visual, 'runtime', 'release-metadata.json'), 'utf8'));
      expect(metadata).toEqual({
        run_key: '160afaf81debf1ee',
        outer_metrics: { f1: .6514285714285715, ppv: .5786802030456852, sensitivity: .7450980392156863, n_tp: 114, n_true: 153, n_pred: 197 },
      });
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });

  it('keeps official test metrics separate from historical crossfit reproduction metrics', async () => {
    const officialMetrics = { f1: .5333333333333333, ppv: .5714285714285714, sensitivity: .5, n_tp: 56, n_true: 112, n_pred: 98, fp: 42, fn: 56 };
    const { directory, visual, stdout } = await fixture({ run_key: 'official_deployment_f1_policy_v3', official_metrics: officialMetrics, reproduction_run_key: '160afaf81debf1ee' });
    try {
      const metadata = JSON.parse(await readFile(join(visual, 'runtime', 'release-metadata.json'), 'utf8'));
      expect(metadata).toEqual({
        run_key: 'official_deployment_f1_policy_v3',
        official_metrics: officialMetrics,
        historical_reproduction: { run_key: '160afaf81debf1ee', outer_metrics: { f1: .6514285714285715, ppv: .5786802030456852, sensitivity: .7450980392156863, n_tp: 114, n_true: 153, n_pred: 197 } },
      });
      const generatedModule = await readFile(join(visual, 'app', 'src', 'runtime', 'release-data.generated.ts'), 'utf8');
      expect(generatedModule).toContain('historical_reproduction');
      expect(generatedModule).toContain('RELEASE_METADATA: ReleaseMetadata');
      expect(stdout).toContain('official_deployment_f1_policy_v3');
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });
});
