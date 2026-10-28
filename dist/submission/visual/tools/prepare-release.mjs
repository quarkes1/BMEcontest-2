// 运行版本元数据只有一个来源：读取当前版本登记
// 及标准交叉拟合摘要，再生成前端使用的运行资源。
// 不手动重复填写版本指标。
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { resolve, join } from 'node:path';
import { existsSync } from 'node:fs';
const visual = resolve(import.meta.dirname, '..');
let root = visual;
while (!existsSync(join(root, 'release/event_stack_incumbent.json'))) {
  const parent = resolve(root, '..');
  if (parent === root) throw new Error('Release registry not found above visual/');
  root = parent;
}
const promoted = JSON.parse(await readFile(join(root, 'release/event_stack_incumbent.json'), 'utf8'));
const finalSubmissionPath = join(root, 'release/final_submission.json');
const finalSubmission = existsSync(finalSubmissionPath)
  ? JSON.parse(await readFile(finalSubmissionPath, 'utf8'))
  : null;
const historicalRunKey = finalSubmission?.reproduction_run_key ?? promoted.run_key;
const summary = JSON.parse(await readFile(join(root, `outputs/crossfit/summary_${historicalRunKey}.json`), 'utf8'));
const output = finalSubmission
  ? {
      run_key: finalSubmission.run_key,
      official_metrics: finalSubmission.official_metrics,
      historical_reproduction: { run_key: historicalRunKey, outer_metrics: summary.outer_metrics },
    }
  : { run_key: promoted.run_key, outer_metrics: summary.outer_metrics };
const target = join(visual, 'runtime/release-metadata.json');
await mkdir(resolve(target, '..'), { recursive: true });
await writeFile(target, JSON.stringify(output, null, 2) + '\n');
// 前端也通过生成模块导入同一份数据。
// 因此独立页面无需请求资源，可通过本地文件方式打开。
const modulePath = join(visual, 'app/src/runtime/release-data.generated.ts');
await writeFile(modulePath,
  '// 由 tools/prepare-release.mjs 根据最终提交元数据及交叉拟合摘要生成。\n' +
  '// 请通过 npm run build 或开发前置步骤重新生成，不直接修改此文件。\n' +
  'export type ReleaseMetrics = { f1: number; ppv: number; sensitivity: number; n_tp: number; n_true: number; n_pred: number; fp?: number; fn?: number };\n' +
  'export type ReleaseMetadata = {\n' +
  '  run_key: string;\n' +
  '  official_metrics?: ReleaseMetrics;\n' +
  '  historical_reproduction?: { run_key: string; outer_metrics: ReleaseMetrics };\n' +
  '  outer_metrics?: ReleaseMetrics;\n' +
  '};\n\n' +
  'export const RELEASE_METADATA: ReleaseMetadata = ' + JSON.stringify(output, null, 2) + ';\n');
console.log(`Prepared release metadata for ${output.run_key}`);
