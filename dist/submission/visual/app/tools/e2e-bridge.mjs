// 端到端检查启动路径：由本地推理服务提供页面，
// 真实 collect_data*.txt 经统一 Predictor 推理后进入界面。
//
// 启动 dist/inference/serve.py，需要对应 Python 环境，
// 通过本机 HTTP 页面向隐藏 TXT 输入框载入真实会话。
// 确认状态就绪、会话出现且事件正常渲染，
// 同时要求页面没有运行错误。
import { spawn } from 'node:child_process';
import { readdir, readFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { chromium } from 'playwright-core';

const app = resolve(import.meta.dirname, '..');
const visual = resolve(app, '..');
const dist = resolve(visual, '..');
const submission = existsSync(resolve(dist, 'build_manifest.json'));
const repo = submission ? dist : resolve(dist, '..');
const PORT = 4199;
const python = process.env.BME_PYTHON || 'python';

// 使用可检出至少一个事件的真实记录会话，
// 对应外层第一折验证受试者，使端到端断言具有实际意义。
const session_id = process.env.BME_E2E_SESSION || 'sensorData-1784562356253-23edff73a65e4e9583c963c5a6ee26dd';
const dataDir = process.env.BME_E2E_DATA_ROOT || resolve(repo, 'Data');
const batchDirs = await readdir(dataDir);
const sessionRoot = batchDirs.map(name => resolve(dataDir, name, session_id)).find(candidate => existsSync(candidate));
if (!sessionRoot) throw new Error(`session fixture not found under ${dataDir}/*/${session_id}`);
const sessionFile = (await readdir(sessionRoot)).find(name => /^collect_data.*\.txt$/.test(name));
if (!sessionFile) throw new Error(`no collect_data*.txt in ${sessionRoot}`);
const sessionPath = resolve(sessionRoot, sessionFile);

const serverArguments = submission
  ? [resolve(dist, 'serve.py'), '--port', String(PORT)]
  : [resolve(dist, 'inference', 'serve.py'), '--port', String(PORT), '--visual-dir', resolve(visual)];
const server = spawn(python, serverArguments,
  { cwd: repo, stdio: ['ignore', 'pipe', 'pipe'] });
server.stderr.on('data', chunk => process.stderr.write(String(chunk)));
let serverExited = false;
server.on('exit', () => { serverExited = true; });

async function waitForHealth(deadlineMs = 30000) {
  const started = Date.now();
  while (Date.now() - started < deadlineMs) {
    if (serverExited) throw new Error('inference bridge exited before becoming ready');
    try {
      const response = await fetch(`http://127.0.0.1:${PORT}/api/health`);
      if (response.ok && (await response.json()).status === 'ok') return;
    } catch { /** 服务尚未就绪。 */ }
    await new Promise(resolveWait => setTimeout(resolveWait, 500));
  }
  throw new Error('inference bridge did not become ready in time');
}

let browser;
try {
  await waitForHealth();
  browser = await chromium.launch({ channel: 'msedge', headless: true }).catch(() => chromium.launch({ channel: 'chrome', headless: true }));
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(String(error)));
  page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
  await page.goto(`http://127.0.0.1:${PORT}/`);
  await page.waitForSelector('.timeline-panel');
  const txtButton = page.locator('button:has-text("Select TXT files")');
  if (await txtButton.isDisabled()) throw new Error('TXT selection should be enabled while the bridge is online');
  await page.setInputFiles('.data-loader input[accept=".txt,text/plain"]', sessionPath);
  // 分析期间时间轴应明确显示分析状态，并隐藏旧曲线。
  await page.waitForSelector('.timeline-analyzing', { timeout: 30000 });
  const analyzingText = (await page.locator('.timeline-analyzing strong').innerText()).trim();
  if (analyzingText !== 'Analyzing…') throw new Error(`unexpected analyzing overlay text: ${analyzingText}`);
  if ((await page.locator('.status-line.busy').count()) !== 1) throw new Error('status toast should show the busy state while analyzing');
  await page.waitForSelector('.status-line', { timeout: 30000 });
  await page.waitForFunction(() => document.querySelector('.status-line')?.textContent?.includes('Ready'), null, { timeout: 180000 });
  if (await page.locator('.timeline-analyzing').count()) throw new Error('analyzing overlay must disappear once the timeline is ready');
  const approximate = await page.locator('.motion-approximate').count();
  if (approximate !== 1) throw new Error('real raw telemetry should show the Approximate calibration badge');
  const sessionValue = await page.inputValue('.session-picker select');
  if (!sessionValue.includes(sessionFile.replace(/\.txt$/, ''))) throw new Error(`unexpected session id: ${sessionValue}`);
  await page.click('nav button:has-text("Events")');
  await page.waitForSelector('.event-record');
  const events = await page.locator('.event-record').count();
  if (events < 1) throw new Error(`expected >=1 canonical event, found ${events}`);
  if (errors.length) throw new Error('page errors: ' + errors.join(' | '));
  console.log(`e2e bridge OK: TXT -> canonical Predictor -> UI rendered ${events} event(s) for ${sessionValue}, no page errors`);
} finally {
  if (browser) await browser.close();
  if (!serverExited) server.kill();
}
