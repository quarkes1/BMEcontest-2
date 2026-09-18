// End-to-end: launcher path — the local inference bridge serves the page and a real
// collect_data*.txt selection flows through the canonical Predictor into the UI.
//
// Spawns dist/inference/serve.py (needs the Python environment), opens the page over
// http://127.0.0.1, feeds a REAL session file through the hidden TXT input, and
// asserts the status line reaches Ready, the session appears, and events render —
// with zero page errors.
import { spawn } from 'node:child_process';
import { readdir, readFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { chromium } from 'playwright-core';

const app = resolve(import.meta.dirname, '..');
const visual = resolve(app, '..');
const dist = resolve(visual, '..');
const repo = resolve(dist, '..');
const PORT = 4199;
const python = process.env.BME_PYTHON || 'python';

// A real recorded session that the promoted release detects at least one event in
// (fold-1 validation subject); keeps the end-to-end assertion meaningful.
const session_id = process.env.BME_E2E_SESSION || 'sensorData-1784562356253-23edff73a65e4e9583c963c5a6ee26dd';
const dataDir = resolve(repo, 'Data');
const batchDirs = await readdir(dataDir);
const sessionRoot = batchDirs.map(name => resolve(dataDir, name, session_id)).find(candidate => existsSync(candidate));
if (!sessionRoot) throw new Error(`session fixture not found under ${dataDir}/*/${session_id}`);
const sessionFile = (await readdir(sessionRoot)).find(name => /^collect_data.*\.txt$/.test(name));
if (!sessionFile) throw new Error(`no collect_data*.txt in ${sessionRoot}`);
const sessionPath = resolve(sessionRoot, sessionFile);

const server = spawn(python, [resolve(dist, 'inference', 'serve.py'), '--port', String(PORT), '--visual-dir', resolve(visual)],
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
    } catch { /* not up yet */ }
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
  await page.waitForSelector('.status-line', { timeout: 30000 });
  await page.waitForFunction(() => document.querySelector('.status-line')?.textContent?.includes('Ready'), null, { timeout: 180000 });
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
