// file:// end-to-end smoke: the standalone page must render without a server.
// Uses the system Edge via playwright-core (no browser download).
import { chromium } from 'playwright-core';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const visual = resolve(import.meta.dirname, '../..');
const url = pathToFileURL(resolve(visual, 'index.html')).href;
let browser;
for (const channel of ['msedge', 'chrome']) {
  try { browser = await chromium.launch({ channel, headless: true }); break; }
  catch (error) { if (channel === 'chrome') throw error; }
}
const page = await browser.newPage();
const errors = [];
page.on('pageerror', error => errors.push(String(error)));
page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
try {
  await page.goto(url);
  try {
    await page.waitForSelector('.timeline-panel', { timeout: 15000 });
  } catch (error) {
    const html = await page.evaluate(() => document.body.innerHTML.slice(0, 400));
    console.error('RENDER FAILED. page errors:', errors);
    console.error('body snapshot:', html);
    throw error;
  }
  const brand = await page.textContent('.brand strong');
  if (brand !== 'EatingSense') throw new Error(`brand mismatch: ${brand}`);
  const badge = await page.textContent('.top-meta');
  if (!badge.includes('DEMO DATA')) throw new Error('demo badge missing');
  const canvasCount = await page.locator('canvas').count();
  if (canvasCount < 2) throw new Error(`expected timeline + motion canvases, found ${canvasCount}`);
  if (await page.locator('.motion-approximate').count()) throw new Error('fully calibrated demo should not show Approximate');
  const before = await Promise.all(['.monitor-workspace', '.status-header', '.monitor-grid', '.detail-grid'].map(selector => page.locator(selector).boundingBox()));
  await page.locator('input[accept=".json,.bin"]').setInputFiles({ name: 'broken.json', mimeType: 'application/json', buffer: Buffer.from('{') });
  await page.waitForSelector('.error-banner');
  const after = await Promise.all(['.monitor-workspace', '.status-header', '.monitor-grid', '.detail-grid'].map(selector => page.locator(selector).boundingBox()));
  if (JSON.stringify(before) !== JSON.stringify(after)) throw new Error('Monitor geometry changed when an error was shown');
  await page.click('.error-banner button');
  await page.click('nav button:has-text("Events")');
  await page.waitForSelector('.event-record');
  const events = await page.locator('.event-record').count();
  if (events < 3) throw new Error(`expected >=3 demo events, found ${events}`);
  await page.click('.event-record >> nth=0');
  await page.waitForSelector('.timeline-panel');
  await page.click('nav button:has-text("Model")');
  await page.waitForSelector('.model-page');
  if (errors.length) throw new Error('page errors: ' + errors.join(' | '));
  console.log(`e2e smoke OK: demo renders from file://, ${events} events listed, no page errors`);
} finally {
  await browser.close();
}
