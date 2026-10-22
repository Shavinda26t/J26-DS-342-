// Browser regression tests. Fixtures exist only here; the shipped UI uses the real API.
// Run with Vite on :5173 and NODE_PATH pointing to a local Playwright installation.
const { chromium } = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const models = ['TOSHIBA MG07ACA14TA', 'ST12000NM0008', 'ST4000DM000', 'ST8000NM0055', 'HGST HMS5C4040BLE640'];
const base = { end_date: '2021-09-30', mode: 'shadow', scorer_version: 'notebook24-sector-v1:' + 'a'.repeat(64), regime_id: null, drift_event: null };
const drives = [
  { ...base, serial_number: 'TEST-WARMING', model: models[0], status: 'warming', history_days: 12, anomaly_score: null, frozen_threshold: null, candidate_flag: null },
  { ...base, serial_number: 'TEST-SCORED', model: models[1], status: 'scored', history_days: 30, anomaly_score: 0.03, frozen_threshold: 0.07124821096658707, candidate_flag: false },
  { ...base, serial_number: 'TEST-CANDIDATE', model: models[2], status: 'scored', history_days: 30, anomaly_score: 3.1, frozen_threshold: 2.7843704223632812, candidate_flag: true },
  { ...base, serial_number: 'TEST-UNSUPPORTED', model: 'Other device', status: 'unsupported_model', history_days: 0, anomaly_score: null, frozen_threshold: null, candidate_flag: null },
];
(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const output = path.resolve('.venv/fleet-ui');
  fs.mkdirSync(output, { recursive: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
    const errors = [];
    page.on('pageerror', error => errors.push(String(error)));
    await page.addInitScript(() => {
      const exp = Date.now() + 3600000;
      sessionStorage.setItem('hddui.auth.session', JSON.stringify({ token: btoa(JSON.stringify({ sub: 'ui-test', exp })), user: { id: 'ui-test', name: 'UI Test' }, expiresAt: exp }));
    });
    let mode = 'loading';
    await page.route('**/api/v1/component1/shadow/history?**', async route => {
      const url = new URL(route.request().url());
      const serial = url.searchParams.get('serial_number');
      const drive = drives.find(item => item.serial_number === serial) || drives[0];
      const count = drive.status === 'unsupported_model' ? 0 : drive.history_days;
      const rows = Array.from({ length: count }, (_, index) => ({
        date: `2021-09-${String(30 - count + index + 1).padStart(2, '0')}`,
        smart_5_raw: index > 9 ? 1 : 0,
        smart_9_raw: 1000 + index * 24,
        smart_194_raw: 28 + (index % 3),
        smart_197_raw: 0,
        smart_198_raw: 0,
      }));
      return route.fulfill({ json: { serial_number: drive.serial_number, model: drive.model, start_date: rows[0]?.date ?? null, end_date: drive.end_date, history_days: rows.length, rows, result: drive } });
    });
    await page.route('**/api/v1/component1/shadow', async route => {
      if (mode === 'loading') await new Promise(resolve => setTimeout(resolve, 700));
      if (mode === 'error') return route.fulfill({ status: 503, contentType: 'application/json', body: JSON.stringify({ detail: 'Unavailable' }) });
      return route.fulfill({ json: { mode: 'shadow', supported_models: models, drives: mode === 'populated' ? drives : [], agent_schema_verified: false } });
    });
    await page.goto('http://127.0.0.1:5173/fleet');
    await page.getByText('Loading device records', { exact: true }).waitFor();
    assert.equal(await page.locator('.fleet-stat strong').first().innerText(), '\u2014');
    await page.getByText('No device records yet', { exact: true }).waitFor();
    assert.ok(await page.getByText('Waiting for live device data', { exact: true }).first().isVisible());
    assert.deepEqual(await page.locator('.fleet-stat strong').allTextContents(), ['0', '0', '0', '5']);
    assert.equal(await page.locator('.fleet-how li').count(), 4);
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.waitForTimeout(400);
    await page.screenshot({ path: path.join(output, 'desktop-empty.png'), fullPage: true });
    await page.getByRole('link', { name: /View integration requirements/ }).click();
    // The linked disclosure must be usable without leaving the dashboard.
    assert.ok(await page.getByText(/Required raw values:/).isVisible());
    mode = 'populated';
    await page.reload();
    await page.getByRole('row').filter({ hasText: 'TEST-WARMING' }).waitFor();
    assert.deepEqual(await page.locator('.fleet-stat strong').allTextContents(), ['4', '2', '1', '5']);
    assert.match(await page.getByRole('row').filter({ hasText: 'TEST-UNSUPPORTED' }).innerText(), /Unavailable.*Unsupported/s);
    await page.getByLabel('Select live device for visualization').selectOption('TEST-SCORED');
    await page.getByText('Power-on time and drive temperature', { exact: true }).waitFor();
    assert.equal(await page.locator('.fleet-live-charts .recharts-responsive-container').count(), 3);
    for (const label of ['Power-on hours', 'Drive temperature', 'Reallocated sectors', 'Pending sectors', 'Uncorrectable sectors']) {
      assert.ok(await page.getByText(label, { exact: true }).isVisible());
    }
    await page.locator('.fleet-live-charts .fleet-chart--score .recharts-bar-rectangle').last().hover();
    const tooltipText = page.locator('.fleet-live-charts .fleet-chart--score .recharts-default-tooltip');
    await tooltipText.waitFor();
    assert.ok((await tooltipText.locator('*').evaluateAll(elements => elements
      .filter(element => element.textContent.trim())
      .every(element => getComputedStyle(element).color === 'rgb(255, 255, 255)'))), 'Score tooltip text should be white');
    await page.getByLabel('Search drives', { exact: true }).fill('TEST-CANDIDATE');
    assert.equal(await page.locator('.fleet-table tbody tr').count(), 1);
    await page.getByLabel('Search drives', { exact: true }).fill('');
    await page.getByLabel('Filter by model').selectOption(models[0]);
    assert.equal(await page.locator('.fleet-table tbody tr').count(), 1);
    await page.getByLabel('Filter by model').selectOption('');
    await page.getByLabel('Filter by status').selectOption('candidate');
    assert.equal(await page.locator('.fleet-table tbody tr').count(), 1);
    await page.getByLabel('Search drives', { exact: true }).fill('no-such-device');
    await page.getByText('No matching devices', { exact: true }).waitFor();
    await page.getByRole('button', { name: 'Clear filters' }).click();
    await page.getByLabel('Scorer details for TEST-CANDIDATE', { exact: true }).click();
    assert.ok(await page.getByRole('row').filter({ hasText: 'TEST-CANDIDATE' }).getByText(base.scorer_version, { exact: true }).isVisible());
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.waitForTimeout(400);
    await page.screenshot({ path: path.join(output, 'desktop-populated.png'), fullPage: true });
    for (const width of [390, 768]) {
      await page.setViewportSize({ width, height: 844 });
      const overflow = await page.evaluate(() => ({
        documentWidth: document.documentElement.scrollWidth,
        viewportWidth: window.innerWidth,
        containers: ['.ui-dash__content', '.fleet', '.fleet-grid', '.fleet-monitor', '.fleet-table-scroll'].map(selector => {
          const element = document.querySelector(selector);
          const box = element?.getBoundingClientRect();
          return { selector, left: box?.left, right: box?.right, width: box?.width, scrollWidth: element?.scrollWidth };
        }),
        offenders: [...document.querySelectorAll('body *')]
          .filter(element => element.getBoundingClientRect().right > window.innerWidth + 1)
          .slice(0, 8)
          .map(element => `${element.tagName.toLowerCase()}.${element.className || ''}`),
      }));
      assert.ok(overflow.documentWidth <= overflow.viewportWidth, `Page overflow at ${width}: ${JSON.stringify(overflow)}`);
      await page.evaluate(() => window.scrollTo(0, 0));
    await page.waitForTimeout(400);
    await page.screenshot({ path: path.join(output, `populated-${width}.png`), fullPage: true });
    }
    mode = 'error';
    // Keep prior results during a failed polling refresh.
    await page.getByRole('alert').waitFor({ timeout: 15000 });
    assert.match(await page.getByRole('alert').innerText(), /may be stale/);
    await page.reload();
    await page.getByText('Device records unavailable', { exact: true }).waitFor();
    assert.equal(await page.locator('.fleet-stat strong').first().innerText(), '\u2014');
    mode = 'empty';
    await page.getByRole('button', { name: 'Retry', exact: true }).click();
    await page.getByText('No device records yet', { exact: true }).waitFor();
    await page.setViewportSize({ width: 390, height: 844 });
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.waitForTimeout(400);
    await page.screenshot({ path: path.join(output, 'mobile-empty.png'), fullPage: true });
    assert.deepEqual(errors, []);
    console.log('PASS: loading, empty, live counts, all row states, search, model/status filters, details, no matches, stale error, initial error, retry; desktop/mobile/tablet overflow checks');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
