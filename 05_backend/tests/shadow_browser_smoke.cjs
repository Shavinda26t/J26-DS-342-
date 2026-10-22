// Manual test: isolated backend on 8000, Vite on 5173, Playwright + Edge installed.
const { chromium, request } = require('playwright');
const assert = require('node:assert/strict');

(async () => {
  const serial = `shadow-browser-${Date.now()}`;
  const api = await request.newContext({ baseURL: process.env.SHADOW_API_URL || 'http://127.0.0.1:8000' });
  const submit = async (n) => {
    const date = new Date(Date.UTC(2021, 8, 1 + n)).toISOString().slice(0, 10);
    const response = await api.post('/api/v1/component1/shadow/records', { data: {
      serial_number: serial, model: 'ST4000DM000', date,
      smart_5_raw: 0, smart_9_raw: 1000 + n * 24, smart_194_raw: 28,
      smart_197_raw: 0, smart_198_raw: 0,
    }});
    assert.equal(response.status(), 200);
    return response.json();
  };
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', error => errors.push(String(error)));
    await page.goto(`${process.env.SHADOW_UI_URL || 'http://127.0.0.1:5173'}/fleet`);
    await page.getByLabel('Email', { exact: true }).fill('researcher@hdd-lab.edu');
    await page.getByLabel('Password', { exact: true }).fill('Demo@123');
    await page.getByRole('button', { name: 'Sign in', exact: true }).click();
    await page.waitForURL('**/fleet');
    await submit(0);
    await page.reload();
    const row = page.getByRole('row').filter({ hasText: serial });
    await row.waitFor();
    assert.match(await row.innerText(), /Warming/);
    assert.match(await row.innerText(), /1\/30/);
    let result;
    for (let n = 1; n < 30; n++) result = await submit(n);
    await page.reload();
    await row.waitFor();
    const displayed = await row.innerText();
    assert.match(displayed, /Scored/);
    assert.ok(displayed.includes(String(result.anomaly_score)));
    assert.ok(displayed.includes(String(result.frozen_threshold)));
    await page.getByLabel(`Scorer details for ${serial}`, { exact: true }).click();
    assert.ok((await row.innerText()).includes(result.scorer_version));
    assert.ok((await page.locator('body').innerText()).includes('Research anomaly candidate only \u00b7 Not a failure probability.'));
    await submit(31);
    await page.reload();
    await row.waitFor();
    assert.match(await row.innerText(), /Gap reset/);
    assert.match(await row.innerText(), /1\/30/);
    assert.deepEqual(errors, []);
    console.log('PASS: real API -> browser warming -> day 30 score/threshold/version -> gap reset; no page errors');
  } finally {
    await browser.close();
    await api.dispose();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
