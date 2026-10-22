// Full browser -> API -> isolated rolling state -> frozen scorer replay test.
const { chromium } = require('playwright');
const assert = require('node:assert/strict');
const path = require('node:path');

const ui = process.env.REPLAY_UI_URL || 'http://127.0.0.1:5181';

function syntheticCsv() {
  const header = 'serial_number,model,date,smart_5_raw,smart_9_raw,smart_194_raw,smart_197_raw,smart_198_raw';
  const rows = [];
  for (let day = 1; day <= 30; day += 1) {
    rows.push(`SYNTHETIC-BACKBLAZE,ST4000DM000,2022-10-${String(day).padStart(2, '0')},0,${1000 + day * 24},28,0,0`);
  }
  return `${header}\n${rows.join('\n')}\n`;
}

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    const pageErrors = [];
    page.on('pageerror', error => pageErrors.push(String(error)));
    await page.addInitScript(() => {
      const exp = Date.now() + 3600000;
      sessionStorage.setItem('hddui.auth.session', JSON.stringify({
        token: btoa(JSON.stringify({ sub: 'replay-ui-test', exp })),
        user: { id: 'replay-ui-test', name: 'Replay UI Test' },
        expiresAt: exp,
      }));
      localStorage.removeItem('hdd.backblaze.replay');
    });
    await page.goto(`${ui}/fleet`);
    await page.getByText('Replay Backblaze data', { exact: true }).click();
    await page.getByLabel('Select extracted CSV').setInputFiles({
      name: 'synthetic_backblaze_window.csv',
      mimeType: 'text/csv',
      buffer: Buffer.from(syntheticCsv()),
    });
    await page.getByText('SYNTHETIC-BACKBLAZE', { exact: true }).first().waitFor();
    assert.equal(await page.locator('.fleet-replay-preview tbody tr').count(), 30);
    assert.match(await page.locator('.fleet-replay-id').innerText(), /REPLAY_[a-f0-9]{32}/);
    assert.equal(await page.getByRole('tab', { name: 'Historical replay' }).getAttribute('aria-selected'), 'true');
    assert.ok(await page.getByText('Power-on time and drive temperature', { exact: true }).isVisible());
    assert.ok(await page.getByText('Drive sector health', { exact: true }).isVisible());
    for (const label of ['Power-on hours', 'Drive temperature', 'Reallocated sectors', 'Pending sectors', 'Uncorrectable sectors']) {
      assert.ok(await page.getByText(label, { exact: true }).isVisible());
    }
    assert.equal(await page.locator('.fleet-replay-monitor .recharts-responsive-container').count(), 2);
    await page.getByRole('button', { name: 'Run replay', exact: true }).click();
    await page.getByText('Day 30 frozen response', { exact: true }).waitFor({ timeout: 60000 });
    assert.match(await page.getByRole('status').innerText(), /30\/30 daily records accepted - Scored/);
    assert.equal(await page.locator('.fleet-replay-result').getByText('No', { exact: true }).count(), 1);
    assert.ok(await page.locator('.fleet-replay-result').getByText('2022-10-30', { exact: true }).isVisible());
    assert.ok((await page.locator('.fleet-replay-result').innerText()).includes('not a failure probability'));
    assert.equal(await page.locator('.fleet-replay-monitor .recharts-responsive-container').count(), 3);
    assert.ok(await page.getByText('Research candidate flag: No', { exact: false }).isVisible());
    await page.getByRole('tab', { name: 'Live devices' }).click();
    assert.equal(await page.locator('.fleet-monitor .fleet-table tbody tr').count(), 0, 'replay leaked into live-device table');
    await page.getByRole('tab', { name: 'Historical replay' }).click();
    await page.screenshot({ path: path.resolve('.venv/backblaze-replay-desktop.png'), fullPage: true });
    await page.setViewportSize({ width: 390, height: 844 });
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth));
    await page.screenshot({ path: path.resolve('.venv/backblaze-replay-mobile.png'), fullPage: true });
    assert.deepEqual(pageErrors, []);
    console.log('PASS: CSV preview -> isolated 30-day replay -> actual day-30 frozen result; live table unchanged; mobile layout fits');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
