/* Drive the built page in a real browser and fail on any thrown error.
 *
 *   NODE_PATH=~/.npm/_npx/<hash>/node_modules node tools/smoke.js [docs/index.html]
 *
 * Everything on this page that depends on the picker is computed in the browser,
 * and a handler that throws does not look broken - it looks like a page that has
 * stopped responding to one control. The bug that prompted this got shipped and
 * was found by a reader: selecting a memory size with no price on record threw
 * inside the picker handler, so the "What for?" selector silently stopped
 * reordering the table. Nothing was visibly wrong. Nothing in CI executed the
 * JavaScript at all.
 *
 * So this walks every chip, several memory sizes including the unpriced ones,
 * one and two units, and every use case, and treats a single pageerror as a
 * failure. It is slower than the other checks and needs a browser, which is why
 * it is a separate tool rather than part of build.py.
 */
const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');

const FILE = path.resolve(process.argv[2] || 'docs/index.html');

(async () => {
  if (!fs.existsSync(FILE)) {
    console.error(`no such file: ${FILE} - run tracker/build.py first`);
    process.exit(1);
  }
  // CI has Playwright's own chromium; a developer machine usually has Chrome
  // installed and no downloaded browser. Try the bundled one first so CI is the
  // predictable path, and fall back rather than making everyone download 150 MB.
  let browser;
  try {
    browser = await chromium.launch({ headless: true });
  } catch (e) {
    browser = await chromium.launch({ headless: true, channel: 'chrome' });
  }
  const failures = [];
  let checks = 0;
  try {
    const page = await browser.newPage();
    const errs = [];
    page.on('pageerror', e => errs.push(e.message.split('\n')[0]));
    page.on('console', m => { if (m.type() === 'error') errs.push('console: ' + m.text().slice(0, 140)); });

    await page.goto('file://' + FILE, { waitUntil: 'domcontentloaded' });
    await page.waitForTimeout(500);
    const chips = await page.$$eval('#rig-chip option', o => o.map(x => x.value).filter(Boolean));
    const ucs = await page.$$eval('#uc-sel .uc-chip', o => o.map(x => x.getAttribute('data-uc')));
    // The What-for control combines jobs: every single job, plus two combined
    // picks - a pair and a triple - so the AND logic rides through CI too.
    const ucSets = ucs.map(u => [u])
      .concat([['agentic', 'coding'], ['agentic', 'coding', 'longctx']]);
    // Press exactly the given set of job chips. Each chip click toggles its
    // pressed state and re-applies, so only the chips out of line get clicked.
    async function setUcs(ids) {
      const jobChips = await page.$$('#uc-sel .uc-chip');
      for (let i = 0; i < jobChips.length; i++) {
        const id = await jobChips[i].getAttribute('data-uc');
        const pressed = (await jobChips[i].getAttribute('aria-pressed')) === 'true';
        if (pressed !== ids.includes(id)) await jobChips[i].click();
      }
      await page.waitForTimeout(40);
    }

    for (const chip of chips) {
      await page.selectOption('#rig-chip', chip);
      await page.waitForTimeout(120);
      const mems = await page.$$eval('#rig-mem option', o => o.map(x => x.value).filter(Boolean));
      for (const mem of mems) {
        for (const n of ['1', '2']) {
          errs.length = 0;
          await page.selectOption('#rig-mem', mem);
          try { await page.selectOption('#rig-n', n); } catch (e) { /* single-unit chips */ }
          for (const sel of ucSets) {
            await setUcs(sel);
            checks++;
          }
          if (errs.length) {
            failures.push(`${chip} ${mem}GB x${n}: ${[...new Set(errs)].slice(0, 2).join(' | ')}`);
          }
        }
      }
    }
  // Each job's lane is a fill inside the track that sits in the row's .ix-id
  // cell (the name above it), and its width is a percentage of that track. So
  // anything making the track width vary per row makes the chart lie. It did:
  // on mobile the name shared a row with the status pill, and a long label like
  // "Fastest, with caveats" cut the cell from 225px to 128px, drawing a
  // SHORTER bar for a BETTER-ranked model.
  //
  // The fill itself is deliberately NOT monotonic down the list: rows follow
  // the category's curated ranking, and the fill is the model's share of a
  // perfect score on that category's own benchmark, which is not one scale -
  // agentic mixes several suites, image is editorial, longctx ranks on KV
  // cost. Only the track width is an invariant: 50% of one track must be 50%
  // of every track. Assert that, per category, per viewport.
  // reducedMotion matters - the lane fill has a 180ms width transition and
  // sampling mid-animation reports widths that are not real.
  const trackIssues = [];
  try {
    const page = await browser.newPage({ viewport: { width: 390, height: 900 },
                                         reducedMotion: 'reduce' });
    for (const vp of [{ w: 390, n: 'mobile' }, { w: 1280, n: 'desktop' }]) {
      await page.setViewportSize({ width: vp.w, height: 900 });
      await page.goto('file://' + FILE, { waitUntil: 'domcontentloaded' });
      await page.waitForTimeout(400);
      const ucs = await page.$$eval('#uc-sel .uc-chip', o => o.map(x => x.getAttribute('data-uc')));
      for (const uc of ucs) {
        const jobChips = await page.$$('#uc-sel .uc-chip');
        for (const c of jobChips) {
          const id = await c.getAttribute('data-uc');
          const pressed = (await c.getAttribute('aria-pressed')) === 'true';
          if (pressed !== (id === uc)) await c.click();
        }
        await page.waitForTimeout(120);
        const widths = await page.$$eval('.ix-row', rs => rs.filter(r => !r.hidden)
          .map(r => Math.round(r.querySelector('.ix-id').getBoundingClientRect().width * 2) / 2));
        const distinct = [...new Set(widths)];
        if (distinct.length > 1) {
          trackIssues.push(`${vp.n} ${uc}: lane track width varies between rows ` +
            `(${Math.min(...distinct)}px to ${Math.max(...distinct)}px)`);
        }
        checks++;
      }
    }
    await page.close();
  } catch (e) {
    failures.push('lane track check: ' + e.message.slice(0, 90));
  }
  if (trackIssues.length) failures.push(...[...new Set(trackIssues)].slice(0, 6));
  } finally {
    await browser.close();
  }

  console.log(`${checks} picker/use-case combinations exercised`);
  if (failures.length) {
    console.error(`\n${failures.length} configurations threw:`);
    [...new Set(failures)].slice(0, 20).forEach(f => console.error('  ' + f));
    process.exit(1);
  }
  console.log('no page errors');
})().catch(e => { console.error('ERR', e.message.slice(0, 200)); process.exit(1); });
