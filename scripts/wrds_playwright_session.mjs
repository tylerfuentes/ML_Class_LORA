#!/usr/bin/env node
// Playwright WRDS extractor — no screenshots, pure JS eval.
// Playwright manages the browser launch (avoids ARM64 sandbox issues with manual launch).

import fs from "fs";
import path from "path";
import { chromium } from "playwright";

const PROFILE_DIR = path.resolve(
  "/home/nathanaelguitar/ML_Class_LORA/admin/local/wrds-playwright-profile"
);
const OUTPUT_DIR = path.resolve(
  "/home/nathanaelguitar/ML_Class_LORA/admin/local/wrds-captures"
);
const LABEL = process.argv[2] || "wrds";
const START_URL = process.argv[3] || "https://wrds-www.wharton.upenn.edu/";
const PID_FILE = path.join(OUTPUT_DIR, "session.pid");

// Capture is triggered by SIGUSR1. Send: kill -USR1 $(cat admin/local/wrds-captures/session.pid)

async function extractPage(page) {
  return page.evaluate(() => {
    const tables = Array.from(document.querySelectorAll("table")).map((tbl, ti) => {
      const rows = Array.from(tbl.querySelectorAll("tr"));
      const headerCells = rows[0]?.querySelectorAll("th,td") || [];
      const headers = Array.from(headerCells).map((c) =>
        c.innerText.replace(/\s+/g, " ").trim()
      );
      const data = rows.slice(1).map((tr) =>
        Array.from(tr.querySelectorAll("td,th")).reduce((obj, td, i) => {
          obj[headers[i] || `col${i}`] = td.innerText.replace(/\s+/g, " ").trim();
          return obj;
        }, {})
      );
      return { tableIndex: ti, headers, rowCount: data.length, data };
    });

    const pre = Array.from(document.querySelectorAll("pre,code")).map((el) =>
      el.innerText.trim()
    ).filter(Boolean);

    const kvPairs = Array.from(document.querySelectorAll("dl")).map((dl) => {
      const obj = {};
      dl.querySelectorAll("dt").forEach((dt, i) => {
        obj[dt.innerText.trim()] = dl.querySelectorAll("dd")[i]?.innerText.trim() || "";
      });
      return obj;
    });

    return { url: location.href, title: document.title, tables, pre, kvPairs };
  });
}

async function main() {
  fs.mkdirSync(PROFILE_DIR, { recursive: true });
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });

  console.log("Launching browser...");
  const context = await chromium.launchPersistentContext(PROFILE_DIR, {
    headless: false,
    args: ["--no-sandbox", "--disable-setuid-sandbox"],
    viewport: { width: 1440, height: 900 },
  });

  const page = context.pages()[0] || (await context.newPage());
  await page.goto(START_URL, { waitUntil: "domcontentloaded", timeout: 30000 }).catch(() => {});

  fs.writeFileSync(PID_FILE, String(process.pid));
  console.log(`Browser open at: ${START_URL}`);
  console.log(`PID file:        ${PID_FILE}`);
  console.log(`Output dir:      ${OUTPUT_DIR}`);
  console.log(`Trigger capture: kill -USR1 ${process.pid}`);
  console.log(`Quit:            kill -USR2 ${process.pid}`);

  let capturing = false;
  let shouldQuit = false;

  async function doCapture() {
    if (capturing) return;
    capturing = true;
    try {
      // Prefer a WRDS tab; fall back to last tab
      const pages = context.pages();
      console.log("  Open tabs:", pages.map((p, i) => `[${i}] ${p.url()}`).join(", "));
      const target =
        pages.find((p) => p.url().includes("wrds")) ||
        pages.find((p) => !p.url().startsWith("chrome://")) ||
        pages[pages.length - 1];
      await target.waitForLoadState("networkidle").catch(() => {});
      const result = await extractPage(target);
      const stamp = new Date().toISOString().replace(/[:.]/g, "-");
      const outFile = path.join(OUTPUT_DIR, `${LABEL}-${stamp}.json`);
      fs.writeFileSync(outFile, JSON.stringify(result, null, 2));
      console.log(`\nCaptured: ${outFile}`);
      console.log(`  URL:   ${result.url}`);
      console.log(`  Title: ${result.title}`);
      result.tables.forEach((t, i) => {
        if (t.headers.length || t.rowCount)
          console.log(`  Table[${i}]: ${t.headers.slice(0, 5).join(" | ")} (${t.rowCount} rows)`);
      });
      if (result.pre.length) console.log(`  Pre blocks: ${result.pre.length}`);
    } finally {
      capturing = false;
    }
  }

  process.on("SIGUSR1", () => { doCapture().catch(console.error); });
  process.on("SIGUSR2", () => { shouldQuit = true; });

  // Keep alive until SIGUSR2
  await new Promise((resolve) => {
    const check = setInterval(() => { if (shouldQuit) { clearInterval(check); resolve(); } }, 500);
  });

  fs.rmSync(PID_FILE, { force: true });
  await context.close();
  console.log("Done.");
}

main().catch((e) => { console.error(e); process.exit(1); });
