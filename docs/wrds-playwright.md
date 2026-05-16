# WRDS Playwright Fallback

This is the fallback path when WRDS web access works but direct PostgreSQL / Python authentication is not yet provisioned.

## When to use this

Use this workflow if:

- the Cornell / WRDS day-pass web login works
- you can reach the WRDS interface in the browser
- but `scripts/test_wrds_connection.py` still fails with database authentication errors

That is the current state on this DGX.

## What the script does

`scripts/wrds_playwright_session.mjs` launches Chromium with a persistent local profile, so your Cornell / WRDS login can be reused across runs.

It then lets you:

1. log in manually in the browser window
2. navigate to the WRDS page or result table you want
3. press Enter in the terminal
4. save the current page as:
   - HTML
   - extracted tables JSON
   - screenshot
   - metadata JSON

The script does not bypass authentication. It only automates capture after you authenticate interactively.

## Local profile and capture storage

These paths are under `admin/local/` and are ignored by git:

- browser profile:
  - `admin/local/wrds-playwright-profile/`
- captures:
  - `admin/local/wrds-captures/`

## Run it

```bash
cd /home/nathanaelguitar/ML_Class_LORA
node scripts/wrds_playwright_session.mjs
```

Default start page:

- `https://johnson.library.cornell.edu/database/wharton-research-data-services-wrds/`

## Useful options

Open a specific URL:

```bash
node scripts/wrds_playwright_session.mjs \
  --url "https://wrds-www-wharton-upenn-edu.proxy.library.cornell.edu/users/your-account/"
```

Use a custom capture label:

```bash
node scripts/wrds_playwright_session.mjs \
  --capture-name "wrds-ibes-query"
```

## Output files

For each capture, the script writes:

- `*.meta.json`
- `*.html`
- `*.tables.json`
- `*.png`

The `tables.json` file is the most useful starting point for turning WRDS web query results into structured data.

## Current limitation

This is a web-session workaround, not a replacement for real WRDS programmatic credentials.

The preferred end state is still:

- successful `~/.pgpass` auth
- direct `wrds` / `psycopg2` queries from Python

See also:

- `docs/wrds-setup.md`
