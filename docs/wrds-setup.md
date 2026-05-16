# WRDS Setup

This repo uses a local `~/.pgpass` file for WRDS credentials and keeps those secrets out of git.

## Credential file

Path:

- `/home/nathanaelguitar/.pgpass`

Required format:

```text
wrds-pgdata.wharton.upenn.edu:9737:wrds:YOUR_WRDS_USERNAME:YOUR_WRDS_PASSWORD
```

Field order:

1. host
2. port
3. database
4. WRDS username
5. WRDS password

Permissions must stay locked down:

```bash
chmod 600 ~/.pgpass
```

## Python environment

The repo-local virtual environment has the WRDS/Postgres client stack installed:

- `wrds`
- `psycopg2-binary`

Activate it with:

```bash
cd /home/nathanaelguitar/ML_Class_LORA
source .venv/bin/activate
```

## Minimal connection test

Run:

```bash
python scripts/test_wrds_connection.py
```

If credentials are correct, it should print:

```text
connected True
current_user <your_wrds_username>
current_database wrds
```

## Current blocker observed on this machine

The latest live test reached the WRDS server successfully but failed authentication:

```text
FATAL: PAM authentication failed for user "<your_wrds_username>"
```

That means:

- host and port are reachable
- SSL connection is working
- the remaining issue is username/password validity or WRDS account activation

In parallel, the repo now includes a browser-session fallback for WRDS web captures:

- `docs/wrds-playwright.md`
- `scripts/wrds_playwright_session.mjs`

## Next step after auth works

Once the connection test passes, the repo can add WRDS data pull scripts for:

- SEC / CRSP event studies
- Compustat fundamentals
- IBES revisions / surprise context
- any Cornell-licensed WRDS modules you confirm are available
