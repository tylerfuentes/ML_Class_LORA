# Local Admin Overrides

Machine-specific SSH and relay state should live outside tracked files.

Use ignored paths for values that only make sense on one DGX or one admin machine:

- `admin/local/README_REMOTE_ACCESS.local.md`
- `admin/local/relay.env`
- `admin/local/sshd/*.conf`
- `docs/local/*.md`

Recommended pattern:

1. Keep tracked docs generic and reusable.
2. Copy a tracked template into `admin/local/` when you need DGX-specific values.
3. Keep machine-specific IPs, ports, tunnel notes, and hardening fragments in `admin/local/`.

Example `admin/local/relay.env`:

```bash
RELAY_HOST=129.158.50.228
RELAY_PORT=57325
SSH_HOST_ALIAS=spark-1b8a-relay
WORKSPACE_ROOT=/srv/dgxteam
```

These files are ignored by git through the repo `.gitignore`.
