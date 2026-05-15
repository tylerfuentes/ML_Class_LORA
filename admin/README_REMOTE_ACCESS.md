# DGX Remote Access

This file is the tracked template. Put machine-specific hostnames, relay ports, and one-off operational notes in `admin/local/README_REMOTE_ACCESS.local.md`.

Connect with:

```bash
ssh -o HostKeyAlias=<relay-host-alias> -p <relay-port> <your_dgx_username>@<relay-host>
```

First-time connect note: accept the DGX relay host key for your configured host alias if prompted.

VS Code Remote SSH config:

```sshconfig
Host dgx-class
    HostName <relay-host>
    Port <relay-port>
    User <your_dgx_username>
    HostKeyAlias <relay-host-alias>
    IdentityFile ~/.ssh/id_ed25519
```

Work inside:

```text
<shared-workspace-path>
```

Rules:
- No sudo.
- No Docker access.
- Do not store secrets in the repo.
- Put configs in `/srv/dgxteam/configs`.
- Put datasets in `/srv/dgxteam/datasets`.
- Put outputs in `/srv/dgxteam/outputs`.
- Ask before installing packages or changing containers.
