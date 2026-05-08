# DGX Remote Access

Connect with:

```bash
ssh -p 57325 <your_dgx_username>@129.158.50.228
```

VS Code Remote SSH config:

```sshconfig
Host dgx-class
    HostName 129.158.50.228
    Port 57325
    User <your_dgx_username>
    IdentityFile ~/.ssh/id_ed25519
```

Work inside:

```text
/srv/dgxteam
```

Rules:
- No sudo.
- No Docker access.
- Do not store secrets in the repo.
- Put configs in `/srv/dgxteam/configs`.
- Put datasets in `/srv/dgxteam/datasets`.
- Put outputs in `/srv/dgxteam/outputs`.
- Ask before installing packages or changing containers.
