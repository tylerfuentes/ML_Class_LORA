# DGX Team Access

SSH command:

ssh -p 57325 <username>@129.158.50.228

Use your assigned Linux username and private SSH key.

VS Code Remote SSH config:

```sshconfig
Host dgx-class
    HostName 129.158.50.228
    Port 57325
    User <username>
    IdentityFile ~/.ssh/id_ed25519
```

Rules:
- Do not store secrets in the repo.
- Work inside /srv/dgxteam.
- Do not write large outputs to home directories.
- Put datasets in /srv/dgxteam/datasets.
- Put configs in /srv/dgxteam/configs.
- Put training outputs in /srv/dgxteam/outputs.
- You do not have sudo.
- You do not have Docker access.
- Container runs are controlled by the owner/admin.
