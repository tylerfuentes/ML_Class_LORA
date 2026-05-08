# SSH Onboarding

This project is set up to use separate Linux accounts per teammate. That is safer than sharing one Unix account and works cleanly with VS Code Remote SSH.

## Inputs you still need

For each teammate:

- Full name
- Cornell NetID
- One public SSH key line from their `.pub` file

Username rule:

- use each teammate's Cornell NetID as their Linux username on the DGX
- keep the username lowercase

## One-time repo configuration

Run this from the repository root:

```bash
sudo ./scripts/configure_shared_repo.sh
```

This creates:

- shared Unix group: `ml-lora`
- group-writable repo permissions
- setgid on directories so new files inherit the project group

## Per-teammate provisioning

Run once per teammate:

```bash
sudo ./scripts/provision_teammate.sh \
  --username abc123 \
  --full-name "Finn Kliewer" \
  --public-key "ssh-ed25519 AAAA... finn@laptop"
```

The script:

- creates the Linux user if needed
- creates `~/.ssh`
- installs the provided public key into `authorized_keys`
- adds the user to the `ml-lora` group
- grants the repo group access

## What to send each teammate

Give each teammate:

- relay host: `129.158.50.228`
- relay port: `57325`
- their Linux username

Their VS Code Remote SSH target will look like:

```bash
ssh -p 57325 <username>@129.158.50.228
```

You can also send this SSH config block:

```sshconfig
Host dgx-class
    HostName 129.158.50.228
    Port 57325
    User <username>
    IdentityFile ~/.ssh/id_ed25519
```

## Relay notes

- Teammates do not get OCI shell access.
- The relay port terminates on the DGX SSH daemon through a reverse tunnel.
- Keep teammate accounts out of `sudo` and `docker`.

## Notes

- Do not collect private keys.
- Do not put multiple teammates on one shared Unix login unless you accept losing accountability and auditability.
- If the repo is moved, rerun `configure_shared_repo.sh` from the new repo path.
