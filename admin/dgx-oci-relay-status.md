# DGX OCI Relay Status

This note records the current state of the reverse SSH relay setup for DGX teammate access.

## Identified values

- OCI public IP: `129.158.50.228`
- OCI private/Tailscale IP: `100.90.51.70`
- OCI SSH admin user: `ubuntu`
- DGX admin user: `nathanaelguitar`
- OCI admin key on DGX: `/home/nathanaelguitar/.ssh/oci_key`
- Chosen relay port: `57325`

## OCI changes already applied

- Created low-privilege relay user: `dgxrelay`
- Installed a dedicated reverse-tunnel public key for `dgxrelay`
- Appended a scoped `Match User dgxrelay` block to `/etc/ssh/sshd_config`
- Validated `sshd_config` with `sshd -t`
- Restarted SSH successfully
- Verified `0.0.0.0:57325` can be opened by reverse tunnel
- Verified the SSH host keys on OCI `127.0.0.1:57325` match DGX `127.0.0.1:22`

## DGX facts confirmed

- Existing teammate account: `omp25`
- `omp25` groups: `omp25 ml-lora`
- Existing teammate access doc still points to old direct path:
  `ssh -p 33290 <username>@172.58.150.52`

## DGX changes now applied

- Saved relay port in `/root/dgx-access/relay_port.txt`
- Installed root-owned tunnel key in `/root/.ssh/`
- Installed and enabled `dgx-oci-reverse-ssh.service`
- Service is active and holding the reverse tunnel open
- Installed rollback instructions at `/root/dgx-access/ROLLBACK_DGX_OCI_RELAY.md`
- Replaced stale teammate access instructions in `/srv/dgxteam/README_ACCESS.md`
- Added `/srv/dgxteam/README_REMOTE_ACCESS.md`
- Confirmed local SSH policy:
  - `pubkeyauthentication yes`
  - `passwordauthentication no`
  - `kbdinteractiveauthentication no`
  - `permitrootlogin no`

## Remaining blocker

The reverse tunnel is healthy and OCI is listening on `0.0.0.0:57325`, but a direct connection to `129.158.50.228:57325` still times out from the DGX. The most likely remaining issue is the OCI VCN ingress rule for TCP `57325`.

OCI host-level firewall changes were not needed from this session, but the OCI VCN security rule still needs to allow inbound TCP `57325` to the VM if it is not already open.

Suggested temporary ingress rule:

- Source CIDR: `0.0.0.0/0`
- IP Protocol: TCP
- Destination Port Range: `57325`
- Description: `temporary DGX reverse SSH relay`

Tighten the source CIDR to teammate IPs if you know them.
