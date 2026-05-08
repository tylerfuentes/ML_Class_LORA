# Rollback DGX OCI Relay

## On DGX

```bash
sudo systemctl stop dgx-oci-reverse-ssh.service
sudo systemctl disable dgx-oci-reverse-ssh.service
sudo rm -f /etc/systemd/system/dgx-oci-reverse-ssh.service
sudo systemctl daemon-reload
sudo rm -f /root/.ssh/dgx_oci_relay_ed25519 /root/.ssh/dgx_oci_relay_ed25519.pub
sudo rm -f /root/dgx-access/relay_port.txt
```

## On OCI

```bash
sudo deluser --remove-home dgxrelay
sudoedit /etc/ssh/sshd_config
sudo sshd -t
sudo systemctl restart ssh || sudo systemctl restart sshd
```

Remove the `Match User dgxrelay` block from `sshd_config`, and remove any OCI ingress rule that exposes TCP `57325`.
