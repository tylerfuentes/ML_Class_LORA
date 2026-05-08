Send this to each teammate and ask them to reply with the three items filled in.

```text
I’m setting up DGX SSH access for the ML Class LoRA project.

Please send me:
1. Your Cornell NetID
2. Your full name
3. Your public SSH key

Your Cornell NetID will be used as your Linux username on the DGX.

Generate a key if you do not already have one:

Mac/Linux:
ssh-keygen -t ed25519 -C "netid@cornell.edu"

Windows PowerShell:
ssh-keygen -t ed25519 -C "netid@cornell.edu"

Then send me the contents of:
~/.ssh/id_ed25519.pub

It should look like:
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI... netid@cornell.edu

Do not send me your private key.

After your key is installed, your login command will look like:
ssh -o HostKeyAlias=spark-1b8a-relay -p 57325 <your-netid>@129.158.50.228

If the repo is private to the team, you can also add your Cornell NetID and public key to `docs/team-ssh-keys.md` and commit it there instead of sending it directly.
```
