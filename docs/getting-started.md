# Getting Started

This document is for teammates joining the `ML_Class_LORA` project.

The project uses:

- GitHub for source control
- SSH for access to the DGX Spark
- VS Code Remote SSH for development on the machine

## What you need to do first

Before you can log into the DGX, you need your own SSH keypair on your own laptop or desktop.

Do not send anyone your private key.

## Step 1: Generate an SSH key on your machine

### Mac or Linux

Run:

```bash
ssh-keygen -t ed25519 -C "netid@cornell.edu"
```

Press Enter to accept the default file location unless you already know you want something different.

### Windows PowerShell

Run:

```powershell
ssh-keygen -t ed25519 -C "netid@cornell.edu"
```

Press Enter to accept the default file location unless you already know you want something different.

## Step 2: Send your public key

After generating the key, send the contents of your public key file to the project admin.

You can do this in either of two ways:

- send the key directly to the project admin
- add it to `docs/team-ssh-keys.md` in this repository and commit it

Only do the repository option if this GitHub repository is private to the team.

### Mac or Linux

```bash
cat ~/.ssh/id_ed25519.pub
```

### Windows PowerShell

```powershell
type $env:USERPROFILE\.ssh\id_ed25519.pub
```

Your public key will look something like:

```text
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI... netid@cornell.edu
```

Send that full line exactly as-is.

Do not send:

- `id_ed25519`
- any file without `.pub`
- screenshots of the key

## Step 2A: Submit your key through the repository

If the team is collecting keys through GitHub, open `docs/team-ssh-keys.md`, fill in your section, and commit the change.

Example workflow:

```bash
git clone https://github.com/nathanaelguitar/ML_Class_LORA.git
cd ML_Class_LORA
git checkout -b add-my-ssh-key
```

Edit `docs/team-ssh-keys.md`, then commit and push:

```bash
git add docs/team-ssh-keys.md
git commit -m "Add SSH key for <your-netid>"
git push origin add-my-ssh-key
```

Then open a pull request or coordinate with the team on how changes are merged.

## Step 3: Wait for your account details

Once your public key is installed on the DGX, you will receive:

- confirmation that your Cornell NetID is your Linux username
- the DGX hostname or public IP

Then your SSH command will look like:

```bash
ssh <your-netid>@<dgx-host-or-ip>
```

## Step 4: Test SSH access

After your account is provisioned, test login from a terminal.

```bash
ssh <your-netid>@<dgx-host-or-ip>
```

If this works, your machine is ready for VS Code Remote SSH.

## Step 5: Use VS Code Remote SSH

Install these in VS Code:

- Remote Development
- Remote - SSH

Then:

1. Open VS Code.
2. Open the Command Palette.
3. Run `Remote-SSH: Connect to Host`.
4. Enter:

```text
<your-netid>@<dgx-host-or-ip>
```

5. Open the project directory on the DGX after login.

## GitHub workflow

Clone the repository after you have access or work with it directly through VS Code on the DGX.

Basic workflow:

```bash
git clone https://github.com/nathanaelguitar/ML_Class_LORA.git
cd ML_Class_LORA
git checkout -b your-branch-name
```

Commit and push your work through your own branch unless the team agrees on a different flow.

## What this project is about

We are fine-tuning a strong base model with LoRA for finance-related tasks.

The goal is to improve performance on structured financial inputs and produce outputs that are easier to evaluate, compare, and explain. We will compare the fine-tuned model against the base model using simple metrics and practical examples, then frame the results as a business use case.

## Current status

Right now, the immediate blocker is collecting each teammate's public SSH key so DGX accounts can be provisioned.
