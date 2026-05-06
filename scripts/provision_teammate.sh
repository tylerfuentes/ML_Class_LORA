#!/usr/bin/env bash
set -euo pipefail

GROUP_NAME="ml-lora"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  provision_teammate.sh --username USERNAME --full-name "Full Name" --public-key "ssh-ed25519 AAAA... comment"
EOF
}

USERNAME=""
FULL_NAME=""
PUBLIC_KEY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --username)
      USERNAME="${2:-}"
      shift 2
      ;;
    --full-name)
      FULL_NAME="${2:-}"
      shift 2
      ;;
    --public-key)
      PUBLIC_KEY="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${USERNAME}" || -z "${FULL_NAME}" || -z "${PUBLIC_KEY}" ]]; then
  usage >&2
  exit 1
fi

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run as root or with sudo." >&2
  exit 1
fi

if ! getent group "${GROUP_NAME}" >/dev/null; then
  groupadd "${GROUP_NAME}"
fi

if ! id "${USERNAME}" >/dev/null 2>&1; then
  useradd -m -s /bin/bash -c "${FULL_NAME}" "${USERNAME}"
fi

usermod -aG "${GROUP_NAME}" "${USERNAME}"

HOME_DIR="$(getent passwd "${USERNAME}" | cut -d: -f6)"
SSH_DIR="${HOME_DIR}/.ssh"
AUTHORIZED_KEYS="${SSH_DIR}/authorized_keys"

install -d -m 700 -o "${USERNAME}" -g "${USERNAME}" "${SSH_DIR}"
touch "${AUTHORIZED_KEYS}"
chown "${USERNAME}:${USERNAME}" "${AUTHORIZED_KEYS}"
chmod 600 "${AUTHORIZED_KEYS}"

if ! grep -qxF "${PUBLIC_KEY}" "${AUTHORIZED_KEYS}"; then
  printf '%s\n' "${PUBLIC_KEY}" >> "${AUTHORIZED_KEYS}"
fi

chgrp -R "${GROUP_NAME}" "${REPO_ROOT}"
find "${REPO_ROOT}" -type d -exec chmod 2775 {} +

echo "Provisioned ${USERNAME} and installed SSH key."
echo "Repo root: ${REPO_ROOT}"
