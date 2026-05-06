#!/usr/bin/env bash
set -euo pipefail

GROUP_NAME="ml-lora"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run as root or with sudo." >&2
  exit 1
fi

if ! getent group "${GROUP_NAME}" >/dev/null; then
  groupadd "${GROUP_NAME}"
fi

chgrp -R "${GROUP_NAME}" "${REPO_ROOT}"
find "${REPO_ROOT}" -type d -exec chmod 2775 {} +
find "${REPO_ROOT}" -type f -exec chmod 664 {} +

echo "Configured ${REPO_ROOT} for shared group ${GROUP_NAME}."
