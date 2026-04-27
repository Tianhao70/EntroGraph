#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if .venv/bin/python -m pip show flash-attn >/dev/null 2>&1; then
  echo "INSTALLED"
  .venv/bin/python -m pip show flash-attn
  exit 0
fi

echo "STATUS"
pgrep -af "pip install flash-attn" || true
pgrep -af "ninja" || true
pgrep -af "nvcc" || true
pgrep -af "cicc" || true

echo "LOG_TAIL"
if [[ -f flash_attn_build.log ]]; then
  tail -n 60 flash_attn_build.log
else
  echo "flash_attn_build.log not found"
fi
