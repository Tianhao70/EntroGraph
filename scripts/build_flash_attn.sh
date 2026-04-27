#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

source /etc/profile.d/cuda-12-8.sh 2>/dev/null || true
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$PWD/.venv/bin:$PATH"
export FLASH_ATTN_CUDA_ARCHS=120
export TORCH_CUDA_ARCH_LIST=12.0
export MAX_JOBS="${MAX_JOBS:-8}"

exec .venv/bin/python -m pip install flash-attn --no-build-isolation --no-cache-dir
