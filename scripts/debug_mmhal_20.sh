#!/bin/bash
set -e

: "${MMHAL_ROOT:?Set MMHAL_ROOT=/path/to/MMHal-Bench}"

METHOD=${1:-greedy}

python3 main.py \
  --task mmhal \
  --mmhal-root "$MMHAL_ROOT" \
  --method "$METHOD" \
  --output-dir outputs/debug_mmhal \
  --trace-dir outputs/debug_mmhal/traces \
  --max-new-tokens 128 \
  --neg-type gaussian \
  --neg-std 0.2 \
  --num-candidates 5 \
  --seed 42 \
  --limit 20

