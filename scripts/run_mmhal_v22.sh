#!/bin/bash
set -e

: "${MMHAL_ROOT:?Set MMHAL_ROOT=/path/to/MMHal-Bench}"

METHODS=("greedy" "token_cd" "eg_mhcd_ae")

for method in "${METHODS[@]}"; do
  python3 main.py \
    --task mmhal \
    --mmhal-root "$MMHAL_ROOT" \
    --method "$method" \
    --output-dir outputs/qwen25vl_mmhal \
    --trace-dir outputs/qwen25vl_mmhal/traces \
    --max-new-tokens 128 \
    --neg-type gaussian \
    --neg-std 0.2 \
    --num-candidates 5 \
    --topk-plausible 50 \
    --top-p 0.9 \
    --seed 42
done

