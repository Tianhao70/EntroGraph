#!/bin/bash
set -e

: "${COCO_IMAGE_ROOT:?Set COCO_IMAGE_ROOT=/path/to/coco/val2014}"
: "${COCO_ANN_ROOT:?Set COCO_ANN_ROOT=/path/to/coco/annotations}"

METHODS=("greedy" "token_cd" "eg_mhcd_ae")

for method in "${METHODS[@]}"; do
  python3 main.py \
    --task chair \
    --dataset-name coco_val2014 \
    --coco-image-root "$COCO_IMAGE_ROOT" \
    --coco-annotation-root "$COCO_ANN_ROOT" \
    --method "$method" \
    --output-dir outputs/qwen25vl_chair \
    --trace-dir outputs/qwen25vl_chair/traces \
    --max-new-tokens 64 \
    --neg-type gaussian \
    --neg-std 0.2 \
    --num-candidates 5 \
    --topk-plausible 50 \
    --top-p 0.9 \
    --seed 42 \
    --limit 500
done

