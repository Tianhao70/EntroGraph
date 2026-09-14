#!/bin/bash
set -e

: "${COCO_IMAGE_ROOT:?Set COCO_IMAGE_ROOT=/path/to/coco/val2014}"
: "${COCO_ANN_ROOT:?Set COCO_ANN_ROOT=/path/to/coco/annotations}"

METHOD=${1:-greedy}

python3 main.py \
  --task chair \
  --dataset-name coco_val2014 \
  --coco-image-root "$COCO_IMAGE_ROOT" \
  --coco-annotation-root "$COCO_ANN_ROOT" \
  --method "$METHOD" \
  --output-dir outputs/debug_chair \
  --trace-dir outputs/debug_chair/traces \
  --max-new-tokens 64 \
  --neg-type gaussian \
  --neg-std 0.2 \
  --num-candidates 5 \
  --seed 42 \
  --limit 20

