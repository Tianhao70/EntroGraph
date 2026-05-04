#!/bin/bash
set -e

: "${COCO_IMAGE_ROOT:?Set COCO_IMAGE_ROOT to your COCO val2014 image root}"

python3 - <<'PY'
import importlib.util

assert importlib.util.find_spec("sentence_transformers") is not None, (
    "Please install: pip install -r requirements-ml.txt"
)
PY

python3 main.py \
  --dataset benchs/pope/coco/coco_pope_random.jsonl \
  --coco-image-root "$COCO_IMAGE_ROOT" \
  --method eg_mhcd_ae \
  --output-dir outputs/debug \
  --trace-dir outputs/debug/traces \
  --neg-type gaussian \
  --neg-std 0.2 \
  --num-candidates 5 \
  --max-new-tokens 8 \
  --topk-plausible 50 \
  --top-p 0.9 \
  --seed 42 \
  --limit 20
