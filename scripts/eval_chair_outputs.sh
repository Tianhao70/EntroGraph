#!/bin/bash
set -e

: "${COCO_ANN_ROOT:?Set COCO_ANN_ROOT=/path/to/coco/annotations}"

METHODS=("greedy" "token_cd" "eg_mhcd_ae")

for method in "${METHODS[@]}"; do
  python3 -m src.eval.chair_eval \
    --input "outputs/qwen25vl_chair/results_chair_${method}.json" \
    --annotation-root "$COCO_ANN_ROOT" \
    --output "outputs/qwen25vl_chair/metrics_chair_${method}.json"
done

python3 scripts/summarize_open_outputs.py \
  --task chair \
  --input-dir outputs/qwen25vl_chair \
  --output outputs/qwen25vl_chair/summary_chair.csv

