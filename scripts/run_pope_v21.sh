#!/bin/bash
set -e

: "${COCO_IMAGE_ROOT:?Set COCO_IMAGE_ROOT to your COCO val2014 image root}"

DATASETS=(
  "benchs/pope/coco/coco_pope_random.jsonl"
  "benchs/pope/coco/coco_pope_popular.jsonl"
  "benchs/pope/coco/coco_pope_adversarial.jsonl"
)

METHODS=(
  "greedy"
  "label_pos"
  "vcd_label_const"
  "eg_label_cd"
  "sample_majority"
)

for dataset in "${DATASETS[@]}"
do
  for method in "${METHODS[@]}"
  do
    echo "==> ${dataset} | ${method}"
    python3 main.py \
      --dataset "$dataset" \
      --coco-image-root "$COCO_IMAGE_ROOT" \
      --method "$method" \
      --output-dir outputs/qwen25vl_pope \
      --trace-dir outputs/traces \
      --neg-type gaussian \
      --neg-std 0.2 \
      --seed 42
  done
done
