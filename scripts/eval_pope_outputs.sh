#!/bin/bash
set -e

for split in random popular adversarial
do
  echo "==> Evaluating ${split}"
  python3 evaluate_results.py \
    --greedy "outputs/qwen25vl_pope/results_coco_pope_${split}_greedy.json" \
    --candidate "outputs/qwen25vl_pope/results_coco_pope_${split}_eg_label_cd.json"
done
