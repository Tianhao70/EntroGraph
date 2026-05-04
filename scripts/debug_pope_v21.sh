#!/bin/bash
set -e

: "${COCO_IMAGE_ROOT:?Set COCO_IMAGE_ROOT to your COCO val2014 image root}"

DATASET="${1:-benchs/pope/coco/coco_pope_random.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/debug_v21}"
TRACE_DIR="${TRACE_DIR:-outputs/traces}"
SEED="${SEED:-42}"
LIMIT="${LIMIT:-3}"

METHODS=(
    "eg_label_cd"
    "token_cd"
    "eg_mhcd_ae"
)

for method in "${METHODS[@]}"
do
    echo "==> Debug ${method} on ${DATASET} (limit=${LIMIT})"
    python3 main.py \
        --dataset "$DATASET" \
        --coco-image-root "$COCO_IMAGE_ROOT" \
        --method "$method" \
        --output-dir "$OUTPUT_DIR" \
        --trace-dir "$TRACE_DIR" \
        --neg-type gaussian \
        --neg-std 0.2 \
        --seed "$SEED" \
        --perturb-seed-base "$SEED" \
        --max-new-tokens 8 \
        --limit "$LIMIT"
done
