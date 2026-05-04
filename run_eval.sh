#!/bin/bash
set -e

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

: "${COCO_IMAGE_ROOT:?Set COCO_IMAGE_ROOT to your COCO val2014 image root}"

echo "==============================================="
echo "  🚀 EG-MHCD-AE v2 POPE 评测流水线启动"
echo "  数据集: ${DATASETS[*]}"
echo "  方法:   ${METHODS[*]}"
echo "  图像根目录: $COCO_IMAGE_ROOT"
echo "==============================================="

for dataset in "${DATASETS[@]}"
do
    for method in "${METHODS[@]}"
    do
        echo ""
        echo "-----------------------------------------------"
        echo "👉 正在评测: $dataset | 模式: $method"
        echo "-----------------------------------------------"

        python3 main.py \
            --dataset "$dataset" \
            --coco-image-root "$COCO_IMAGE_ROOT" \
            --method "$method" \
            --output-dir outputs/qwen25vl_pope \
            --neg-type gaussian \
            --neg-std 0.2 \
            --seed 42

        echo "✅ [$dataset × $method] 评测完成！"
    done
done

echo ""
echo "==============================================="
echo "  🎉 全部评测完成！            "
echo "==============================================="
