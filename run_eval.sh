#!/bin/bash
set -e  # 任何一步出错，立刻 exit 1 中断整条流水线

DATASETS=("benchs/pope/coco/coco_pope_adversarial.jsonl")
METHODS=("greedy" "mhcd-ae")

echo "==============================================="
echo "  🚀 双轨评测流水线启动 (adversarial only)    "
echo "  数据集: ${DATASETS[*]}"
echo "  方法:   ${METHODS[*]}"
echo "==============================================="

for dataset in "${DATASETS[@]}"
do
    for method in "${METHODS[@]}"
    do
        echo ""
        echo "-----------------------------------------------"
        echo "👉 正在评测: $dataset | 模式: $method"
        echo "-----------------------------------------------"

        python main.py --dataset "$dataset" --method "$method"

        echo "✅ [$dataset × $method] 评测完成！"
    done
done

echo ""
echo "==============================================="
echo "  🎉 全部 双轨×三数据集 评测完成！            "
echo "==============================================="
