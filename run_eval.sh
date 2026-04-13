#!/bin/bash

# 定义需要评测的数据集列表
DATASETS=("benchs/pope/coco" "aokvqa" "gqa")

echo "==============================================="
echo "  🚀 开始批量执行 MHCD-AE 测试集评测流水线  "
echo "==============================================="

for dataset in "${DATASETS[@]}"
do
    echo ""
    echo "-----------------------------------------------"
    echo "👉 正在评测数据集: $dataset"
    echo "-----------------------------------------------"
    
    # 调用 main.py，通过命令行参数传递当前数据集名称与默认方法 (你可以手动修改为 greedy 测试)
    python main.py --dataset "$dataset" --method mhcd-ae
    
    # 检查上一条命令是否执行成功
    if [ $? -eq 0 ]; then
        echo "✅ 数据集 $dataset 评测完成！"
    else
        echo "❌ 数据集 $dataset 评测失败或被中断！"
        # 可以选择是否遇到错误就退出
        # exit 1
    fi
done

echo ""
echo "==============================================="
echo "  🎉 所有数据集批量评测完成！  "
echo "==============================================="
