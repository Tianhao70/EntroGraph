import torch
import json
import time
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# 导入我们在 Day 3 打造的三大核心引擎
from qwen_data_engine import build_high_throughput_dataloader
from qwen_generation_engine import MHCDGenerator
from qwen_entropy_scorer import MHCDScorer

def main():
    print("="*50)
    print("🚀 启动 MHCD-AE 终极评测流水线 (Powered by RTX 5090)")
    print("="*50)

    # ---------------------------------------------------------
    # 步骤 1：全副武装，请神登基 (加载 Qwen2.5-VL)
    # ---------------------------------------------------------
    print("\n[1/5] 正在装载主模型 (BF16 + FlashAttention-2)...")
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cuda"
    )
    processor = AutoProcessor.from_pretrained(model_id)

    # ---------------------------------------------------------
    # 步骤 2：唤醒裁决官与生成器
    # ---------------------------------------------------------
    print("\n[2/5] 正在初始化多假设生成器与 BGE 语义裁决官...")
    generator = MHCDGenerator(model, processor)
    scorer = MHCDScorer(device="cuda")

    # ---------------------------------------------------------
    # 步骤 3：接入评测数据 (此处以 POPE 风格的 Mock 数据为例)
    # ---------------------------------------------------------
    print("\n[3/5] 正在挂载评测数据集 (准备锁页内存传输)...")
    # 真实场景下，你应该在这里写一段逻辑读取 POPE 或 CHAIR 的 jsonl 文件
    # 这里我们用两笔高难度测试数据来演示流水线
    mock_dataset = [
        {
            "image_path": "assets/demo1.jpg", # 确保你的目录下随便放两张测试图
            "question": "Is there a dog in the image? Please answer yes or no.",
            "ground_truth": "no"
        },
        {
            "image_path": "assets/demo2.jpg",
            "question": "What is the person in the red shirt doing?",
            "ground_truth": "playing soccer"
        }
    ]
    
    # 召唤你的 48G 大内存数据引擎，Batch Size 设为 1 (因为我们要对每张图生成 5 次)
    dataloader = build_high_throughput_dataloader(mock_dataset, processor, batch_size=1)

    # ---------------------------------------------------------
    # 步骤 4：引擎点火，执行多假设生成
    # ---------------------------------------------------------
    print("\n[4/5] ⚡ 引擎点火！开始高吞吐并发生成...")
    start_time = time.time()
    
    # 返回的结果是一个列表，每个元素包含 raw_question 和 5 个 candidates
    generation_results = generator.generate_candidates(dataloader)
    
    gen_time = time.time() - start_time
    print(f"✅ 生成完毕！耗时: {gen_time:.2f} 秒")

    # ---------------------------------------------------------
    # 步骤 5：移交裁决，输出最终报告
    # ---------------------------------------------------------
    print("\n[5/5] ⚖️ 移交裁决官进行熵值重排与幻觉清洗...")
    
    final_report = []
    
    for item in generation_results:
        question = item["question"]
        candidates_texts = [cand["text"] for cand in item["candidates"]]
        
        # 核心算法介入：计算 AE 熵并择优
        best_ans, ae_scores, clusters = scorer.score_and_select(question, candidates_texts)
        
        final_report.append({
            "question": question,
            "best_answer": best_ans,
            "all_candidates": candidates_texts,
            "ae_scores": ae_scores.tolist(),
            "clusters": clusters.tolist()
        })
        
        print(f"\n❓ 问题: {question}")
        print(f"🌟 最终优选答案 (最低熵): {best_ans}")

    # 保存最终结果到本地
    with open("mhcd_eval_results.json", "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)
        
    print("\n🎉 全线贯通！评测报告已保存至 mhcd_eval_results.json")
    print("="*50)

if __name__ == "__main__":
    # 请确保在运行前，你在 EntroGraph 目录下建了一个 assets 文件夹，并放了两张任意的 jpg 图片测试用
    main()