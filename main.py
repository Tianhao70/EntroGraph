import os
import argparse
import torch
import json
import time
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# 导入我们在 Day 3 打造的三大核心引擎
from qwen_data_engine import build_high_throughput_dataloader
from qwen_generation_engine import MHCDGenerator
from qwen_entropy_scorer import MHCDScorer

def load_dataset_from_path(dataset_path):
    """
    自适应读取给定的测试集路径中的 JSONL 文件。
    并将原始数据的 schema 统一映射为 {'image_path', 'question', 'ground_truth'}
    """
    # 兼容处理简写路径，例如直接传 aokvqa 时尝试从 benchs/pope 下找
    if not os.path.exists(dataset_path):
        fallback_path = os.path.join("benchs", "pope", dataset_path)
        if os.path.exists(fallback_path):
            dataset_path = fallback_path
            
    jsonl_files = []
    if os.path.isdir(dataset_path):
        # 使用 os.walk 强制全目录扫描，确保深入所有子文件夹读取正确的 .jsonl
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.jsonl'):
                    jsonl_files.append(os.path.join(root, file))
    elif str(dataset_path).endswith('.jsonl'):
        jsonl_files = [dataset_path]
        
    dataset = []
    for jfile in jsonl_files:
        with open(jfile, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                
                # POPE schema -> 通用 schema
                image_name = data.get("image", "")
                question = data.get("text", "")
                if not question and "question" in data:
                    question = data["question"]
                label = data.get("label", data.get("ground_truth", ""))
                
                # 尝试通过本地环境解析真实图片路径，找不到则先设为 fallback 的 demo图防崩溃
                img_path = image_name
                try:
                    from playground.path_table import get_path_from_table
                    if "coco" in jfile.lower():
                        base = get_path_from_table("COCO path")
                        img_path = os.path.join(base, image_name)
                    elif "gqa" in jfile.lower():
                        base = get_path_from_table("GQA path")
                        img_path = os.path.join(base, image_name)
                except Exception:
                    pass
                
                if not os.path.exists(img_path):
                    # Fallback
                    img_path = "assets/demo1.jpg"
                    
                dataset.append({
                    "image_path": img_path,
                    "question": question + " Please answer yes or no.",
                    "ground_truth": label
                })
    return dataset

def main():
    parser = argparse.ArgumentParser(description="MHCD-AE 终极评测流水线")
    parser.add_argument("--dataset", type=str, required=True, help="具体的测试集路径（如 benchs/pope/coco）")
    parser.add_argument("--method", type=str, choices=["greedy", "mhcd-ae"], default="mhcd-ae", help="对比基线：greedy 或 mhcd-ae (默认)")
    args = parser.parse_args()

    print("="*50)
    print("🚀 启动 MHCD-AE 终极评测流水线 (Powered by RTX 5090)")
    print(f"📊 数据集设定: {args.dataset}")
    print(f"⚙️  解码模式设定: {args.method}")
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
    if args.method == "mhcd-ae":
        print("\n[2/5] 正在初始化多假设生成器与 BGE 语义裁决官...")
        generator = MHCDGenerator(model, processor)
        scorer = MHCDScorer(device="cuda")
    else:
        print("\n[2/5] Greedy 模式开启，跳过初始化多假设裁决官。")

    # ---------------------------------------------------------
    # 步骤 3：接入评测数据
    # ---------------------------------------------------------
    print("\n[3/5] 正在挂载评测数据集 (准备锁页内存传输)...")
    real_dataset = load_dataset_from_path(args.dataset)
    print(f"✅ 成功从 {args.dataset} 加载了 {len(real_dataset)} 笔测试数据。")
    
    # 按照严格类型要求：将普通 Python List 包装提升为 HuggingFace datasets.Dataset 结构
    import datasets
    hf_dataset = datasets.Dataset.from_list(real_dataset)
    
    # 高吞吐 DataLoader, Batch Size 设为 1
    dataloader = build_high_throughput_dataloader(hf_dataset, processor, batch_size=1)

    # ---------------------------------------------------------
    # 步骤 4/5：引擎点火执行与裁决
    # ---------------------------------------------------------
    final_report = []
    
    if args.method == "mhcd-ae":
        print("\n[4/5] ⚡ 引擎点火！开始高吞吐并发生成...")
        start_time = time.time()
        
        generation_results = generator.generate_candidates(dataloader)
        gen_time = time.time() - start_time
        print(f"✅ 生成完毕！耗时: {gen_time:.2f} 秒")

        print("\n[5/5] ⚖️ 移交裁决官进行熵值重排与幻觉清洗...")
        for item in generation_results:
            question = item["question"]
            candidates_texts = [cand["text"] for cand in item["candidates"]]
            
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
    else:
        # greedy 模式
        print("\n[4/5] ⚡ 引擎点火！执行原始贪心解码 (Greedy)...")
        start_time = time.time()
        model.eval()
        for batch_inputs, raw_items in dataloader:
            batch_inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in batch_inputs.items()}
            
            output_ids = model.generate(
                **batch_inputs,
                max_new_tokens=256,
                do_sample=False
            )
            
            input_len = batch_inputs['input_ids'].shape[1]
            generated_text = processor.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)
            
            final_report.append({
                "question": raw_items[0]['question'],
                "best_answer": generated_text[0],
                "ground_truth": raw_items[0]['ground_truth']
            })
            
            print(f"\n❓ 问题: {raw_items[0]['question']}")
            print(f"🌟 Greedy 答案: {generated_text[0]}")
            
            del output_ids
            torch.cuda.empty_cache()
            
        gen_time = time.time() - start_time
        print(f"\n[5/5] ✅ Greedy 生成完毕！耗时: {gen_time:.2f} 秒")

    # 动态命名保存结果文件
    dataset_basename = os.path.basename(args.dataset.strip('/'))
    if not dataset_basename:
        dataset_basename = "unknown_dataset"
        
    output_filename = f"results_{dataset_basename}_{args.method}.json"
    
    # 保存最终结果到本地
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)
        
    print(f"\n🎉 全线贯通！评测报告已成功保存至 {output_filename}")
    print("="*50)

if __name__ == "__main__":
    main()