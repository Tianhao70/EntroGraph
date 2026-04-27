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


def get_attention_implementation():
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except ImportError:
        print("⚠️ 未检测到 flash-attn，自动使用 PyTorch SDPA 注意力实现。")
        return "sdpa"


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
        official_files = []
        resampled_files = []
        # For final POPE evaluation, prefer the official adversarial split and
        # do not mix it with resampled head-identification files.
        for root, dirs, files in os.walk(dataset_path):
            for file in sorted(files):
                lower_name = file.lower()
                if not file.endswith('.jsonl') or 'adversarial' not in lower_name:
                    continue
                full_path = os.path.join(root, file)
                if 'resampled' in lower_name:
                    resampled_files.append(full_path)
                else:
                    official_files.append(full_path)
        jsonl_files = official_files if official_files else resampled_files
    elif str(dataset_path).endswith('.jsonl'):
        jsonl_files = [dataset_path]

    if not jsonl_files:
        raise FileNotFoundError(f"严重错误：在 {dataset_path} 下未找到任何 adversarial .jsonl 文件！")

    print(f"📂 扫描到 {len(jsonl_files)} 个对抗集文件: {[os.path.basename(f) for f in jsonl_files]}")

    dataset = []
    for jfile in sorted(jsonl_files):
        with open(jfile, "r", encoding="utf-8") as f:
            row_idx = 0
            for line in f:
                if not line.strip():
                    continue
                row_idx += 1
                data = json.loads(line)

                # POPE schema -> 通用 schema
                image_name = data.get("image", "")
                question = data.get("text", "")
                if not question and "question" in data:
                    question = data["question"]
                label = data.get("label", data.get("ground_truth", ""))

                # 强制拼接 COCO val2014 图片绝对路径
                COCO_IMAGE_ROOT = "/home/tianhao/LLM_Workspace/datasets/coco/val2014"
                img_path = os.path.join(COCO_IMAGE_ROOT, image_name)
                if not os.path.exists(img_path):
                    raise FileNotFoundError(f"严重错误：找不到测试图像 {img_path}")

                dataset.append({
                    "image_path": img_path,
                    "image_name": image_name,
                    "question": question + " Please answer yes or no.",
                    "ground_truth": label,
                    "question_id": data.get("question_id"),
                    "source_file": os.path.basename(jfile),
                    "source_index": row_idx
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
    attn_implementation = get_attention_implementation()
    print(f"\n[1/5] 正在装载主模型 (BF16 + {attn_implementation})...")
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
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
    
    # 高吞吐 DataLoader, Batch Size 设为 1（直接传入 List，无需额外类型转换）
    dataloader = build_high_throughput_dataloader(real_dataset, processor, batch_size=1)

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
            
            report_item = {
                "question": question,
                "best_answer": best_ans,
                "all_candidates": candidates_texts,
                "ae_scores": ae_scores.tolist(),
                "clusters": clusters.tolist(),
                "selection_mode": getattr(scorer, "last_mode", None),
                "candidate_labels": getattr(scorer, "last_candidate_labels", None),
                "label_counts": getattr(scorer, "last_label_counts", None)
            }
            for key in ("ground_truth", "image_path", "image_name", "question_id", "source_file", "source_index"):
                if item.get(key) is not None:
                    report_item[key] = item[key]
            final_report.append(report_item)
            
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
            
            report_item = {
                "question": raw_items[0]['question'],
                "best_answer": generated_text[0],
                "ground_truth": raw_items[0]['ground_truth']
            }
            for key in ("image_path", "image_name", "question_id", "source_file", "source_index"):
                if raw_items[0].get(key) is not None:
                    report_item[key] = raw_items[0][key]
            final_report.append(report_item)
            
            print(f"\n❓ 问题: {raw_items[0]['question']}")
            print(f"🌟 Greedy 答案: {generated_text[0]}")
            
            del output_ids
            torch.cuda.empty_cache()
            
        gen_time = time.time() - start_time
        print(f"\n[5/5] ✅ Greedy 生成完毕！耗时: {gen_time:.2f} 秒")

    # 动态命名保存结果文件
    dataset_basename = os.path.splitext(os.path.basename(args.dataset.strip('/')))[0]
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
