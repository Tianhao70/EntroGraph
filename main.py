import os
import argparse
import json
import time
import random


def get_attention_implementation():
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except ImportError:
        print("⚠️ 未检测到 flash-attn，自动使用 PyTorch SDPA 注意力实现。")
        return "sdpa"


def seed_everything(seed):
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_max_new_tokens(method):
    if method in ("greedy", "label_pos", "vcd_label_const", "eg_label_cd"):
        return 8
    return 128


def load_dataset_from_path(dataset_path, coco_image_root):
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

                img_path = os.path.join(coco_image_root, image_name)
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


def write_trace_file(final_report, trace_dir, dataset_basename, method):
    trace_paths = []
    trace_root = os.path.join(trace_dir, _safe_path_segment(method), _safe_path_segment(dataset_basename))
    os.makedirs(trace_root, exist_ok=True)

    for item_index, item in enumerate(final_report):
        base = {
            "question_id": item.get("question_id"),
            "question": item.get("question"),
            "image_name": item.get("image_name"),
            "method": method,
        }
        rows = []
        generation = item.get("generation")
        if generation and generation.get("trace"):
            for step in generation["trace"]:
                rows.append({**base, "kind": "generation", **step})

        for candidate in item.get("candidate_details", []) or []:
            if candidate.get("trace"):
                for step in candidate["trace"]:
                    rows.append({
                        **base,
                        "kind": "candidate",
                        "path_id": candidate.get("path_id"),
                        "config": candidate.get("config"),
                        **step,
                    })

        if not rows:
            continue

        trace_id = _trace_file_id(item, item_index)
        trace_path = os.path.join(trace_root, f"{trace_id}.jsonl")
        with open(trace_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        trace_paths.append(trace_path)

    return trace_paths


def build_result_base(item, method, best_answer):
    return {
        "question_id": item.get("question_id"),
        "image_name": item.get("image_name"),
        "image_path": item.get("image_path"),
        "question": item.get("question"),
        "ground_truth": item.get("ground_truth"),
        "method": method,
        "best_answer": best_answer,
    }


def _trace_file_id(item, item_index):
    value = item.get("question_id")
    if value is None:
        source_name = os.path.splitext(str(item.get("source_file") or ""))[0]
        source_index = item.get("source_index")
        if source_name and source_index is not None:
            value = f"{source_name}_{source_index}"
        elif item.get("image_name"):
            value = os.path.splitext(str(item["image_name"]))[0]
        else:
            value = f"item_{item_index:06d}"
    return _safe_path_segment(value)


def _safe_path_segment(value):
    text = str(value).strip()
    safe_chars = []
    for char in text:
        if char.isalnum() or char in ("-", "_", "."):
            safe_chars.append(char)
        else:
            safe_chars.append("_")
    return "".join(safe_chars).strip("._") or "unknown"


def main():
    parser = argparse.ArgumentParser(description="EG-MHCD-AE v2 评测流水线")
    parser.add_argument(
        "--method",
        type=str,
        choices=[
            "greedy",
            "sample_majority",
            "label_pos",
            "vcd_label_const",
            "eg_label_cd",
            "token_cd",
            "eg_mhcd_ae",
        ],
        default="eg_label_cd",
        help="解码/打分方法。",
    )
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--dataset", type=str, required=True, help="具体的测试集路径（如 benchs/pope/coco）")
    parser.add_argument("--coco-image-root", type=str, required=True, help="COCO val2014 图像根目录")
    parser.add_argument("--output-dir", type=str, default="outputs", help="结果 JSON 输出目录")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="未设置时：POPE/label 方法默认 8，生成方法默认 128")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--neg-type", type=str, choices=["gaussian", "blur", "gray"], default="gaussian")
    parser.add_argument("--neg-std", type=float, default=0.2)
    parser.add_argument("--alpha0", type=float, default=0.5)
    parser.add_argument("--alpha-max", type=float, default=2.0)
    parser.add_argument("--k-entropy", type=float, default=0.8)
    parser.add_argument("--topk-plausible", type=int, default=50)
    parser.add_argument("--num-candidates", type=int, default=5, help="eg_mhcd_ae 候选路径数")
    parser.add_argument("--trace-dir", type=str, default="outputs/traces", help="token trace 输出目录")
    parser.add_argument("--temperature", type=float, default=1.0, help="token/label logits 温度")
    parser.add_argument("--top-p", type=float, default=0.9, help="token_cd / eg_mhcd_ae nucleus sampling top_p")
    args = parser.parse_args()
    method = args.method
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else default_max_new_tokens(method)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.trace_dir, exist_ok=True)

    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    from qwen_data_engine import build_high_throughput_dataloader
    from src.decoding.candidate_generator import EGMHCDGenerator, SampleMajorityGenerator
    from src.decoding.label_cd import EGLabelCDScorer
    from src.decoding.token_cd import TokenCDGenerator
    from src.scoring.sindex_answer_entropy import EGAnswerEntropyScorer, MHCDScorer

    seed_everything(args.seed)

    print("="*50)
    print("🚀 启动 EG-MHCD-AE v2 评测流水线 (Powered by RTX 5090)")
    print(f"📊 数据集设定: {args.dataset}")
    print(f"🖼️  COCO 图像根目录: {args.coco_image_root}")
    print(f"⚙️  解码模式设定: {method}")
    print(f"🎲 随机种子: {args.seed}")
    print(f"🧪 负样本扰动: {args.neg_type} (std={args.neg_std})")
    print(f"🔢 max_new_tokens: {max_new_tokens}")
    print("="*50)

    # ---------------------------------------------------------
    # 步骤 1：全副武装，请神登基 (加载 Qwen2.5-VL)
    # ---------------------------------------------------------
    attn_implementation = get_attention_implementation()
    print(f"\n[1/5] 正在装载主模型 (BF16 + {attn_implementation})...")
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        device_map="cuda"
    )
    processor = AutoProcessor.from_pretrained(args.model_id)

    # ---------------------------------------------------------
    # 步骤 2：唤醒裁决官与生成器
    # ---------------------------------------------------------
    generator = None
    scorer = None
    label_scorer = None
    if method in ("label_pos", "vcd_label_const", "eg_label_cd"):
        label_beta = 0.0 if method == "label_pos" else args.alpha0
        label_k_entropy = args.k_entropy if method == "eg_label_cd" else 0.0
        print(f"\n[2/5] 正在初始化 POPE label-level scorer (β={label_beta})...")
        label_scorer = EGLabelCDScorer(
            model,
            processor,
            beta=label_beta,
            temperature=args.temperature,
            alpha_max=args.alpha_max,
            k_entropy=label_k_entropy,
            neg_type=args.neg_type,
            neg_std=args.neg_std,
        )
    elif method == "token_cd":
        print("\n[2/5] 正在初始化 true token-level contrastive decoder...")
        generator = TokenCDGenerator(
            model,
            processor,
            beta=args.alpha0,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.topk_plausible,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    elif method == "eg_mhcd_ae":
        print("\n[2/5] 正在初始化 EG-CD 多候选生成器与 EntroGraph AE 裁决官...")
        generator = EGMHCDGenerator(
            model,
            processor,
            max_new_tokens=max_new_tokens,
            num_candidates=args.num_candidates,
            top_p=args.top_p,
            topk_plausible=args.topk_plausible,
            neg_type=args.neg_type,
            neg_std=args.neg_std,
        )
        scorer = EGAnswerEntropyScorer(device="cuda")
    elif method == "sample_majority":
        print("\n[2/5] 正在初始化旧版 sample_majority 生成器与多数派裁决官...")
        generator = SampleMajorityGenerator(
            model,
            processor,
            max_new_tokens=max_new_tokens,
        )
        scorer = MHCDScorer(device="cuda")
    else:
        print("\n[2/5] Greedy 模式开启，跳过初始化多假设裁决官。")

    # ---------------------------------------------------------
    # 步骤 3：接入评测数据
    # ---------------------------------------------------------
    print("\n[3/5] 正在挂载评测数据集 (准备锁页内存传输)...")
    real_dataset = load_dataset_from_path(args.dataset, args.coco_image_root)
    print(f"✅ 成功从 {args.dataset} 加载了 {len(real_dataset)} 笔测试数据。")
    
    # 高吞吐 DataLoader, Batch Size 设为 1（直接传入 List，无需额外类型转换）
    dataloader = build_high_throughput_dataloader(real_dataset, processor, batch_size=1)

    # ---------------------------------------------------------
    # 步骤 4/5：引擎点火执行与裁决
    # ---------------------------------------------------------
    final_report = []
    trace_report = []
    
    if method in ("label_pos", "vcd_label_const", "eg_label_cd"):
        print("\n[4/5] ⚡ 执行 POPE Yes/No label-level contrastive logprob scoring...")
        start_time = time.time()
        model.eval()
        assert label_scorer is not None
        for batch_inputs, raw_items in dataloader:
            batch_inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in batch_inputs.items()}
            raw_item = raw_items[0]
            score_result = label_scorer.score(batch_inputs, raw_item)
            report_item = build_result_base(raw_item, method, score_result["best_answer"])
            report_item.update(
                {
                    "scores_pos": score_result["positive_label_logprobs"],
                    "scores_neg": score_result["negative_label_logprobs"],
                    "scores_cd": score_result["cd_label_scores"],
                    "H_label": score_result.get("answer_entropy"),
                    "JS_label": score_result.get("JS_label"),
                    "alpha": score_result.get("alpha"),
                    "risk": score_result.get("risk"),
                    "neg_type": args.neg_type,
                }
            )
            final_report.append(report_item)

            print(f"\n❓ 问题: {raw_item['question']}")
            print(
                f"🌟 {method}: "
                f"{score_result['best_answer']} "
                f"(P_yes={score_result['label_probs'].get('yes', 0.0):.4f}, "
                f"P_no={score_result['label_probs'].get('no', 0.0):.4f}, "
                f"H={score_result['answer_entropy']:.4f})"
            )

            torch.cuda.empty_cache()

        gen_time = time.time() - start_time
        print(f"\n[5/5] ✅ label-level CD 打分完毕！耗时: {gen_time:.2f} 秒")

    elif method == "token_cd":
        print("\n[4/5] ⚡ 执行 true token-level contrastive decoding...")
        start_time = time.time()
        assert generator is not None
        generation_report = generator.generate(dataloader)
        trace_report.extend(generation_report)
        for item in generation_report:
            generation = item.get("generation", {})
            report_item = build_result_base(item, method, item.get("best_answer", generation.get("text", "")))
            report_item.update(
                {
                    "text": generation.get("text", item.get("best_answer", "")),
                    "token_ids": generation.get("token_ids", []),
                    "H_cd": generation.get("H_cd"),
                    "D_vis": generation.get("D_vis"),
                    "S_graph": generation.get("S_graph"),
                    "avg_logprob_cd": generation.get("avg_logprob_cd"),
                }
            )
            final_report.append(report_item)
        gen_time = time.time() - start_time
        for item in generation_report:
            print(f"\n❓ 问题: {item['question']}")
            print(f"🌟 Token-CD 答案: {item['best_answer']}")
        print(f"\n[5/5] ✅ token_cd 生成完毕！耗时: {gen_time:.2f} 秒")

    elif method in ("eg_mhcd_ae", "sample_majority"):
        print("\n[4/5] ⚡ 引擎点火！开始多候选生成...")
        start_time = time.time()
        assert generator is not None
        
        generation_results = generator.generate_candidates(dataloader)
        trace_report.extend(
            {
                **{key: item.get(key) for key in ("question_id", "question", "image_name", "source_file", "source_index")},
                "candidate_details": item.get("candidates", []),
            }
            for item in generation_results
        )
        gen_time = time.time() - start_time
        print(f"✅ 生成完毕！耗时: {gen_time:.2f} 秒")

        print("\n[5/5] ⚖️ 移交裁决官进行重排...")
        assert scorer is not None
        for item in generation_results:
            question = item["question"]
            candidates = item["candidates"]

            if method == "eg_mhcd_ae":
                rerank_result = scorer.score_and_select(question, candidates)
                best_ans = rerank_result.best_text
                ae_scores = [score.AE for score in rerank_result.scores]
                clusters = rerank_result.clusters
                final_scores = [score.final_score for score in rerank_result.scores]
                candidate_scores = [score.__dict__ for score in rerank_result.scores]
                risk_high = rerank_result.risk_high
                h_cluster = rerank_result.H_cluster
                delta_ae = rerank_result.delta_AE
                best_index = rerank_result.best_index
                worst_index = rerank_result.worst_index
                rerank_mode = rerank_result.mode
            else:
                candidates_texts = [cand["text"] for cand in candidates]
                best_ans, ae_scores, clusters = scorer.score_and_select(question, candidates_texts)
                ae_scores = ae_scores.tolist()
                clusters = clusters.tolist()
                final_scores = None
                candidate_scores = None
                risk_high = None
                h_cluster = getattr(scorer, "last_answer_entropy", None)
                delta_ae = None
                best_index = None
                worst_index = None
                rerank_mode = getattr(scorer, "last_mode", None)
            
            report_item = build_result_base(item, method, best_ans)
            report_item.update(
                {
                    "all_candidates": [cand["text"] for cand in candidates],
                    "ae_scores": ae_scores,
                    "clusters": clusters,
                    "mode": rerank_mode,
                }
            )
            if method == "eg_mhcd_ae":
                report_item.update(
                    {
                        "H_cluster": h_cluster,
                        "delta_AE": delta_ae,
                        "best_index": best_index,
                        "worst_index": worst_index,
                        "candidate_scores": candidate_scores,
                        "risk_high": risk_high,
                        "final_scores": final_scores,
                    }
                )
            final_report.append(report_item)
            
            print(f"\n❓ 问题: {question}")
            print(f"🌟 最终优选答案: {best_ans}")
    else:
        # greedy 模式
        print("\n[4/5] ⚡ 引擎点火！执行原始贪心解码 (Greedy)...")
        start_time = time.time()
        model.eval()
        for batch_inputs, raw_items in dataloader:
            batch_inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in batch_inputs.items()}
            
            output_ids = model.generate(
                **batch_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
            
            input_len = batch_inputs['input_ids'].shape[1]
            generated_text = processor.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)
            
            report_item = build_result_base(raw_items[0], method, generated_text[0])
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
        
    trace_paths = write_trace_file(trace_report, args.trace_dir, dataset_basename, method)
    output_filename = os.path.join(args.output_dir, f"results_{dataset_basename}_{method}.json")
    
    # 保存最终结果到本地
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)
        
    print(f"\n🎉 全线贯通！评测报告已成功保存至 {output_filename}")
    if trace_paths:
        trace_root = os.path.join(args.trace_dir, _safe_path_segment(method), _safe_path_segment(dataset_basename))
        print(f"🧾 token trace 已保存至 {trace_root} ({len(trace_paths)} files)")
    print("="*50)

if __name__ == "__main__":
    main()
