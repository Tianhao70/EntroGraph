import json

def get_metrics(results, gt_list=None):
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    unknown = 0
    yes_answers = 0
    total_questions = len(results)

    for i, item in enumerate(results):
        gt_answer = item.get("ground_truth")
        if gt_answer is None and gt_list is not None:
            gt_answer = gt_list[i]
            
        if not gt_answer:
            continue
            
        gen_answer = item.get("best_answer", "")
        
        gt_answer = str(gt_answer).lower().strip()
        gen_answer = str(gen_answer).lower().strip()
        
        # pos = 'yes', neg = 'no'
        if gt_answer == "yes":
            if "yes" in gen_answer:
                true_pos += 1
                yes_answers += 1
            else:
                false_neg += 1
        elif gt_answer == "no":
            if "no" in gen_answer:
                true_neg += 1
            else:
                yes_answers += 1
                false_pos += 1
        else:
            unknown += 1

    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (true_pos + true_neg) / total_questions if total_questions > 0 else 0
    yes_proportion = yes_answers / total_questions if total_questions > 0 else 0
    
    return {
        "Accuracy": accuracy * 100,
        "Precision": precision * 100,
        "Recall": recall * 100,
        "F1": f1 * 100,
        "Yes Ratio": yes_proportion * 100
    }

def main():
    print("Loading greedy results...")
    try:
        with open("results_coco_greedy.json", "r", encoding="utf-8") as f:
            greedy_data = json.load(f)
    except Exception as e:
        print("Failed to load greedy results:", e)
        return

    # Extract ground truth sequence
    gt_list = [item.get("ground_truth") for item in greedy_data]
    
    print("-" * 50)
    print("====== GREEDY MODE (Baseline) ======")
    greedy_metrics = get_metrics(greedy_data)
    for k, v in greedy_metrics.items():
        print(f"{k:12s}: {v:.2f}%")
        
    print("\nLoading MHCD-AE results...")
    try:
        with open("results_coco_mhcd-ae.json", "r", encoding="utf-8") as f:
            mhcd_data = json.load(f)
    except Exception as e:
        print("Failed to load mhcd-ae results:", e)
        return
        
    print("-" * 50)
    print("====== MHCD-AE MODE (Ours) ======")
    mhcd_metrics = get_metrics(mhcd_data, gt_list)
    for k, v in mhcd_metrics.items():
        print(f"{k:12s}: {v:.2f}%")
        
    print("-" * 50)
    print("====== COMPARISON (MHCD-AE vs GREEDY) ======")
    for k in greedy_metrics:
        diff = mhcd_metrics[k] - greedy_metrics[k]
        sign = "+" if diff > 0 else ""
        print(f"{k:12s}: {sign}{diff:.2f}%")

if __name__ == "__main__":
    main()
