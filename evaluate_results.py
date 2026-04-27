import argparse
import json
import os
import re
from collections import Counter


YES_NO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def extract_yes_no(answer):
    matches = YES_NO_RE.findall(str(answer).lower())
    if not matches:
        return None
    unique = set(matches)
    if len(unique) != 1:
        return None
    return matches[0]


def resolve_path(path, fallback):
    if os.path.exists(path):
        return path
    if fallback and os.path.exists(fallback):
        return fallback
    return path


def get_metrics(results, gt_list=None):
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    unknown_pred = 0
    unknown_gt = 0
    yes_answers = 0
    evaluated = 0
    confusion = Counter()

    for i, item in enumerate(results):
        gt_answer = item.get("ground_truth")
        if gt_answer is None and gt_list is not None:
            gt_answer = gt_list[i]

        gt_answer = str(gt_answer).lower().strip() if gt_answer is not None else None
        if gt_answer not in ("yes", "no"):
            unknown_gt += 1
            continue

        pred_answer = extract_yes_no(item.get("best_answer", ""))
        if pred_answer is None:
            unknown_pred += 1
            pred_answer = "unknown"

        evaluated += 1
        if pred_answer == "yes":
            yes_answers += 1
        confusion[(gt_answer, pred_answer)] += 1

        if gt_answer == "yes":
            if pred_answer == "yes":
                true_pos += 1
            else:
                false_neg += 1
        elif gt_answer == "no":
            if pred_answer == "no":
                true_neg += 1
            else:
                false_pos += 1

    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (true_pos + true_neg) / evaluated if evaluated > 0 else 0
    yes_proportion = yes_answers / evaluated if evaluated > 0 else 0

    return {
        "N": evaluated,
        "TP": true_pos,
        "TN": true_neg,
        "FP": false_pos,
        "FN": false_neg,
        "Unknown Pred": unknown_pred,
        "Unknown GT": unknown_gt,
        "Accuracy": accuracy * 100,
        "Precision": precision * 100,
        "Recall": recall * 100,
        "F1": f1 * 100,
        "Yes Ratio": yes_proportion * 100,
        "Confusion": confusion,
    }


def print_metrics(title, metrics):
    print(f"====== {title} ======")
    for key in ("N", "TP", "TN", "FP", "FN", "Unknown Pred", "Unknown GT"):
        print(f"{key:12s}: {metrics[key]}")
    for key in ("Accuracy", "Precision", "Recall", "F1", "Yes Ratio"):
        print(f"{key:12s}: {metrics[key]:.2f}%")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Evaluate greedy vs MHCD-AE yes/no results.")
    parser.add_argument(
        "--greedy",
        default="results_coco_pope_adversarial_greedy.json",
        help="Path to greedy result JSON.",
    )
    parser.add_argument(
        "--mhcd",
        default="results_coco_pope_adversarial_mhcd-ae.json",
        help="Path to MHCD-AE result JSON.",
    )
    args = parser.parse_args()

    greedy_path = resolve_path(args.greedy, "results_coco_greedy.json")
    mhcd_path = resolve_path(args.mhcd, "results_coco_mhcd-ae.json")

    print(f"Loading greedy results: {greedy_path}")
    greedy_data = load_json(greedy_path)
    print(f"Loading MHCD-AE results: {mhcd_path}")
    mhcd_data = load_json(mhcd_path)

    if len(greedy_data) != len(mhcd_data):
        raise ValueError(f"Result length mismatch: greedy={len(greedy_data)}, mhcd={len(mhcd_data)}")

    greedy_questions = [item.get("question") for item in greedy_data]
    mhcd_questions = [item.get("question") for item in mhcd_data]
    if greedy_questions != mhcd_questions:
        raise ValueError("Question order mismatch between greedy and MHCD-AE results.")

    gt_list = [item.get("ground_truth") for item in greedy_data]

    print("-" * 50)
    greedy_metrics = get_metrics(greedy_data)
    print_metrics("GREEDY MODE (Baseline)", greedy_metrics)

    print("-" * 50)
    mhcd_metrics = get_metrics(mhcd_data, gt_list)
    print_metrics("MHCD-AE MODE (Ours)", mhcd_metrics)

    print("-" * 50)
    print("====== COMPARISON (MHCD-AE vs GREEDY) ======")
    for key in ("Accuracy", "Precision", "Recall", "F1", "Yes Ratio"):
        diff = mhcd_metrics[key] - greedy_metrics[key]
        sign = "+" if diff > 0 else ""
        print(f"{key:12s}: {sign}{diff:.2f}%")


if __name__ == "__main__":
    main()
