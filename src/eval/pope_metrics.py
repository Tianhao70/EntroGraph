from __future__ import annotations

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
    fpr = false_pos / (false_pos + true_neg) if (false_pos + true_neg) > 0 else 0
    tnr = true_neg / (true_neg + false_pos) if (true_neg + false_pos) > 0 else 0
    fnr = false_neg / (false_neg + true_pos) if (false_neg + true_pos) > 0 else 0
    balanced_accuracy = 0.5 * (recall + tnr)
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
        "FPR": fpr * 100,
        "TNR": tnr * 100,
        "Specificity": tnr * 100,
        "FNR": fnr * 100,
        "Balanced Accuracy": balanced_accuracy * 100,
        "Confusion": confusion,
    }


def print_metrics(title, metrics):
    print(f"====== {title} ======")
    for key in ("N", "TP", "TN", "FP", "FN", "Unknown Pred", "Unknown GT"):
        print(f"{key:12s}: {metrics[key]}")
    for key in ("Accuracy", "Precision", "Recall", "F1", "Yes Ratio", "FPR", "FNR"):
        print(f"{key:12s}: {metrics[key]:.2f}%")
    print(f"{'TNR / Specificity':17s}: {metrics['TNR']:.2f}%")
    print(f"{'Balanced Accuracy':17s}: {metrics['Balanced Accuracy']:.2f}%")
