import argparse
import json
import os

from src.eval.pope_metrics import get_metrics, print_metrics


def resolve_path(path, fallback):
    if os.path.exists(path):
        return path
    if fallback and os.path.exists(fallback):
        return fallback
    return path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Evaluate greedy vs an EG-MHCD-AE yes/no result file.")
    parser.add_argument(
        "--greedy",
        default="results_coco_pope_adversarial_greedy.json",
        help="Path to greedy result JSON.",
    )
    parser.add_argument(
        "--candidate",
        "--mhcd",
        dest="candidate",
        default="results_coco_pope_adversarial_eg_label_cd.json",
        help="Path to candidate method result JSON.",
    )
    args = parser.parse_args()

    greedy_path = resolve_path(args.greedy, "results_coco_greedy.json")
    candidate_path = resolve_path(args.candidate, "results_coco_eg_label_cd.json")

    print(f"Loading greedy results: {greedy_path}")
    greedy_data = load_json(greedy_path)
    print(f"Loading candidate results: {candidate_path}")
    candidate_data = load_json(candidate_path)

    if len(greedy_data) != len(candidate_data):
        raise ValueError(f"Result length mismatch: greedy={len(greedy_data)}, candidate={len(candidate_data)}")

    greedy_questions = [item.get("question") for item in greedy_data]
    candidate_questions = [item.get("question") for item in candidate_data]
    if greedy_questions != candidate_questions:
        raise ValueError("Question order mismatch between greedy and candidate results.")

    gt_list = [item.get("ground_truth") for item in greedy_data]

    print("-" * 50)
    greedy_metrics = get_metrics(greedy_data)
    print_metrics("GREEDY MODE (Baseline)", greedy_metrics)

    print("-" * 50)
    candidate_metrics = get_metrics(candidate_data, gt_list)
    print_metrics("CANDIDATE MODE", candidate_metrics)

    print("-" * 50)
    print("====== COMPARISON (CANDIDATE vs GREEDY) ======")
    for key in (
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "Yes Ratio",
        "FPR",
        "TNR",
        "FNR",
        "Balanced Accuracy",
    ):
        diff = candidate_metrics[key] - greedy_metrics[key]
        sign = "+" if diff > 0 else ""
        print(f"{key:12s}: {sign}{diff:.2f}%")


if __name__ == "__main__":
    main()
