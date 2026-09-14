#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from statistics import mean

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.eval.caption_metrics import word_count


def summarize_chair(input_dir: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(input_dir, "metrics_chair_*.json"))):
        method = os.path.basename(path)[len("metrics_chair_") : -len(".json")]
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        metrics = data.get("metrics", data)
        rows.append(
            {
                "task": "chair",
                "method": metrics.get("method", method),
                "N": metrics.get("N"),
                "CHAIRs": metrics.get("CHAIRs"),
                "CHAIRi": metrics.get("CHAIRi"),
                "Recall": metrics.get("Recall"),
                "Precision": metrics.get("Precision"),
                "F1": metrics.get("F1"),
                "AvgLen": metrics.get("AvgLen"),
                "ObjMentioned": metrics.get("ObjMentioned"),
            }
        )
    return rows


def summarize_mmhal(input_dir: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(input_dir, "results_mmhal_*.json"))):
        method = os.path.basename(path)[len("results_mmhal_") : -len(".json")]
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        lengths = [word_count(item.get("best_answer", "")) for item in data]
        risk_values = [item.get("risk_high") for item in data if item.get("risk_high") is not None]
        h_cluster = [float(item.get("H_cluster")) for item in data if item.get("H_cluster") is not None]
        ae_values = []
        for item in data:
            for score in item.get("candidate_scores", []) or []:
                if score.get("AE") is not None:
                    ae_values.append(float(score["AE"]))
        rows.append(
            {
                "task": "mmhal",
                "method": method,
                "N": len(data),
                "AvgLen": mean(lengths) if lengths else 0.0,
                "risk_high_ratio": mean([1.0 if value else 0.0 for value in risk_values]) if risk_values else "",
                "H_cluster_mean": mean(h_cluster) if h_cluster else "",
                "AE_mean": mean(ae_values) if ae_values else "",
            }
        )
    return rows


def write_rows(rows: list[dict], output: str) -> None:
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved summary: {output}")
    for row in rows:
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["chair", "mmhal"], required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = summarize_chair(args.input_dir) if args.task == "chair" else summarize_mmhal(args.input_dir)
    write_rows(rows, args.output)


if __name__ == "__main__":
    main()
