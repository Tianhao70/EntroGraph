from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def convert_to_mmhal_response(
    input_json: str,
    output_json: str,
    template_json: str | None = None,
) -> list[dict[str, Any]]:
    with open(input_json, "r", encoding="utf-8") as f:
        results = json.load(f)

    template_records = None
    if template_json:
        with open(template_json, "r", encoding="utf-8") as f:
            template_records = json.load(f)

    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(results):
        answer = item.get("best_answer", item.get("caption", ""))
        if template_records is not None and idx < len(template_records):
            row = dict(template_records[idx])
            row["question_id"] = item.get("question_id", row.get("question_id", str(idx)))
            row["model_answer"] = answer
        else:
            row = {
                "question_id": item.get("question_id", str(idx)),
                "image_id": item.get("image_id"),
                "image_content": item.get("image_content", []),
                "question": item.get("question", ""),
                "gt_answer": item.get("gt_answer", item.get("ground_truth", "")),
                "model_answer": answer,
            }
            if item.get("question_type") is not None:
                row["question_type"] = item["question_type"]
            if item.get("question_topic") is not None:
                row["question_topic"] = item["question_topic"]
        rows.append(row)

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    return rows


def validate_mmhal_response(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    bad_indices = []
    for idx, row in enumerate(rows):
        if not str(row.get("model_answer", "")).strip():
            bad_indices.append(idx)
    return {
        "path": path,
        "N": len(rows),
        "bad_indices": bad_indices,
        "valid": not bad_indices,
    }

