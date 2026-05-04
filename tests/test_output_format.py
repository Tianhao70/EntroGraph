from __future__ import annotations

import json
import os
import tempfile
import unittest

from main import build_result_base, write_trace_file


class OutputFormatTest(unittest.TestCase):
    def test_build_result_base_uses_required_fields(self):
        item = {
            "question_id": "q1",
            "image_name": "COCO_val2014_000000000001.jpg",
            "image_path": "/data/coco/val2014/COCO_val2014_000000000001.jpg",
            "question": "Is there a dog?",
            "ground_truth": "no",
        }

        result = build_result_base(item, "eg_label_cd", "no")

        self.assertEqual(
            list(result.keys()),
            [
                "question_id",
                "image_name",
                "image_path",
                "question",
                "ground_truth",
                "method",
                "best_answer",
            ],
        )
        self.assertEqual(result["method"], "eg_label_cd")
        self.assertEqual(result["best_answer"], "no")

    def test_write_trace_file_uses_method_dataset_question_layout(self):
        report = [
            {
                "question_id": "q/1",
                "question": "Is there a dog?",
                "image_name": "sample.jpg",
                "generation": {
                    "trace": [
                        {"t": 0, "token_id": 3, "token_text": "yes"},
                    ]
                },
                "candidate_details": [
                    {
                        "path_id": 2,
                        "config": {"beta": 0.5},
                        "trace": [
                            {"t": 0, "token_id": 4, "token_text": "no"},
                        ],
                    }
                ],
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = write_trace_file(report, tmpdir, "coco_pope_random", "eg_mhcd_ae")

            self.assertEqual(len(paths), 1)
            expected_path = os.path.join(tmpdir, "eg_mhcd_ae", "coco_pope_random", "q_1.jsonl")
            self.assertEqual(paths[0], expected_path)
            with open(expected_path, "r", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f]

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["kind"], "generation")
        self.assertEqual(rows[0]["token_text"], "yes")
        self.assertEqual(rows[1]["kind"], "candidate")
        self.assertEqual(rows[1]["path_id"], 2)
        self.assertEqual(rows[1]["config"], {"beta": 0.5})
        self.assertEqual(rows[1]["token_text"], "no")


if __name__ == "__main__":
    unittest.main()
