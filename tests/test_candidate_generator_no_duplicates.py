from __future__ import annotations

import unittest

from src.decoding.candidate_generator import EGMHCDGenerator


class FakeInnerGenerator:
    def __init__(self):
        self.calls = []

    def generate(self, image_path, question, num_candidates=5, raw_item=None):
        self.calls.append(
            {
                "image_path": image_path,
                "question": question,
                "num_candidates": num_candidates,
                "question_id": raw_item.get("question_id") if raw_item else None,
            }
        )
        return [{"text": f"candidate-{i}", "H_cd": 0.0, "D_vis": 0.0} for i in range(num_candidates)]


class EGMHCDNoDuplicateTest(unittest.TestCase):
    def test_egmhcd_generator_outputs_one_result_per_sample(self):
        generator = EGMHCDGenerator.__new__(EGMHCDGenerator)
        generator.num_candidates = 5
        generator.generator = FakeInnerGenerator()

        dataloader = []
        for i in range(3):
            dataloader.append(
                (
                    {},
                    [
                        {
                            "question": f"Question {i}?",
                            "image_path": f"/tmp/{i}.jpg",
                            "image_name": f"{i}.jpg",
                            "question_id": f"q{i}",
                            "ground_truth": "no",
                        }
                    ],
                )
            )

        results = generator.generate_candidates(dataloader)

        self.assertEqual(len(results), 3)
        self.assertEqual(len(generator.generator.calls), 3)
        for i, result in enumerate(results):
            self.assertEqual(result["question_id"], f"q{i}")
            self.assertEqual(len(result["candidates"]), 5)
            self.assertEqual(generator.generator.calls[i]["num_candidates"], 5)


if __name__ == "__main__":
    unittest.main()
