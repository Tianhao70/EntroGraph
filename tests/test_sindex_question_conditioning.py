from __future__ import annotations

import unittest

import numpy as np

from src.scoring.sindex_answer_entropy import EGAnswerEntropyScorer


class SINDexQuestionConditioningTest(unittest.TestCase):
    def test_embedding_inputs_include_question_sep_answer(self):
        scorer = EGAnswerEntropyScorer(device="cpu")
        captured = []

        def fake_encode(texts):
            captured.extend(texts)
            return np.eye(len(texts), dtype=float)

        scorer._encode_texts = fake_encode
        candidates = [
            {"text": "yes", "H_cd": 0.1, "D_vis": 0.2, "avg_logprob_cd": -0.1},
            {"text": "no", "H_cd": 0.2, "D_vis": 0.3, "avg_logprob_cd": -0.2},
        ]

        scorer.score_and_select("Is there a dog?", candidates)

        self.assertEqual(captured, ["Is there a dog? [SEP] yes", "Is there a dog? [SEP] no"])


if __name__ == "__main__":
    unittest.main()
