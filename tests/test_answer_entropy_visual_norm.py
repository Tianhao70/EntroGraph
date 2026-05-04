from __future__ import annotations

import unittest

import numpy as np

from src.scoring.sindex_answer_entropy import EGAnswerEntropyScorer


class AnswerEntropyVisualNormTest(unittest.TestCase):
    def make_scorer(self):
        scorer = EGAnswerEntropyScorer(device="cpu", distance_threshold=0.05)
        scorer._encode_texts = lambda texts: np.eye(len(texts), dtype=float)
        return scorer

    def test_equal_high_d_vis_uses_absolute_norm_not_relative_one(self):
        scorer = self.make_scorer()
        candidates = [
            {"text": "a", "H_cd": 0.1, "D_vis": 0.69, "avg_logprob_cd": -0.1},
            {"text": "b", "H_cd": 0.1, "D_vis": 0.69, "avg_logprob_cd": -0.1},
            {"text": "c", "H_cd": 0.1, "D_vis": 0.69, "avg_logprob_cd": -0.1},
        ]

        result = scorer.score_and_select("Question?", candidates)

        for score in result.scores:
            self.assertLess(score.H_vis_abs, 0.01)
            self.assertLess(score.H_vis, 0.01)
            self.assertAlmostEqual(score.H_vis_rel, 1.0)

    def test_equal_low_d_vis_stays_visually_uncertain(self):
        scorer = self.make_scorer()
        candidates = [
            {"text": "a", "H_cd": 0.1, "D_vis": 0.01, "avg_logprob_cd": -0.1},
            {"text": "b", "H_cd": 0.1, "D_vis": 0.01, "avg_logprob_cd": -0.1},
            {"text": "c", "H_cd": 0.1, "D_vis": 0.01, "avg_logprob_cd": -0.1},
        ]

        result = scorer.score_and_select("Question?", candidates)

        for score in result.scores:
            self.assertGreater(score.H_vis_abs, 0.98)
            self.assertGreater(score.H_vis, 0.98)
            self.assertAlmostEqual(score.H_vis_rel, 1.0)


if __name__ == "__main__":
    unittest.main()
