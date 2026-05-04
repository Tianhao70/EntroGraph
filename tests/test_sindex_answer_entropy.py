from __future__ import annotations

import unittest

import numpy as np

from src.scoring.sindex_answer_entropy import CandidateScore, EGAnswerEntropyScorer, RerankResult


class FakeEncoder:
    def __init__(self, vectors):
        self.vectors = {key: np.asarray(value, dtype=float) for key, value in vectors.items()}

    def encode(self, texts, normalize_embeddings=True):
        rows = []
        for text in texts:
            row = self.vectors[str(text)]
            if normalize_embeddings:
                row = row / max(float(np.linalg.norm(row)), 1e-12)
            rows.append(row)
        return np.asarray(rows, dtype=float)


class EGAnswerEntropyScorerTest(unittest.TestCase):
    def test_score_and_select_returns_structured_min_ae_result(self):
        scorer = EGAnswerEntropyScorer(device="cpu", distance_threshold=0.15, tau=0.1)
        scorer.encoder = FakeEncoder(
            {
                "What is shown? [SEP] red apple": [1.0, 0.0, 0.0],
                "What is shown? [SEP] red fruit": [0.99, 0.10, 0.0],
                "What is shown? [SEP] blue car": [0.0, 1.0, 0.0],
            }
        )
        candidates = [
            {"text": "red apple", "H_cd": 0.05, "D_vis": 0.9, "avg_logprob_cd": -0.1},
            {"text": "red fruit", "H_cd": 0.50, "D_vis": 0.1, "avg_logprob_cd": -1.0},
            {"text": "blue car", "H_cd": 0.70, "D_vis": 0.1, "avg_logprob_cd": -0.5},
        ]

        result = scorer.score_and_select("What is shown?", candidates)

        self.assertIsInstance(result, RerankResult)
        self.assertEqual(result.best_index, 0)
        self.assertEqual(result.best_text, "red apple")
        self.assertEqual(result.clusters, [0, 0, 1])
        self.assertEqual(result.mode, "low_cluster_entropy")
        self.assertLess(result.H_cluster, 0.25)
        self.assertGreater(result.delta_AE, 0.0)
        self.assertEqual(len(result.scores), 3)
        self.assertIsInstance(result.scores[0], CandidateScore)
        self.assertAlmostEqual(result.scores[0].H_vis, 0.0)
        self.assertAlmostEqual(result.scores[1].H_vis, 1.0 - (0.1 / np.log(2.0)))
        self.assertAlmostEqual(result.scores[0].avg_logprob_cd, -0.1)
        self.assertAlmostEqual(result.scores[0].avg_logprob_norm, 1.0)
        self.assertAlmostEqual(result.scores[1].avg_logprob_norm, 0.0)
        self.assertEqual(result.embedding_mode, "question_answer")
        self.assertEqual(scorer.last_mode, result.mode)
        self.assertEqual(scorer.last_ae_scores, [score.AE for score in result.scores])

    def test_high_uncertainty_rule_marks_risk_high(self):
        scorer = EGAnswerEntropyScorer(device="cpu", distance_threshold=0.05, tau=0.1)
        scorer.encoder = FakeEncoder(
            {
                "Pick one. [SEP] alpha": [1.0, 0.0, 0.0],
                "Pick one. [SEP] beta": [0.0, 1.0, 0.0],
                "Pick one. [SEP] gamma": [0.0, 0.0, 1.0],
            }
        )
        candidates = [
            {"text": "alpha", "H_cd": 0.1, "D_vis": 0.2, "avg_logprob_cd": -0.2},
            {"text": "beta", "H_cd": 0.1, "D_vis": 0.2, "avg_logprob_cd": -0.2},
            {"text": "gamma", "H_cd": 0.1, "D_vis": 0.2, "avg_logprob_cd": -0.2},
        ]

        result = scorer.score_and_select("Pick one.", candidates)

        self.assertEqual(result.mode, "high_uncertainty")
        self.assertTrue(result.risk_high)
        self.assertGreater(result.H_cluster, 0.65)
        self.assertLess(result.delta_AE, 0.10)
        self.assertEqual(scorer.last_risk_high, True)


if __name__ == "__main__":
    unittest.main()
