from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from src.decoding.token_cd import contrastive_generate


class TinyTokenizer:
    eos_token_id = None

    def decode(self, token_ids, skip_special_tokens=False):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return "".join(str(int(token_id)) for token_id in token_ids)


class StaticLogitModel:
    def __init__(self, pos_values, neg_values=None):
        self.pos_values = pos_values
        self.neg_values = neg_values if neg_values is not None else pos_values

    def eval(self):
        return self

    def __call__(self, **inputs):
        branch = inputs["branch"]
        logits = torch.full((1, 1, 8), -100.0)
        values = self.pos_values if branch == "pos" else self.neg_values
        for token_id, value in values.items():
            logits[:, -1, token_id] = float(value)
        return SimpleNamespace(logits=logits)


def make_inputs(branch):
    return {
        "branch": branch,
        "input_ids": torch.tensor([[0]], dtype=torch.long),
        "attention_mask": torch.tensor([[1]], dtype=torch.long),
    }


class TokenCDMetricsTest(unittest.TestCase):
    def assert_metric_ranges(self, result):
        self.assertGreaterEqual(result.H_cd, 0.0)
        self.assertLessEqual(result.H_cd, 1.0)
        self.assertGreaterEqual(result.D_vis_norm, 0.0)
        self.assertLessEqual(result.D_vis_norm, 1.0)

    def test_h_cd_is_low_for_near_one_hot_cd_distribution(self):
        model = StaticLogitModel({3: 80.0, 4: -80.0})

        result = contrastive_generate(
            model,
            processor=None,
            inputs_pos=make_inputs("pos"),
            inputs_neg=make_inputs("neg"),
            tokenizer=TinyTokenizer(),
            max_new_tokens=1,
            beta=0.5,
            top_p=1.0,
            topk_plausible=2,
        )

        self.assert_metric_ranges(result)
        self.assertLess(result.trace[0].H_cd_t, 1e-4)
        self.assertLess(result.H_cd, 1e-4)
        self.assertAlmostEqual(result.S_graph, result.risk_graph)
        self.assertAlmostEqual(result.grounding_score, result.D_vis_norm - result.H_cd)

    def test_h_cd_uses_cd_distribution_not_positive_distribution(self):
        model = StaticLogitModel(
            pos_values={3: 5.0, 4: 5.0},
            neg_values={3: -5.0, 4: 5.0},
        )

        result = contrastive_generate(
            model,
            processor=None,
            inputs_pos=make_inputs("pos"),
            inputs_neg=make_inputs("neg"),
            tokenizer=TinyTokenizer(),
            max_new_tokens=1,
            beta=1.0,
            top_p=1.0,
            topk_plausible=2,
        )

        self.assert_metric_ranges(result)
        self.assertGreater(result.trace[0].H_pos_t, 0.65)
        self.assertLess(result.trace[0].H_cd_t, 1e-3)
        self.assertLess(result.H_cd, 1e-3)

    def test_h_cd_is_high_for_uniform_cd_distribution_and_risk_sign(self):
        model = StaticLogitModel({3: 5.0, 4: 5.0})

        result = contrastive_generate(
            model,
            processor=None,
            inputs_pos=make_inputs("pos"),
            inputs_neg=make_inputs("neg"),
            tokenizer=TinyTokenizer(),
            max_new_tokens=1,
            beta=0.5,
            top_p=1.0,
            topk_plausible=2,
        )

        self.assert_metric_ranges(result)
        self.assertAlmostEqual(result.trace[0].H_cd_t, 1.0, places=5)
        self.assertAlmostEqual(result.H_cd, 1.0, places=5)
        self.assertAlmostEqual(result.D_vis_norm, 0.0, places=6)
        self.assertAlmostEqual(result.risk_graph, result.H_cd - result.D_vis_norm, places=6)
        self.assertAlmostEqual(result.grounding_score, result.D_vis_norm - result.H_cd, places=6)
        self.assertAlmostEqual(result.S_graph, result.risk_graph, places=6)


if __name__ == "__main__":
    unittest.main()
