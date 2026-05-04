from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from src.decoding.token_cd import StepTrace, TokenCDResult, contrastive_generate


class FakeTokenizer:
    eos_token_id = 9

    def __init__(self):
        self.vocab = {
            0: "<pad>",
            3: "A",
            4: "B",
            7: "C",
            9: "<eos>",
        }

    def decode(self, token_ids, skip_special_tokens=False):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        pieces = []
        for token_id in token_ids:
            if skip_special_tokens and token_id == self.eos_token_id:
                continue
            pieces.append(self.vocab.get(int(token_id), f"<{token_id}>"))
        return "".join(pieces)


class FakeCDModel:
    def __init__(self):
        self.calls = []
        self.generate_called = False

    def generate(self, *args, **kwargs):
        self.generate_called = True
        raise AssertionError("standard generate must not be called")

    def eval(self):
        return self

    def __call__(self, **inputs):
        branch = inputs["branch"]
        input_ids = inputs["input_ids"]
        self.calls.append((branch, input_ids.clone()))
        batch_size = input_ids.shape[0]
        logits = torch.full((batch_size, 1, 10), -20.0)
        step = input_ids.shape[1] - 2

        if step == 0:
            if branch == "pos":
                logits[:, -1, 3] = 5.0
                logits[:, -1, 4] = 4.0
                logits[:, -1, 9] = 0.0
            else:
                logits[:, -1, 4] = 6.0
                logits[:, -1, 3] = 1.0
                logits[:, -1, 9] = 0.0
        else:
            if branch == "pos":
                logits[:, -1, 9] = 6.0
                logits[:, -1, 7] = 1.0
            else:
                logits[:, -1, 7] = 6.0
                logits[:, -1, 9] = 1.0

        return SimpleNamespace(logits=logits)


class TokenCDTest(unittest.TestCase):
    def make_inputs(self, branch):
        return {
            "branch": branch,
            "input_ids": torch.tensor([[0, 0]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        }

    def test_contrastive_generate_is_true_token_cd_and_stops_on_eos(self):
        model = FakeCDModel()
        tokenizer = FakeTokenizer()

        result = contrastive_generate(
            model,
            processor=None,
            inputs_pos=self.make_inputs("pos"),
            inputs_neg=self.make_inputs("neg"),
            tokenizer=tokenizer,
            max_new_tokens=5,
            beta=1.0,
            temperature=1.0,
            top_p=1.0,
            topk_plausible=3,
        )

        self.assertIsInstance(result, TokenCDResult)
        self.assertFalse(model.generate_called)
        self.assertEqual(result.token_ids, [3, 9])
        self.assertEqual(result.text, "A")
        self.assertEqual(len(result.trace), 2)
        self.assertIsInstance(result.trace[0], StepTrace)
        self.assertEqual(result.trace[0].token_text, "A")
        self.assertEqual(result.trace[0].alpha_t, 1.0)
        self.assertGreaterEqual(result.H_cd, 0.0)
        self.assertGreaterEqual(result.D_vis, 0.0)
        self.assertLessEqual(result.avg_logprob_cd, 0.0)

        # Both branches must receive the same generated prefix.
        self.assertEqual(model.calls[0][0], "pos")
        self.assertEqual(model.calls[1][0], "neg")
        self.assertEqual(model.calls[2][0], "pos")
        self.assertEqual(model.calls[3][0], "neg")
        self.assertEqual(model.calls[2][1].tolist(), [[0, 0, 3]])
        self.assertEqual(model.calls[3][1].tolist(), [[0, 0, 3]])

    def test_dynamic_alpha_uses_entropy(self):
        model = FakeCDModel()
        tokenizer = FakeTokenizer()

        result = contrastive_generate(
            model,
            processor=None,
            inputs_pos=self.make_inputs("pos"),
            inputs_neg=self.make_inputs("neg"),
            tokenizer=tokenizer,
            max_new_tokens=1,
            beta=0.1,
            temperature=1.0,
            top_p=1.0,
            topk_plausible=3,
            dynamic_alpha=True,
            alpha0=0.5,
            k_entropy=0.8,
            alpha_max=2.0,
        )

        self.assertGreaterEqual(result.trace[0].alpha_t, 0.5)
        self.assertLessEqual(result.trace[0].alpha_t, 2.0)
        self.assertNotEqual(result.trace[0].alpha_t, 0.1)
        self.assertTrue(result.trace[0].topk_pos)
        self.assertTrue(result.trace[0].topk_cd)

    def test_topk_plausibility_blocks_implausible_cd_winner(self):
        model = FakeCDModel()
        tokenizer = FakeTokenizer()

        result = contrastive_generate(
            model,
            processor=None,
            inputs_pos=self.make_inputs("pos"),
            inputs_neg=self.make_inputs("neg"),
            tokenizer=tokenizer,
            max_new_tokens=1,
            beta=10.0,
            temperature=1.0,
            top_p=1.0,
            topk_plausible=1,
        )

        # Only token A is in the positive top-1 set, so CD cannot choose B.
        self.assertEqual(result.token_ids, [3])


if __name__ == "__main__":
    unittest.main()
