from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace

import torch
from PIL import Image

from src.decoding.candidate_generator import ContrastiveCandidateGenerator


class FakeTokenizer:
    eos_token_id = 9

    def decode(self, token_ids, skip_special_tokens=False):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        vocab = {3: "A", 4: "B", 9: "<eos>"}
        return "".join(vocab.get(int(token_id), f"<{token_id}>") for token_id in token_ids)


class BetaSensitiveModel:
    def eval(self):
        return self

    def __call__(self, **inputs):
        branch = inputs["branch"]
        logits = torch.full((1, 1, 10), -20.0)
        if branch == "pos":
            logits[:, -1, 3] = 5.0
            logits[:, -1, 4] = 4.0
        else:
            logits[:, -1, 3] = 4.0
            logits[:, -1, 4] = 0.0
        return SimpleNamespace(logits=logits)


class FakeAdapter:
    def __init__(self):
        self.model = BetaSensitiveModel()
        self.processor = object()
        self.tokenizer = FakeTokenizer()
        self.negative_image_size = None

    def build_inputs(self, image_path: str, question: str):
        return self._inputs("pos")

    def build_inputs_from_pil(self, image: Image.Image, question: str):
        self.negative_image_size = image.size
        return self._inputs("neg")

    def move_to_device(self, inputs):
        return inputs

    def _inputs(self, branch: str):
        return {
            "branch": branch,
            "input_ids": torch.tensor([[0, 0]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        }


class CandidateGeneratorTest(unittest.TestCase):
    def test_candidate_beta_enters_token_cd_logits(self):
        adapter = FakeAdapter()
        generator = ContrastiveCandidateGenerator(
            adapter=adapter,
            max_new_tokens=1,
            top_p=1.0,
            topk_plausible=2,
            neg_type="gray",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, "sample.png")
            Image.new("RGB", (6, 4), color=(128, 64, 32)).save(image_path)
            candidates = generator.generate(image_path, "Is there an object?", num_candidates=5)

        self.assertEqual(len(candidates), 5)
        self.assertEqual(adapter.negative_image_size, (6, 4))
        self.assertEqual([cand["config"]["beta"] for cand in candidates], [0.20, 0.35, 0.50, 0.65, 0.80])
        self.assertEqual(candidates[0]["text"], "A")
        self.assertEqual(candidates[0]["token_ids"], [3])
        self.assertEqual(candidates[1]["text"], "B")
        self.assertEqual(candidates[1]["token_ids"], [4])
        self.assertEqual(candidates[0]["trace"][0]["alpha_t"], 0.20)
        self.assertEqual(candidates[1]["trace"][0]["alpha_t"], 0.35)


if __name__ == "__main__":
    unittest.main()
