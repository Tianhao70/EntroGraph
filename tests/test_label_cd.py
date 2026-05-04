from __future__ import annotations

import math
import tempfile
import unittest

import torch
from PIL import Image

from src.decoding.label_cd import EntroGraphLabelCD, LabelCDResult


class FakeAdapter:
    def __init__(self):
        self.negative_image = None

    def build_inputs(self, image_path, question):
        return {"branch": "pos", "image_path": image_path, "question": question}

    def build_inputs_from_pil(self, image, question):
        self.negative_image = image
        return {"branch": "neg", "question": question}

    def move_to_device(self, inputs):
        return inputs

    def sequence_logprob(self, inputs, label_text):
        scores = {
            "pos": {
                "Yes": 2.0,
                " yes": 1.0,
                "No": 0.0,
                " no": -1.0,
            },
            "neg": {
                "Yes": 0.0,
                " yes": -1.0,
                "No": 2.0,
                " no": 1.0,
            },
        }
        return torch.tensor([scores[inputs["branch"]][label_text]], dtype=torch.float32)


class EntroGraphLabelCDTest(unittest.TestCase):
    def make_image_path(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        Image.new("RGB", (8, 6), color=(120, 80, 40)).save(tmp.name)
        return tmp.name

    def test_predict_returns_yes_no_and_scores(self):
        adapter = FakeAdapter()
        predictor = EntroGraphLabelCD(
            adapter,
            alpha0=0.5,
            alpha_max=2.0,
            k_entropy=0.8,
            eta=1.0,
            neg_type="gray",
            neg_std=0.2,
        )

        result = predictor.predict(self.make_image_path(), "Is there a cat?")

        self.assertIsInstance(result, LabelCDResult)
        self.assertIn(result.pred, ("yes", "no"))
        self.assertEqual(result.pred, "yes")
        self.assertEqual(result.neg_type, "gray")
        self.assertEqual(adapter.negative_image.mode, "RGB")
        self.assertEqual(adapter.negative_image.size, (8, 6))
        self.assertAlmostEqual(sum(result.p_pos.values()), 1.0, places=6)
        self.assertAlmostEqual(sum(result.p_neg.values()), 1.0, places=6)
        self.assertGreaterEqual(result.H_label, 0.0)
        self.assertGreaterEqual(result.JS_label, 0.0)
        self.assertGreaterEqual(result.alpha, 0.0)
        self.assertLessEqual(result.alpha, 2.0)
        self.assertEqual(set(result.scores_pos), {"yes", "no"})
        self.assertEqual(set(result.scores_neg), {"yes", "no"})
        self.assertEqual(set(result.scores_cd), {"yes", "no"})

    def test_alpha_formula_uses_entropy_and_clamps(self):
        predictor = EntroGraphLabelCD(
            FakeAdapter(),
            alpha0=0.5,
            alpha_max=0.75,
            k_entropy=10.0,
            neg_type="gray",
        )

        result = predictor.predict(self.make_image_path(), "Question?")

        unclamped = 0.5 + 10.0 * result.H_label / math.log(2)
        self.assertGreater(unclamped, 0.75)
        self.assertEqual(result.alpha, 0.75)

    def test_constant_alpha_when_k_entropy_zero(self):
        predictor = EntroGraphLabelCD(
            FakeAdapter(),
            alpha0=0.3,
            alpha_max=2.0,
            k_entropy=0.0,
            neg_type="gray",
        )

        result = predictor.predict(self.make_image_path(), "Question?")

        self.assertAlmostEqual(result.alpha, 0.3)


if __name__ == "__main__":
    unittest.main()
