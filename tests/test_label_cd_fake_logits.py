from __future__ import annotations

import tempfile
import unittest

import torch
from PIL import Image

from src.decoding.label_cd import EntroGraphLabelCD


class FakeLogprobAdapter:
    def build_inputs(self, image_path, question):
        return {"branch": "pos", "image_path": image_path, "question": question}

    def build_inputs_from_pil(self, image, question):
        return {"branch": "neg", "question": question, "size": image.size}

    def move_to_device(self, inputs):
        return inputs

    def sequence_logprob(self, inputs, label_text):
        table = {
            "pos": {"Yes": -0.1, " yes": -2.0, "No": -1.4, " no": -3.0},
            "neg": {"Yes": -1.2, " yes": -3.0, "No": -0.2, " no": -2.0},
        }
        return torch.tensor([table[inputs["branch"]][label_text]], dtype=torch.float32)


class LabelCDFakeLogitsTest(unittest.TestCase):
    def test_fake_logits_produce_strict_yes_or_no_and_cd_scores(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        Image.new("RGB", (5, 7), color=(20, 40, 60)).save(tmp.name)

        scorer = EntroGraphLabelCD(
            FakeLogprobAdapter(),
            alpha0=0.5,
            alpha_max=2.0,
            k_entropy=0.0,
            neg_type="gray",
        )

        result = scorer.predict(tmp.name, "Is there an object?")

        self.assertEqual(result.pred, "yes")
        self.assertIn(result.pred, ("yes", "no"))
        self.assertEqual(set(result.scores_pos), {"yes", "no"})
        self.assertEqual(set(result.scores_neg), {"yes", "no"})
        self.assertEqual(set(result.scores_cd), {"yes", "no"})
        self.assertAlmostEqual(
            result.scores_cd["yes"],
            (1.0 + result.alpha) * result.scores_pos["yes"] - result.alpha * result.scores_neg["yes"],
            places=6,
        )
        self.assertAlmostEqual(
            result.scores_cd["no"],
            (1.0 + result.alpha) * result.scores_pos["no"] - result.alpha * result.scores_neg["no"],
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
