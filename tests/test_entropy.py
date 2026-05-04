from __future__ import annotations

import math
import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - depends on local ML env
    torch = None

from src.entrograph.entropy import (
    entropy_from_probs,
    js_div,
    safe_softmax,
    topk_plausibility_mask,
)


@unittest.skipIf(torch is None, "torch is not installed")
class EntropyUtilsTest(unittest.TestCase):
    def test_entropy_supports_batch_dimension(self):
        probs = torch.tensor(
            [
                [0.5, 0.5],
                [1.0, 0.0],
            ],
            dtype=torch.float32,
        )

        entropy = entropy_from_probs(probs)

        self.assertEqual(tuple(entropy.shape), (2,))
        self.assertAlmostEqual(float(entropy[0]), math.log(2), places=5)
        self.assertLess(float(entropy[1]), 1e-5)
        self.assertTrue(torch.isfinite(entropy).all())

    def test_js_divergence_is_symmetric_and_finite(self):
        p = torch.tensor([[0.9, 0.1], [0.5, 0.5]], dtype=torch.float32)
        q = torch.tensor([[0.1, 0.9], [0.5, 0.5]], dtype=torch.float32)

        js_pq = js_div(p, q)
        js_qp = js_div(q, p)
        js_same = js_div(p, p)

        self.assertTrue(torch.allclose(js_pq, js_qp, atol=1e-6))
        self.assertTrue(torch.isfinite(js_pq).all())
        self.assertTrue(torch.all(js_pq >= 0))
        self.assertTrue(torch.allclose(js_same, torch.zeros_like(js_same), atol=1e-6))

    def test_topk_plausibility_mask_uses_positive_logits(self):
        logits_pos = torch.tensor(
            [
                [0.1, 3.0, 2.0, 1.0],
                [4.0, 1.0, 2.0, 3.0],
            ],
            dtype=torch.float32,
        )
        logits_cd = torch.tensor(
            [
                [10.0, 11.0, 12.0, 13.0],
                [20.0, 21.0, 22.0, 23.0],
            ],
            dtype=torch.float32,
        )

        masked = topk_plausibility_mask(logits_pos, logits_cd, k=2)
        masked_value = torch.finfo(masked.dtype).min

        self.assertTrue(torch.isfinite(masked).all())
        self.assertEqual(float(masked[0, 1]), 11.0)
        self.assertEqual(float(masked[0, 2]), 12.0)
        self.assertEqual(float(masked[1, 0]), 20.0)
        self.assertEqual(float(masked[1, 3]), 23.0)
        self.assertEqual(float(masked[0, 0]), masked_value)
        self.assertEqual(float(masked[0, 3]), masked_value)
        self.assertEqual(float(masked[1, 1]), masked_value)
        self.assertEqual(float(masked[1, 2]), masked_value)

    def test_safe_softmax_handles_nonfinite_logits(self):
        logits = torch.tensor(
            [
                [float("-inf"), float("-inf")],
                [float("inf"), float("inf")],
                [0.0, float("nan")],
            ],
            dtype=torch.float32,
        )

        probs = safe_softmax(logits)

        self.assertTrue(torch.isfinite(probs).all())
        self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.ones(3), atol=1e-6))
        self.assertTrue(torch.allclose(probs[1], torch.tensor([0.5, 0.5]), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
