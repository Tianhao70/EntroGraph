from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from src.perturbations.image_perturb import perturb_image_pil


class SeedNumpyTest(unittest.TestCase):
    def test_gaussian_perturbation_reproducible_with_same_seed(self):
        image = Image.new("RGB", (8, 8), color=(128, 128, 128))

        out_a = perturb_image_pil(image, mode="gaussian", std=0.2, seed=123)
        out_b = perturb_image_pil(image, mode="gaussian", std=0.2, seed=123)

        self.assertTrue(np.array_equal(np.asarray(out_a), np.asarray(out_b)))

    def test_gaussian_perturbation_changes_with_different_seed(self):
        image = Image.new("RGB", (8, 8), color=(128, 128, 128))

        out_a = perturb_image_pil(image, mode="gaussian", std=0.2, seed=123)
        out_b = perturb_image_pil(image, mode="gaussian", std=0.2, seed=456)

        self.assertFalse(np.array_equal(np.asarray(out_a), np.asarray(out_b)))


if __name__ == "__main__":
    unittest.main()
