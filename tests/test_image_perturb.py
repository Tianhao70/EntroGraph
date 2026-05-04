from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from src.perturbations.image_perturb import perturb_image_pil


class ImagePerturbTest(unittest.TestCase):
    def make_image(self, mode="RGB"):
        arr = np.zeros((6, 8, 3), dtype=np.uint8)
        arr[..., 0] = 64
        arr[..., 1] = 128
        arr[..., 2] = 192
        image = Image.fromarray(arr).convert("RGB")
        return image.convert(mode)

    def test_modes_return_rgb_same_size(self):
        image = self.make_image(mode="RGBA")

        for mode in ("gaussian", "blur", "gray"):
            with self.subTest(mode=mode):
                out = perturb_image_pil(image, mode=mode, std=0.1)
                self.assertEqual(out.mode, "RGB")
                self.assertEqual(out.size, image.size)

    def test_gaussian_clamps_to_valid_uint8_range(self):
        np.random.seed(0)
        image = Image.new("RGB", (10, 10), color=(255, 0, 127))

        out = perturb_image_pil(image, mode="gaussian", std=5.0)
        arr = np.asarray(out)

        self.assertEqual(arr.dtype, np.uint8)
        self.assertGreaterEqual(int(arr.min()), 0)
        self.assertLessEqual(int(arr.max()), 255)

    def test_gaussian_zero_std_keeps_rgb_pixels(self):
        image = self.make_image(mode="RGB")

        out = perturb_image_pil(image, mode="gaussian", std=0.0)

        self.assertTrue(np.array_equal(np.asarray(out), np.asarray(image)))

    def test_gaussian_seed_is_reproducible(self):
        image = self.make_image(mode="RGB")

        out_a = perturb_image_pil(image, mode="gaussian", std=0.2, seed=123)
        out_b = perturb_image_pil(image, mode="gaussian", std=0.2, seed=123)
        out_c = perturb_image_pil(image, mode="gaussian", std=0.2, seed=456)

        self.assertTrue(np.array_equal(np.asarray(out_a), np.asarray(out_b)))
        self.assertFalse(np.array_equal(np.asarray(out_a), np.asarray(out_c)))

    def test_gray_output_has_equal_channels(self):
        image = self.make_image(mode="RGB")

        out = perturb_image_pil(image, mode="gray")
        arr = np.asarray(out)

        self.assertTrue(np.array_equal(arr[..., 0], arr[..., 1]))
        self.assertTrue(np.array_equal(arr[..., 1], arr[..., 2]))

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            perturb_image_pil(self.make_image(), mode="unknown")


if __name__ == "__main__":
    unittest.main()
