from __future__ import annotations

import json
import os
import tempfile
import unittest

from PIL import Image

from main import load_dataset_from_path


class DatasetLoadingSplitsTest(unittest.TestCase):
    def make_split_fixture(self):
        tmpdir = tempfile.TemporaryDirectory()
        dataset_dir = os.path.join(tmpdir.name, "pope")
        image_root = os.path.join(tmpdir.name, "images")
        os.makedirs(dataset_dir)
        os.makedirs(image_root)

        for split in ("random", "popular", "adversarial"):
            image_name = f"{split}.jpg"
            Image.new("RGB", (4, 4), color=(10, 20, 30)).save(os.path.join(image_root, image_name))
            row = {
                "image": image_name,
                "text": f"Is this {split}?",
                "label": "no",
                "question_id": split,
            }
            with open(os.path.join(dataset_dir, f"coco_pope_{split}.jsonl"), "w", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

        return tmpdir, dataset_dir, image_root

    def test_split_all_loads_random_popular_adversarial(self):
        tmpdir, dataset_dir, image_root = self.make_split_fixture()
        with tmpdir:
            rows = load_dataset_from_path(dataset_dir, image_root, split="all")

        self.assertEqual(len(rows), 3)
        self.assertEqual({row["question_id"] for row in rows}, {"random", "popular", "adversarial"})

    def test_split_adversarial_loads_only_adversarial(self):
        tmpdir, dataset_dir, image_root = self.make_split_fixture()
        with tmpdir:
            rows = load_dataset_from_path(dataset_dir, image_root, split="adversarial")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["question_id"], "adversarial")

    def test_missing_split_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = os.path.join(tmpdir, "pope")
            image_root = os.path.join(tmpdir, "images")
            os.makedirs(dataset_dir)
            os.makedirs(image_root)
            with open(os.path.join(dataset_dir, "coco_pope_random.jsonl"), "w", encoding="utf-8") as f:
                f.write("{}\n")

            with self.assertRaisesRegex(FileNotFoundError, "split=popular"):
                load_dataset_from_path(dataset_dir, image_root, split="popular")


if __name__ == "__main__":
    unittest.main()
