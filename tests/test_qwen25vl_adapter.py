from __future__ import annotations

import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from src.models.qwen25vl_adapter import Qwen25VLAdapter


class FakeTokenizer:
    def __init__(self):
        self.vocab = {
            "Yes": [1],
            "No": [2],
            " yes": [3, 4],
            " no": [5, 6],
        }

    def encode(self, text, add_special_tokens=False):
        return list(self.vocab[text])


class FakeProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.last_messages = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.last_messages = messages
        return "<chat>"

    def __call__(self, text, images=None, videos=None, padding=True, return_tensors="pt"):
        return {
            "input_ids": torch.tensor([[0, 0]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
            "images_seen": images,
        }


class FakeModel:
    def __init__(self):
        self.calls = []

    def __call__(self, **inputs):
        input_ids = inputs["input_ids"]
        self.calls.append(input_ids.clone())
        batch_size = input_ids.shape[0]
        logits = torch.full((batch_size, 1, 8), -20.0)
        step = input_ids.shape[1] - 2
        if step == 0:
            logits[:, -1, 3] = 3.0
            logits[:, -1, 5] = 2.0
            logits[:, -1, 1] = 1.0
            logits[:, -1, 2] = 0.5
        elif step == 1:
            logits[:, -1, 4] = 4.0
            logits[:, -1, 6] = 1.0
        else:
            logits[:, -1, 7] = 5.0
        return SimpleNamespace(logits=logits)


class Qwen25VLAdapterTest(unittest.TestCase):
    def test_build_inputs_uses_chat_template_and_process_vision_info(self):
        processor = FakeProcessor()
        adapter = Qwen25VLAdapter(FakeModel(), processor, device="cpu")

        with patch("src.models.qwen25vl_adapter.process_vision_info") as mock_process:
            mock_process.return_value = (["image-tensor"], None)
            inputs = adapter.build_inputs("/tmp/image.jpg", "Is there a cat?")

        self.assertIn("input_ids", inputs)
        self.assertEqual(inputs["images_seen"], ["image-tensor"])
        mock_process.assert_called_once()
        content = processor.last_messages[0]["content"]
        self.assertEqual(content[0]["type"], "image")
        self.assertEqual(content[0]["image"], "/tmp/image.jpg")

    def test_build_inputs_from_pil_uses_rgb_image(self):
        processor = FakeProcessor()
        adapter = Qwen25VLAdapter(FakeModel(), processor, device="cpu")
        image = Image.new("RGBA", (4, 3), color=(1, 2, 3, 4))

        with patch("src.models.qwen25vl_adapter.process_vision_info") as mock_process:
            mock_process.return_value = (["pil-image-tensor"], None)
            inputs = adapter.build_inputs_from_pil(image, "Question?")

        self.assertEqual(inputs["images_seen"], ["pil-image-tensor"])
        pil_payload = processor.last_messages[0]["content"][0]["image"]
        self.assertEqual(pil_payload.mode, "RGB")
        self.assertEqual(pil_payload.size, image.size)

    def test_move_to_device_moves_tensors_only(self):
        adapter = Qwen25VLAdapter(FakeModel(), FakeProcessor(), device="cpu")
        inputs = {"input_ids": torch.tensor([[1]]), "meta": "keep"}

        moved = adapter.move_to_device(inputs)

        self.assertEqual(moved["input_ids"].device.type, "cpu")
        self.assertEqual(moved["meta"], "keep")

    def test_next_token_logits_returns_last_position(self):
        adapter = Qwen25VLAdapter(FakeModel(), FakeProcessor(), device="cpu")
        inputs = {"input_ids": torch.tensor([[0, 0]]), "attention_mask": torch.tensor([[1, 1]])}

        logits = adapter.next_token_logits(inputs)

        self.assertEqual(tuple(logits.shape), (1, 8))
        self.assertAlmostEqual(float(logits[0, 3]), 3.0)

    def test_sequence_logprob_supports_multi_token_labels(self):
        model = FakeModel()
        adapter = Qwen25VLAdapter(model, FakeProcessor(), device="cpu")
        inputs = {"input_ids": torch.tensor([[0, 0]]), "attention_mask": torch.tensor([[1, 1]])}

        score = adapter.sequence_logprob(inputs, " yes")

        first_logits = model.calls[0].new_full((8,), -20.0, dtype=torch.float32)
        first_logits[3] = 3.0
        first_logits[5] = 2.0
        first_logits[1] = 1.0
        first_logits[2] = 0.5
        second_logits = model.calls[1].new_full((8,), -20.0, dtype=torch.float32)
        second_logits[4] = 4.0
        second_logits[6] = 1.0
        expected = torch.log_softmax(first_logits, dim=-1)[3] + torch.log_softmax(second_logits, dim=-1)[4]

        self.assertEqual(len(model.calls), 2)
        self.assertEqual(model.calls[1].tolist(), [[0, 0, 3]])
        self.assertEqual(tuple(score.shape), (1,))
        self.assertTrue(math.isclose(float(score[0]), float(expected), rel_tol=1e-6, abs_tol=1e-6))

    def test_sequence_logprob_label_variants(self):
        adapter = Qwen25VLAdapter(FakeModel(), FakeProcessor(), device="cpu")
        inputs = {"input_ids": torch.tensor([[0, 0]]), "attention_mask": torch.tensor([[1, 1]])}

        for label_text in ("Yes", "No", " yes", " no"):
            with self.subTest(label_text=label_text):
                score = adapter.sequence_logprob(inputs, label_text)
                self.assertEqual(tuple(score.shape), (1,))
                self.assertTrue(torch.isfinite(score).all())


if __name__ == "__main__":
    unittest.main()
