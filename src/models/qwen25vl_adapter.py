from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F
from PIL import Image


def process_vision_info(messages):
    from qwen_vl_utils import process_vision_info as _process_vision_info

    return _process_vision_info(messages)


def get_tokenizer(processor):
    return getattr(processor, "tokenizer", processor)


def move_inputs_to_device(inputs: dict[str, Any], device: torch.device | str) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}


def clone_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    return {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}


def append_token(inputs: dict[str, Any], token_id: int) -> dict[str, Any]:
    input_ids = inputs["input_ids"]
    old_len = input_ids.shape[1]
    next_token = torch.tensor([[token_id]], dtype=input_ids.dtype, device=input_ids.device)
    out: dict[str, Any] = {}

    for key, value in inputs.items():
        if not isinstance(value, torch.Tensor):
            out[key] = value
            continue

        if key == "input_ids":
            out[key] = torch.cat([value, next_token], dim=1)
        elif key == "attention_mask":
            next_mask = torch.ones((value.shape[0], 1), dtype=value.dtype, device=value.device)
            out[key] = torch.cat([value, next_mask], dim=1)
        elif value.dim() == 2 and value.shape[0] == input_ids.shape[0] and value.shape[1] == old_len:
            out[key] = torch.cat([value, value[:, -1:]], dim=1)
        else:
            out[key] = value

    return out


def build_qwen_text_messages(question: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [{"type": "text", "text": question}],
        }
    ]


def build_text_only_inputs_from_question(processor, question: str, device: torch.device | str | None = None) -> dict[str, Any]:
    messages = build_qwen_text_messages(question)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=None, videos=None, padding=True, return_tensors="pt")
    inputs = dict(inputs)
    return move_inputs_to_device(inputs, device) if device is not None else inputs


def build_text_only_inputs(processor, raw_item: dict[str, Any], device: torch.device | str) -> dict[str, Any]:
    return build_text_only_inputs_from_question(processor, raw_item["question"], device)


class Qwen25VLAdapter:
    def __init__(self, model, processor, device="cuda"):
        self.model = model
        self.processor = processor
        self.device = device
        self.tokenizer = get_tokenizer(processor)

    def build_inputs(self, image_path: str, question: str) -> Dict[str, Any]:
        """
        使用 Qwen2.5-VL chat template 构造原图输入。
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": question},
                ],
            }
        ]
        return self._build_inputs_from_messages(messages)

    def build_inputs_from_pil(self, image: Image.Image, question: str) -> Dict[str, Any]:
        """
        使用 PIL 图像构造输入，用于 negative branch。
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image.convert("RGB")},
                    {"type": "text", "text": question},
                ],
            }
        ]
        return self._build_inputs_from_messages(messages)

    def build_text_inputs(self, question: str) -> Dict[str, Any]:
        return build_text_only_inputs_from_question(self.processor, question)

    def _build_inputs_from_messages(self, messages: list[dict[str, Any]]) -> Dict[str, Any]:
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs if image_inputs else None,
            videos=video_inputs if video_inputs else None,
            padding=True,
            return_tensors="pt",
        )
        return dict(inputs)

    def move_to_device(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return move_inputs_to_device(inputs, self.device)

    @torch.no_grad()
    def next_token_logits(self, inputs: Dict[str, Any]) -> torch.Tensor:
        """
        Forward once, return logits[:, -1, :].
        """
        outputs = self.model(**inputs)
        return outputs.logits[:, -1, :]

    @torch.no_grad()
    def sequence_logprob(self, inputs: Dict[str, Any], label_text: str) -> torch.Tensor:
        """
        Teacher-forcing sequence logprob for label_text.
        Must support multi-token labels.
        Return shape [batch].
        """
        token_ids = self.tokenizer.encode(label_text, add_special_tokens=False)
        if not token_ids:
            batch_size = int(inputs["input_ids"].shape[0])
            return torch.zeros(batch_size, dtype=torch.float32, device=inputs["input_ids"].device)

        running_inputs = clone_inputs(inputs)
        batch_size = int(running_inputs["input_ids"].shape[0])
        total_logprob = torch.zeros(batch_size, dtype=torch.float32, device=running_inputs["input_ids"].device)

        for token_id in token_ids:
            logits = self.next_token_logits(running_inputs).float()
            log_probs = F.log_softmax(logits, dim=-1)
            token_tensor = torch.full(
                (batch_size,),
                int(token_id),
                dtype=torch.long,
                device=log_probs.device,
            )
            total_logprob = total_logprob + log_probs.gather(dim=-1, index=token_tensor[:, None]).squeeze(-1)
            running_inputs = append_token_batch(running_inputs, token_tensor)

        return total_logprob


def append_token_batch(inputs: dict[str, Any], token_ids: torch.Tensor) -> dict[str, Any]:
    input_ids = inputs["input_ids"]
    old_len = input_ids.shape[1]
    next_tokens = token_ids.to(device=input_ids.device, dtype=input_ids.dtype).view(input_ids.shape[0], 1)
    out: dict[str, Any] = {}

    for key, value in inputs.items():
        if not isinstance(value, torch.Tensor):
            out[key] = value
            continue

        if key == "input_ids":
            out[key] = torch.cat([value, next_tokens], dim=1)
        elif key == "attention_mask":
            next_mask = torch.ones((value.shape[0], 1), dtype=value.dtype, device=value.device)
            out[key] = torch.cat([value, next_mask], dim=1)
        elif value.dim() == 2 and value.shape[0] == input_ids.shape[0] and value.shape[1] == old_len:
            out[key] = torch.cat([value, value[:, -1:]], dim=1)
        else:
            out[key] = value

    return out
