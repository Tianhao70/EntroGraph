from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from src.models.qwen25vl_adapter import build_text_only_inputs


def perturb_image_pil(
    image: Image.Image,
    mode: str = "gaussian",
    std: float = 0.2,
    seed: int | None = None,
) -> Image.Image:
    """
    Perturb a PIL image and return an RGB image with the same size.

    mode:
      gaussian: add Gaussian noise on [0,1] tensor
      blur: GaussianBlur radius=5
      gray: gray RGB image
    """
    rgb = image.convert("RGB")

    if mode == "gaussian":
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        noise = rng.normal(loc=0.0, scale=max(float(std), 0.0), size=arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0.0, 1.0)
        return Image.fromarray((arr * 255.0).round().astype(np.uint8)).convert("RGB")

    if mode == "blur":
        return rgb.filter(ImageFilter.GaussianBlur(radius=5)).convert("RGB")

    if mode == "gray":
        return ImageOps.grayscale(rgb).convert("RGB")

    raise ValueError(f"Unknown perturbation mode: {mode!r}")


def drop_image_content(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    text_only_messages: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            content = [part for part in content if part.get("type") != "image"]
        text_only_messages.append({**message, "content": content})
    return text_only_messages


def build_negative_text_only_inputs(processor, raw_item: dict[str, Any], device):
    return build_text_only_inputs(processor, raw_item, device)


def stable_sample_seed(global_seed: int | None, raw_item: dict[str, Any]) -> int | None:
    if global_seed is None:
        return None
    key = f"{raw_item.get('question_id')}|{raw_item.get('image_name')}|{raw_item.get('source_index')}"
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return (int(global_seed) + int(digest[:8], 16)) % (2**32)
