from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch


DEFAULT_CAPTION_PROMPT = (
    "Describe the image in one concise sentence. "
    "Mention only objects that are clearly visible."
)


class CocoCaptionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_root: str,
        annotation_root: str,
        image_ids_file: str | None = None,
        limit: int | None = None,
        prompt: str = DEFAULT_CAPTION_PROMPT,
    ):
        self.image_root = Path(image_root).expanduser()
        self.annotation_root = Path(annotation_root).expanduser()
        self.image_ids_file = image_ids_file
        self.limit = limit
        self.prompt = prompt

        if not self.image_root.exists():
            raise FileNotFoundError(f"COCO image root not found: {self.image_root}")
        if not self.annotation_root.exists():
            raise FileNotFoundError(f"COCO annotation root not found: {self.annotation_root}")

        self.instances_path = self.annotation_root / "instances_val2014.json"
        self.captions_path = self.annotation_root / "captions_val2014.json"
        if not self.instances_path.exists():
            raise FileNotFoundError(f"Missing COCO instances annotation: {self.instances_path}")
        if not self.captions_path.exists():
            raise FileNotFoundError(f"Missing COCO captions annotation: {self.captions_path}")

        self.items = self._build_items()

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.items[idx]

    def to_list(self) -> list[dict[str, Any]]:
        return list(self.items)

    def _build_items(self) -> list[dict[str, Any]]:
        with self.instances_path.open("r", encoding="utf-8") as f:
            instances = json.load(f)

        images_by_id = {int(image["id"]): image for image in instances.get("images", [])}
        if not images_by_id:
            raise ValueError(f"No images found in {self.instances_path}")

        if self.image_ids_file:
            image_ids = _load_image_ids(self.image_ids_file)
        else:
            image_ids = sorted(images_by_id)

        rows: list[dict[str, Any]] = []
        for image_id in image_ids:
            image_id = int(image_id)
            if image_id not in images_by_id:
                continue
            image_info = images_by_id[image_id]
            image_name = image_info.get("file_name") or f"COCO_val2014_{image_id:012d}.jpg"
            image_path = self.image_root / image_name
            if not image_path.exists():
                raise FileNotFoundError(f"COCO image not found: {image_path}")
            rows.append(
                {
                    "question_id": str(image_id),
                    "image_id": image_id,
                    "image_name": image_name,
                    "image_path": str(image_path),
                    "question": self.prompt,
                    "task": "chair",
                    "ground_truth": None,
                }
            )
            if self.limit is not None and len(rows) >= self.limit:
                break

        return rows


def _load_image_ids(path: str) -> list[int]:
    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"chair image ids file not found: {source}")

    if source.suffix.lower() == ".json":
        data = json.load(source.open("r", encoding="utf-8"))
        if isinstance(data, list):
            if data and isinstance(data[0], dict):
                return [int(row.get("image_id", row.get("id"))) for row in data]
            return [int(value) for value in data]
        if isinstance(data, dict):
            values = data.get("image_ids", data.get("ids"))
            if values is None:
                raise ValueError(f"JSON image id file must contain image_ids or ids: {source}")
            return [int(value) for value in values]

    ids: list[int] = []
    with source.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids.append(int(line))
    return ids

