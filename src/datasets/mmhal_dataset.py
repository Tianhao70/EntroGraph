from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Any

import torch


DEFAULT_MMMHAL_REPO = "Shengcao1006/MMHal-Bench"


class MMHalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        response_template: str | None = None,
        limit: int | None = None,
        download_images: bool = True,
    ):
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.response_template = response_template
        self.limit = limit
        self.download_images = download_images
        self.template_path = self._resolve_template()
        self.records = self._load_records()
        self.items = self._build_items()

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.items[idx]

    def to_list(self) -> list[dict[str, Any]]:
        return list(self.items)

    def _resolve_template(self) -> Path:
        candidates: list[Path] = []
        if self.response_template:
            candidates.append(Path(self.response_template).expanduser())

        candidates.extend(
            [
                self.root / "response_template.json",
                self.root / "data" / "response_template.json",
                self.root / "MMHal-Bench" / "response_template.json",
            ]
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate

        try:
            from huggingface_hub import hf_hub_download

            downloaded = hf_hub_download(
                repo_id=DEFAULT_MMMHAL_REPO,
                repo_type="dataset",
                filename="response_template.json",
                local_dir=str(self.root),
            )
            return Path(downloaded)
        except Exception as exc:  # pragma: no cover - depends on network
            raise FileNotFoundError(
                "Could not find MMHal response_template.json locally and automatic "
                f"download from {DEFAULT_MMMHAL_REPO} failed. Set --mmhal-root or "
                "--mmhal-response-template to a local template file."
            ) from exc

    def _load_records(self) -> list[dict[str, Any]]:
        with self.template_path.open("r", encoding="utf-8") as f:
            records = json.load(f)
        if not isinstance(records, list):
            raise ValueError(f"MMHal template must be a list: {self.template_path}")
        if self.limit is not None:
            records = records[: self.limit]
        return records

    def _build_items(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for idx, record in enumerate(self.records):
            image_id = str(record.get("image_id", idx))
            image_path = self._resolve_image(record, image_id)
            items.append(
                {
                    "question_id": str(record.get("question_id", idx)),
                    "image_id": image_id,
                    "image_name": os.path.basename(image_path),
                    "image_path": image_path,
                    "question": str(record.get("question", "")),
                    "ground_truth": record.get("gt_answer", record.get("ground_truth")),
                    "gt_answer": record.get("gt_answer", record.get("ground_truth")),
                    "image_content": record.get("image_content", []),
                    "question_type": record.get("question_type"),
                    "question_topic": record.get("question_topic"),
                    "image_src": record.get("image_src"),
                    "task": "mmhal",
                    "mmhal_template_record": record,
                    "source_index": idx + 1,
                }
            )
        return items

    def _resolve_image(self, record: dict[str, Any], image_id: str) -> str:
        image_root_candidates = [
            self.root / "images",
            self.root / "imgs",
            self.root / "MMHal-Bench" / "images",
            self.root,
        ]
        for image_root in image_root_candidates:
            for ext in (".jpg", ".jpeg", ".png", ".webp"):
                candidate = image_root / f"{image_id}{ext}"
                if candidate.exists():
                    return str(candidate)

        if not self.download_images:
            raise FileNotFoundError(f"MMHal image not found for image_id={image_id}")

        image_src = record.get("image_src")
        if not image_src:
            raise FileNotFoundError(f"MMHal record has no image_src for image_id={image_id}")

        image_dir = self.root / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(str(image_src).split("?")[0]).suffix.lower()
        if suffix not in (".jpg", ".jpeg", ".png", ".webp"):
            suffix = ".jpg"
        target = image_dir / f"{image_id}{suffix}"
        if not target.exists():
            try:
                urllib.request.urlretrieve(str(image_src), target)
            except Exception as exc:  # pragma: no cover - depends on network
                raise FileNotFoundError(
                    f"Failed to download MMHal image for image_id={image_id} from {image_src}"
                ) from exc
        return str(target)

