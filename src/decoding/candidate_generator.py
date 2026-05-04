from __future__ import annotations

from typing import Any, Dict, List

import torch
from PIL import Image

from src.decoding.token_cd import contrastive_generate, token_cd_result_to_dict
from src.models.qwen25vl_adapter import Qwen25VLAdapter, clone_inputs, move_inputs_to_device
from src.perturbations.image_perturb import perturb_image_pil, stable_sample_seed


class ContrastiveCandidateGenerator:
    path_configs = [
        {"beta": 0.20, "temperature": 0.6},
        {"beta": 0.35, "temperature": 0.8},
        {"beta": 0.50, "temperature": 1.0},
        {"beta": 0.65, "temperature": 1.1},
        {"beta": 0.80, "temperature": 1.2},
    ]

    def __init__(
        self,
        model=None,
        processor=None,
        adapter=None,
        max_new_tokens: int = 128,
        top_p: float = 0.9,
        topk_plausible: int = 50,
        neg_type: str = "gaussian",
        neg_std: float = 0.2,
        device: str = "cuda",
        perturb_seed_base: int | None = None,
    ):
        if adapter is None and processor is None and hasattr(model, "build_inputs"):
            adapter = model
            model = getattr(adapter, "model", None)
            processor = getattr(adapter, "processor", None)
        if adapter is None:
            if model is None or processor is None:
                raise ValueError("model and processor are required when adapter is not provided.")
            adapter = Qwen25VLAdapter(model, processor, device=device)

        self.adapter = adapter
        self.model = model if model is not None else adapter.model
        self.processor = processor if processor is not None else adapter.processor
        self.tokenizer = adapter.tokenizer
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.topk_plausible = topk_plausible
        self.neg_type = neg_type
        self.neg_std = neg_std
        self.perturb_seed_base = perturb_seed_base

    def generate(
        self,
        image_path: str,
        question: str,
        num_candidates: int = 5,
        raw_item: dict[str, Any] | None = None,
        perturb_seed: int | None = None,
    ) -> List[Dict]:
        """
        For each path, call token_cd.contrastive_generate.
        Return candidate text + trace metrics.
        """
        inputs_pos = self.adapter.move_to_device(self.adapter.build_inputs(image_path, question))
        if perturb_seed is None and raw_item is not None:
            perturb_seed = stable_sample_seed(self.perturb_seed_base, raw_item)

        if self.neg_type == "text_only":
            inputs_neg = self.adapter.move_to_device(self.adapter.build_text_inputs(question))
        else:
            with Image.open(image_path) as image:
                neg_image = perturb_image_pil(
                    image.convert("RGB"),
                    mode=self.neg_type,
                    std=self.neg_std,
                    seed=perturb_seed,
                )
            inputs_neg = self.adapter.move_to_device(self.adapter.build_inputs_from_pil(neg_image, question))

        candidates: list[dict[str, Any]] = []
        for path_id, config in enumerate(self.path_configs[:num_candidates]):
            beta = float(config["beta"])
            temperature = float(config["temperature"])
            result = contrastive_generate(
                self.model,
                self.processor,
                clone_inputs(inputs_pos),
                clone_inputs(inputs_neg),
                self.tokenizer,
                max_new_tokens=self.max_new_tokens,
                beta=beta,
                temperature=temperature,
                top_p=self.top_p,
                topk_plausible=self.topk_plausible,
                dynamic_alpha=False,
            )
            candidate = token_cd_result_to_dict(result)
            candidate["path_id"] = path_id
            candidate["config"] = {
                **config,
                "top_p": self.top_p,
                "topk_plausible": self.topk_plausible,
                "neg_type": self.neg_type,
                "neg_std": self.neg_std,
            }
            candidates.append(candidate)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return candidates


class EGMHCDGenerator:
    """
    EG-MHCD-AE v2 candidate generator: each path is true token-level CD.
    """

    def __init__(
        self,
        model,
        processor,
        max_new_tokens: int = 32,
        num_candidates: int = 5,
        top_p: float = 0.9,
        topk_plausible: int = 50,
        neg_type: str = "gaussian",
        neg_std: float = 0.2,
        perturb_seed_base: int | None = None,
    ):
        self.num_candidates = num_candidates
        self.generator = ContrastiveCandidateGenerator(
            model,
            processor,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            topk_plausible=topk_plausible,
            neg_type=neg_type,
            neg_std=neg_std,
            perturb_seed_base=perturb_seed_base,
        )
        self.path_configs = self.generator.path_configs[:num_candidates]

    def generate_candidates(self, dataloader):
        all_results = []
        for batch_inputs, raw_items in dataloader:
            raw_item = raw_items[0]
            print(
                f"正在为样本生成 {self.num_candidates} 条 EG-CD 候选: "
                f"{raw_item.get('question_id', raw_item.get('image_name', 'unknown'))}"
            )
            batch_candidates = self.generator.generate(
                raw_item["image_path"],
                raw_item["question"],
                num_candidates=self.num_candidates,
                raw_item=raw_item,
            )

            result_item = {
                "question": raw_item["question"],
                "candidates": batch_candidates,
            }
            for key in ("ground_truth", "image_path", "image_name", "question_id", "source_file", "source_index"):
                if raw_item.get(key) is not None:
                    result_item[key] = raw_item[key]
            all_results.append(result_item)

        return all_results


class SampleMajorityGenerator:
    """
    Legacy MHCD-AE candidate generator retained as sample_majority.
    """

    def __init__(self, model, processor, max_new_tokens: int = 256):
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens
        self.path_configs = [
            {"beta": 0.20, "temp": 0.6},
            {"beta": 0.35, "temp": 0.8},
            {"beta": 0.50, "temp": 1.0},
            {"beta": 0.65, "temp": 1.1},
            {"beta": 0.80, "temp": 1.2},
        ]

    @torch.no_grad()
    def generate_candidates(self, dataloader):
        all_results = []
        self.model.eval()

        for batch_inputs, raw_items in dataloader:
            batch_inputs = move_inputs_to_device(dict(batch_inputs), "cuda")
            batch_candidates = []

            for i, config in enumerate(self.path_configs):
                print(f"正在生成 sample_majority 路径 {i + 1}/5 (T={config['temp']})...")
                output_ids = self.model.generate(
                    **batch_inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=config["temp"],
                    top_p=0.9,
                )

                input_len = batch_inputs["input_ids"].shape[1]
                generated_text = self.processor.batch_decode(
                    output_ids[:, input_len:],
                    skip_special_tokens=True,
                )

                batch_candidates.append(
                    {
                        "path_id": i,
                        "text": generated_text[0],
                        "config": config,
                    }
                )

                del output_ids
                torch.cuda.empty_cache()

            raw_item = raw_items[0]
            result_item = {
                "question": raw_item["question"],
                "candidates": batch_candidates,
            }
            for key in ("ground_truth", "image_path", "image_name", "question_id", "source_file", "source_index"):
                if raw_item.get(key) is not None:
                    result_item[key] = raw_item[key]
            all_results.append(result_item)

        return all_results
