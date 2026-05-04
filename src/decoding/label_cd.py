from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict

import torch
from PIL import Image

from src.entrograph.entropy import entropy_from_probs, js_div, safe_softmax
from src.models.qwen25vl_adapter import Qwen25VLAdapter
from src.perturbations.image_perturb import perturb_image_pil


LABEL_TEXT_VARIANTS = {
    "yes": ("Yes", " yes"),
    "no": ("No", " no"),
}
LABELS = ("yes", "no")


@dataclass
class LabelCDResult:
    pred: str
    scores_pos: Dict[str, float]
    scores_neg: Dict[str, float]
    scores_cd: Dict[str, float]
    p_pos: Dict[str, float]
    p_neg: Dict[str, float]
    H_label: float
    JS_label: float
    alpha: float
    risk: float
    neg_type: str


class EntroGraphLabelCD:
    def __init__(
        self,
        adapter,
        alpha0=0.5,
        alpha_max=2.0,
        k_entropy=0.8,
        eta=1.0,
        neg_type="gaussian",
        neg_std=0.2,
    ):
        self.adapter = adapter
        self.alpha0 = float(alpha0)
        self.alpha_max = float(alpha_max)
        self.k_entropy = float(k_entropy)
        self.eta = float(eta)
        self.neg_type = neg_type
        self.neg_std = float(neg_std)

    @torch.no_grad()
    def predict(self, image_path: str, question: str) -> LabelCDResult:
        positive_inputs = self.adapter.move_to_device(self.adapter.build_inputs(image_path, question))

        image = Image.open(image_path).convert("RGB")
        negative_image = perturb_image_pil(image, mode=self.neg_type, std=self.neg_std)
        negative_inputs = self.adapter.move_to_device(
            self.adapter.build_inputs_from_pil(negative_image, question)
        )

        scores_pos_t = self._score_labels(positive_inputs)
        scores_neg_t = self._score_labels(negative_inputs)

        pos_vec = torch.stack([scores_pos_t[label] for label in LABELS], dim=-1)
        neg_vec = torch.stack([scores_neg_t[label] for label in LABELS], dim=-1)
        p_pos_vec = safe_softmax(pos_vec)
        p_neg_vec = safe_softmax(neg_vec)

        h_label_t = entropy_from_probs(p_pos_vec)
        js_label_t = js_div(p_pos_vec, p_neg_vec)
        h_label = float(h_label_t.reshape(-1)[0].item())
        js_label = float(js_label_t.reshape(-1)[0].item())
        alpha = min(max(self.alpha0 + self.k_entropy * h_label / math.log(2), 0.0), self.alpha_max)

        cd_vec = (1.0 + alpha) * pos_vec - alpha * neg_vec
        pred_idx = int(torch.argmax(cd_vec, dim=-1).reshape(-1)[0].item())
        pred = LABELS[pred_idx]
        risk = h_label - self.eta * js_label

        return LabelCDResult(
            pred=pred,
            scores_pos=self._tensor_label_dict(scores_pos_t),
            scores_neg=self._tensor_label_dict(scores_neg_t),
            scores_cd=self._vector_label_dict(cd_vec),
            p_pos=self._vector_label_dict(p_pos_vec),
            p_neg=self._vector_label_dict(p_neg_vec),
            H_label=h_label,
            JS_label=js_label,
            alpha=float(alpha),
            risk=float(risk),
            neg_type=self.neg_type,
        )

    def _score_labels(self, inputs) -> dict[str, torch.Tensor]:
        return {
            label: self._score_label_variants(inputs, LABEL_TEXT_VARIANTS[label])
            for label in LABELS
        }

    def _score_label_variants(self, inputs, variants: tuple[str, ...]) -> torch.Tensor:
        scores = [self.adapter.sequence_logprob(inputs, variant).float() for variant in variants]
        return torch.logsumexp(torch.stack(scores, dim=0), dim=0)

    @staticmethod
    def _tensor_label_dict(scores: dict[str, torch.Tensor]) -> Dict[str, float]:
        return {label: float(value.reshape(-1)[0].item()) for label, value in scores.items()}

    @staticmethod
    def _vector_label_dict(values: torch.Tensor) -> Dict[str, float]:
        flat = values.reshape(-1, len(LABELS))[0]
        return {label: float(flat[i].item()) for i, label in enumerate(LABELS)}


class EGLabelCDScorer:
    """
    Backward-compatible wrapper around EntroGraphLabelCD.
    """

    def __init__(
        self,
        model,
        processor,
        beta: float = 0.5,
        temperature: float = 1.0,
        labels: tuple[str, str] = LABELS,
        alpha_max: float = 2.0,
        k_entropy: float = 0.8,
        eta: float = 1.0,
        neg_type: str = "gaussian",
        neg_std: float = 0.2,
        device: str = "cuda",
    ):
        if tuple(labels) != LABELS:
            raise ValueError("EGLabelCDScorer currently supports yes/no labels only.")
        self.temperature = temperature
        self.adapter = Qwen25VLAdapter(model, processor, device=device)
        self.engine = EntroGraphLabelCD(
            self.adapter,
            alpha0=beta,
            alpha_max=alpha_max,
            k_entropy=k_entropy,
            eta=eta,
            neg_type=neg_type,
            neg_std=neg_std,
        )

    def score(self, batch_inputs, raw_item) -> dict:
        result = self.engine.predict(raw_item["image_path"], raw_item["question"])
        cd_probs = self._softmax_dict(result.scores_cd)
        sorted_cd = sorted(result.scores_cd.values(), reverse=True)
        payload = {
            "best_answer": result.pred,
            "positive_label_logprobs": result.scores_pos,
            "negative_label_logprobs": result.scores_neg,
            "cd_label_scores": result.scores_cd,
            "label_probs": cd_probs,
            "p_pos": result.p_pos,
            "p_neg": result.p_neg,
            "label_margin": sorted_cd[0] - sorted_cd[1],
            "answer_entropy": result.H_label,
            "JS_label": result.JS_label,
            "sindex": result.JS_label,
            "beta": result.alpha,
            "alpha": result.alpha,
            "risk": result.risk,
            "neg_type": result.neg_type,
            "temperature": self.temperature,
            "mode": "eg_label_cd",
        }
        payload.update({f"result_{key}": value for key, value in asdict(result).items()})
        return payload

    @staticmethod
    def _softmax_dict(scores: Dict[str, float]) -> Dict[str, float]:
        values = torch.tensor([scores[label] for label in LABELS], dtype=torch.float32)
        probs = safe_softmax(values)
        return {label: float(probs[i].item()) for i, label in enumerate(LABELS)}
