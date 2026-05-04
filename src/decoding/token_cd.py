from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, List, Tuple

import torch

from src.entrograph.entropy import (
    entropy_from_probs,
    js_div,
    normalised_entropy_from_masked_logits,
    safe_softmax,
    topk_plausibility_mask,
)
from src.models.qwen25vl_adapter import (
    Qwen25VLAdapter,
    append_token_batch,
    build_text_only_inputs,
    clone_inputs,
    get_tokenizer,
    move_inputs_to_device,
)
from src.perturbations.image_perturb import perturb_image_pil, stable_sample_seed


@dataclass
class StepTrace:
    t: int
    token_id: int
    token_text: str
    H_t: float
    H_pos_t: float
    H_cd_t: float
    JS_t: float
    JS_norm_t: float
    alpha_t: float
    topk_pos: List[Tuple[str, float]]
    topk_cd: List[Tuple[str, float]]


@dataclass
class TokenCDResult:
    text: str
    token_ids: List[int]
    H_pos: float
    H_cd: float
    D_vis: float
    D_vis_norm: float
    risk_graph: float
    grounding_score: float
    S_graph: float
    avg_logprob_cd: float
    trace: List[StepTrace]


@dataclass(frozen=True)
class TokenCDConfig:
    beta: float = 0.5
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    max_new_tokens: int = 128
    do_sample: bool = False
    dynamic_alpha: bool = False
    alpha0: float = 0.5
    k_entropy: float = 0.8
    alpha_max: float = 2.0
    neg_type: str = "gaussian"
    neg_std: float = 0.2
    perturb_seed_base: int | None = None


def contrastive_logits(
    positive_logits: torch.Tensor,
    negative_logits: torch.Tensor,
    beta: float,
    temperature: float = 1.0,
) -> torch.Tensor:
    return (1.0 + beta) * positive_logits.float() - beta * negative_logits.float()


def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float | None = None,
) -> torch.Tensor:
    if filter_value is None:
        filter_value = torch.finfo(logits.float().dtype).min
    logits = logits.float()

    if top_k and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        threshold = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, filter_value)

    if 0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = safe_softmax(sorted_logits)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1,
            index=sorted_indices,
            src=sorted_indices_to_remove,
        )
        logits = logits.masked_fill(indices_to_remove, filter_value)

    return logits


@torch.no_grad()
def contrastive_generate(
    model,
    processor,
    inputs_pos,
    inputs_neg,
    tokenizer,
    max_new_tokens: int = 128,
    beta: float = 0.5,
    temperature: float = 1.0,
    top_p: float = 0.9,
    topk_plausible: int = 50,
    dynamic_alpha: bool = False,
    alpha0: float = 0.5,
    k_entropy: float = 0.8,
    alpha_max: float = 2.0,
) -> TokenCDResult:
    """
    Correctness-first token-level contrastive decoding.

    Qwen2.5-VL cache inputs can differ across transformer versions, so this
    implementation recomputes both branches with the full shared generated
    token prefix at every step. The generated token sequence is appended to
    both positive and negative branches identically.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if inputs_pos["input_ids"].shape[0] != 1 or inputs_neg["input_ids"].shape[0] != 1:
        raise ValueError("contrastive_generate currently expects batch_size=1.")

    model.eval()
    inputs_pos = clone_inputs(inputs_pos)
    inputs_neg = clone_inputs(inputs_neg)
    eos_token_ids = _get_eos_token_ids(model, tokenizer)

    token_ids: list[int] = []
    trace: list[StepTrace] = []
    selected_logprobs: list[float] = []
    h_pos_values: list[float] = []
    h_cd_values: list[float] = []
    js_values: list[float] = []
    js_norm_values: list[float] = []

    for t in range(max_new_tokens):
        logits_pos = model(**inputs_pos).logits[:, -1, :].float()
        logits_neg = model(**inputs_neg).logits[:, -1, :].float()

        p_pos = safe_softmax(logits_pos, temperature=temperature)
        p_neg = safe_softmax(logits_neg, temperature=temperature)
        h_pos_t = entropy_from_probs(p_pos)
        js_t = js_div(p_pos, p_neg)

        h_pos_scalar = float(h_pos_t.reshape(-1)[0].item())
        js_scalar = float(js_t.reshape(-1)[0].item())
        js_norm_scalar = min(max(js_scalar / math.log(2), 0.0), 1.0)
        if dynamic_alpha:
            vocab_size = max(int(logits_pos.shape[-1]), 2)
            h_norm = min(max(h_pos_scalar / math.log(vocab_size), 0.0), 1.0)
            alpha_t = min(max(float(alpha0) + float(k_entropy) * h_norm, 0.0), float(alpha_max))
        else:
            alpha_t = float(beta)

        logits_cd = (1.0 + alpha_t) * logits_pos - alpha_t * logits_neg
        logits_cd_masked = topk_plausibility_mask(logits_pos, logits_cd, topk_plausible)
        h_cd_t = normalised_entropy_from_masked_logits(logits_cd_masked, temperature=temperature)
        h_cd_scalar = float(h_cd_t.reshape(-1)[0].item())

        p_cd = safe_softmax(logits_cd_masked, temperature=temperature)
        sample_logits = top_k_top_p_filtering(logits_cd_masked, top_p=top_p)
        p_sample = safe_softmax(sample_logits, temperature=temperature)
        next_token = torch.multinomial(p_sample, num_samples=1) if 0 < top_p < 1.0 else torch.argmax(p_sample, dim=-1, keepdim=True)
        token_id = int(next_token.reshape(-1)[0].item())
        token_text = _decode_token(tokenizer, token_id)

        token_ids.append(token_id)
        selected_logprob = float(torch.log(p_sample[0, token_id].clamp_min(1e-12)).item())
        selected_logprobs.append(selected_logprob)
        h_pos_values.append(h_pos_scalar)
        h_cd_values.append(h_cd_scalar)
        js_values.append(js_scalar)
        js_norm_values.append(js_norm_scalar)

        trace.append(
            StepTrace(
                t=t,
                token_id=token_id,
                token_text=token_text,
                H_t=h_pos_scalar,
                H_pos_t=h_pos_scalar,
                H_cd_t=h_cd_scalar,
                JS_t=js_scalar,
                JS_norm_t=js_norm_scalar,
                alpha_t=alpha_t,
                topk_pos=_topk_trace(tokenizer, p_pos, k=5),
                topk_cd=_topk_trace(tokenizer, p_cd, k=5),
            )
        )

        token_tensor = next_token.reshape(1)
        inputs_pos = append_token_batch(inputs_pos, token_tensor)
        inputs_neg = append_token_batch(inputs_neg, token_tensor)

        if token_id in eos_token_ids:
            break

    h_pos = _mean(h_pos_values)
    h_cd = _mean(h_cd_values)
    d_vis = _mean(js_values)
    d_vis_norm = _mean(js_norm_values)
    risk_graph = h_cd - d_vis_norm
    grounding_score = d_vis_norm - h_cd
    avg_logprob_cd = _mean(selected_logprobs)
    text = _decode_sequence(tokenizer, token_ids)
    return TokenCDResult(
        text=text,
        token_ids=token_ids,
        H_pos=h_pos,
        H_cd=h_cd,
        D_vis=d_vis,
        D_vis_norm=d_vis_norm,
        risk_graph=risk_graph,
        grounding_score=grounding_score,
        S_graph=risk_graph,
        avg_logprob_cd=avg_logprob_cd,
        trace=trace,
    )


class TokenContrastiveDecoder:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.tokenizer = get_tokenizer(processor)

    @torch.no_grad()
    def generate_one(
        self,
        batch_inputs: dict[str, Any],
        raw_item: dict[str, Any],
        config: TokenCDConfig,
    ) -> dict[str, Any]:
        device = batch_inputs["input_ids"].device
        positive_inputs = clone_inputs(batch_inputs)
        negative_inputs = self.build_negative_inputs(
            raw_item,
            device=device,
            neg_type=config.neg_type,
            neg_std=config.neg_std,
            perturb_seed_base=config.perturb_seed_base,
        )
        result = contrastive_generate(
            self.model,
            self.processor,
            positive_inputs,
            negative_inputs,
            self.tokenizer,
            max_new_tokens=config.max_new_tokens,
            beta=config.beta,
            temperature=config.temperature,
            top_p=config.top_p if config.do_sample else 1.0,
            topk_plausible=config.top_k,
            dynamic_alpha=config.dynamic_alpha,
            alpha0=config.alpha0,
            k_entropy=config.k_entropy,
            alpha_max=config.alpha_max,
        )
        payload = token_cd_result_to_dict(result)
        payload["config"] = asdict(config)
        return payload

    def build_negative_inputs(
        self,
        raw_item,
        device,
        neg_type: str,
        neg_std: float,
        perturb_seed_base: int | None = None,
    ):
        if neg_type == "text_only":
            return build_text_only_inputs(self.processor, raw_item, device)

        from PIL import Image

        adapter = Qwen25VLAdapter(self.model, self.processor, device=device)
        perturb_seed = stable_sample_seed(perturb_seed_base, raw_item)

        with Image.open(raw_item["image_path"]) as image:
            neg_image = perturb_image_pil(
                image.convert("RGB"),
                mode=neg_type,
                std=neg_std,
                seed=perturb_seed,
            )

        return adapter.move_to_device(adapter.build_inputs_from_pil(neg_image, raw_item["question"]))


class TokenCDGenerator:
    def __init__(
        self,
        model,
        processor,
        beta: float = 0.5,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        max_new_tokens: int = 128,
        do_sample: bool = False,
        dynamic_alpha: bool = False,
        alpha0: float = 0.5,
        k_entropy: float = 0.8,
        alpha_max: float = 2.0,
        neg_type: str = "gaussian",
        neg_std: float = 0.2,
        perturb_seed_base: int | None = None,
    ):
        self.decoder = TokenContrastiveDecoder(model, processor)
        self.config = TokenCDConfig(
            beta=beta,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            dynamic_alpha=dynamic_alpha,
            alpha0=alpha0,
            k_entropy=k_entropy,
            alpha_max=alpha_max,
            neg_type=neg_type,
            neg_std=neg_std,
            perturb_seed_base=perturb_seed_base,
        )

    def generate(self, dataloader):
        results = []
        for batch_inputs, raw_items in dataloader:
            batch_inputs = move_inputs_to_device(dict(batch_inputs), "cuda")
            raw_item = raw_items[0]
            decoded = self.decoder.generate_one(batch_inputs, raw_item, self.config)
            report_item = {
                "question": raw_item["question"],
                "best_answer": decoded["text"],
                "generation": decoded,
                "selection_mode": "token_cd",
            }
            for key in ("ground_truth", "image_path", "image_name", "question_id", "source_file", "source_index"):
                if raw_item.get(key) is not None:
                    report_item[key] = raw_item[key]
            results.append(report_item)
        return results


def token_cd_result_to_dict(result: TokenCDResult) -> dict[str, Any]:
    return {
        "text": result.text,
        "token_ids": result.token_ids,
        "H_pos": result.H_pos,
        "H_cd": result.H_cd,
        "D_vis": result.D_vis,
        "D_vis_norm": result.D_vis_norm,
        "risk_graph": result.risk_graph,
        "grounding_score": result.grounding_score,
        "S_graph": result.S_graph,
        "avg_logprob_cd": result.avg_logprob_cd,
        "mean_cd_logprob": result.avg_logprob_cd,
        "trace": [asdict(step) for step in result.trace],
    }


def _get_eos_token_ids(model, tokenizer) -> set[int]:
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None:
        eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if eos is None:
        return set()
    if isinstance(eos, (list, tuple, set)):
        return {int(x) for x in eos if x is not None}
    return {int(eos)}


def _decode_token(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode([token_id], skip_special_tokens=False)
    except TypeError:
        return tokenizer.decode([token_id])


def _decode_sequence(tokenizer, token_ids: list[int]) -> str:
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=True).strip()
    except TypeError:
        return tokenizer.decode(token_ids).strip()


def _topk_trace(tokenizer, probs: torch.Tensor, k: int) -> list[tuple[str, float]]:
    k = min(k, int(probs.shape[-1]))
    values, indices = torch.topk(probs, k=k, dim=-1)
    return [
        (_decode_token(tokenizer, int(token_id)), float(prob))
        for token_id, prob in zip(indices[0].tolist(), values[0].tolist())
    ]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
