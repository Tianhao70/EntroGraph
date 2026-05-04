from __future__ import annotations

import math
from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def _torch():
    import torch

    return torch


def _normalise_probs(p: torch.Tensor, eps: float):
    torch = _torch()
    probs = torch.nan_to_num(p.float(), nan=0.0, posinf=0.0, neginf=0.0)
    probs = probs.clamp_min(eps)
    denom = probs.sum(dim=-1, keepdim=True).clamp_min(eps)
    return probs / denom


def entropy_from_probs(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return entropy over the last dimension, preserving batch dimensions."""
    probs = _normalise_probs(p, eps)
    return -(probs * probs.clamp_min(eps).log()).sum(dim=-1)


def kl_div(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return KL(p || q) over the last dimension, preserving batch dimensions."""
    probs_p = _normalise_probs(p, eps)
    probs_q = _normalise_probs(q, eps)
    return (probs_p * (probs_p.clamp_min(eps).log() - probs_q.clamp_min(eps).log())).sum(dim=-1)


def js_div(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return Jensen-Shannon divergence over the last dimension."""
    probs_p = _normalise_probs(p, eps)
    probs_q = _normalise_probs(q, eps)
    midpoint = 0.5 * (probs_p + probs_q)
    return 0.5 * kl_div(probs_p, midpoint, eps=eps) + 0.5 * kl_div(probs_q, midpoint, eps=eps)


def _finite_min(logits: torch.Tensor):
    torch = _torch()
    if torch.is_floating_point(logits):
        return torch.finfo(logits.dtype).min
    return -1_000_000_000


def _finite_max(logits: torch.Tensor):
    torch = _torch()
    if torch.is_floating_point(logits):
        return torch.finfo(logits.dtype).max
    return 1_000_000_000


def _sanitize_logits(logits: torch.Tensor) -> torch.Tensor:
    torch = _torch()
    return torch.nan_to_num(
        logits,
        nan=_finite_min(logits),
        posinf=_finite_max(logits),
        neginf=_finite_min(logits),
    )


def topk_plausibility_mask(logits_pos: torch.Tensor, logits_cd: torch.Tensor, k: int) -> torch.Tensor:
    """
    Return logits_cd masked so only top-k tokens from logits_pos are allowed.
    """
    torch = _torch()
    if logits_pos.shape != logits_cd.shape:
        raise ValueError("logits_pos and logits_cd must have the same shape")
    if logits_pos.ndim == 0:
        raise ValueError("logits tensors must have at least one dimension")

    logits_cd = _sanitize_logits(logits_cd)
    vocab_size = logits_pos.shape[-1]
    if k >= vocab_size:
        return logits_cd

    masked_value = _finite_min(logits_cd)
    if k <= 0:
        return torch.full_like(logits_cd, masked_value)

    safe_pos = _sanitize_logits(logits_pos)
    topk_indices = torch.topk(safe_pos, k=min(k, vocab_size), dim=-1).indices
    allowed = torch.zeros_like(safe_pos, dtype=torch.bool)
    allowed.scatter_(dim=-1, index=topk_indices, value=True)
    return logits_cd.masked_fill(~allowed, masked_value)


def normalised_entropy_from_masked_logits(
    masked_logits: torch.Tensor,
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute entropy over unmasked logits and normalise by log(num_allowed).

    topk_plausibility_mask uses the finite dtype floor as its masked value for
    backward compatibility, so this helper treats values near that floor as
    masked in addition to non-finite values.
    """
    torch = _torch()
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if masked_logits.ndim == 0:
        raise ValueError("masked_logits must have at least one dimension")

    logits = _sanitize_logits(masked_logits.float())
    allowed = torch.isfinite(masked_logits)
    if torch.is_floating_point(masked_logits):
        floor = torch.finfo(masked_logits.dtype).min
        allowed = allowed & (masked_logits > floor / 2.0)

    num_allowed = allowed.sum(dim=-1)
    softmax_logits = logits.masked_fill(~allowed, float("-inf"))
    probs = safe_softmax(softmax_logits, temperature=temperature).masked_fill(~allowed, 0.0)
    denom = probs.sum(dim=-1, keepdim=True).clamp_min(eps)
    probs = probs / denom
    entropy = -(probs * probs.clamp_min(eps).log()).sum(dim=-1)
    normaliser = torch.log(num_allowed.clamp_min(1).float()).clamp_min(eps)
    normalised = entropy / normaliser
    normalised = torch.where(num_allowed > 1, normalised, torch.zeros_like(normalised))
    return torch.nan_to_num(normalised, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)


def safe_softmax(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Softmax over the last dimension with finite output for pathological logits."""
    torch = _torch()
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    scaled = logits.float() / temperature
    pos_inf = scaled == float("inf")
    has_pos_inf = pos_inf.any(dim=-1, keepdim=True)
    inf_counts = pos_inf.sum(dim=-1, keepdim=True).clamp_min(1)
    inf_probs = pos_inf.float() / inf_counts

    safe_logits = torch.nan_to_num(
        scaled,
        nan=torch.finfo(scaled.dtype).min,
        posinf=torch.finfo(scaled.dtype).max,
        neginf=torch.finfo(scaled.dtype).min,
    )
    probs = torch.softmax(safe_logits, dim=-1)
    probs = torch.where(has_pos_inf, inf_probs, probs)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    denom = probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return probs / denom


def safe_softmax_scores(scores: Iterable[float]) -> list[float]:
    values = [float(score) for score in scores]
    if not values:
        return []

    finite_values = [score for score in values if math.isfinite(score)]
    if not finite_values:
        return [1.0 / len(values) for _ in values]

    floor = min(finite_values) - 50.0
    safe_values = [score if math.isfinite(score) else floor for score in values]
    max_score = max(safe_values)
    exps = [math.exp(score - max_score) for score in safe_values]
    denom = sum(exps)
    if denom <= 0:
        return [1.0 / len(values) for _ in values]
    return [value / denom for value in exps]


def normalise_scores(scores: Iterable[float], default: float = 0.5) -> list[float]:
    values = [float(score) for score in scores]
    finite_values = [score for score in values if math.isfinite(score)]
    if not finite_values:
        return [default for _ in values]

    lo = min(finite_values)
    hi = max(finite_values)
    if hi - lo < 1e-9:
        return [default for _ in values]

    return [
        (score - lo) / (hi - lo) if math.isfinite(score) else 0.0
        for score in values
    ]


def entropy_from_prob_list(probs: Iterable[float], normalise: bool = True) -> float:
    values = [max(float(prob), 0.0) for prob in probs]
    total = sum(values)
    if total <= 0:
        return 0.0

    values = [value / total for value in values]
    entropy = -sum(prob * math.log(max(prob, 1e-12)) for prob in values)
    if not normalise or len(values) <= 1:
        return entropy

    max_entropy = math.log(len(values))
    return entropy / max_entropy if max_entropy > 0 else 0.0


def self_information_scores(cluster_ids: Iterable[int], cluster_masses: dict[int, float]) -> list[float]:
    max_entropy = math.log(len(cluster_masses)) if len(cluster_masses) > 1 else 1.0
    if max_entropy <= 0:
        return [0.0 for _ in cluster_ids]

    return [
        -math.log(max(cluster_masses[int(cluster_id)], 1e-12)) / max_entropy
        for cluster_id in cluster_ids
    ]
