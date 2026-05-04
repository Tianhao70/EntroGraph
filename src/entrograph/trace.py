from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenStepTrace:
    step: int
    token_id: int
    cd_logprob: float
    positive_logprob: float
    negative_logprob: float
    cd_margin: float


def mean_trace_value(trace: list[dict[str, float | int]], key: str) -> float:
    values = [float(item[key]) for item in trace if key in item]
    return sum(values) / len(values) if values else float("-inf")


def summarise_trace(trace: list[dict[str, float | int]]) -> dict[str, float]:
    return {
        "mean_cd_logprob": mean_trace_value(trace, "cd_logprob"),
        "mean_positive_logprob": mean_trace_value(trace, "positive_logprob"),
        "mean_negative_logprob": mean_trace_value(trace, "negative_logprob"),
        "mean_cd_margin": mean_trace_value(trace, "cd_margin"),
    }
