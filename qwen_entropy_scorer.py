"""
Backward-compatible exports for the legacy root-level entropy scorer module.

New code should import from src.scoring.sindex_answer_entropy directly.
"""

from src.scoring.sindex_answer_entropy import (
    CandidateScore,
    EGAnswerEntropyScorer,
    EntroGraphAnswerEntropyScorer,
    MHCDScorer,
    RerankResult,
    YES_NO_RE,
    candidate_confidence as _candidate_confidence,
    extract_yes_no,
    normalise_candidates as _normalise_candidates,
)


__all__ = [
    "CandidateScore",
    "EGAnswerEntropyScorer",
    "EntroGraphAnswerEntropyScorer",
    "MHCDScorer",
    "RerankResult",
    "YES_NO_RE",
    "_candidate_confidence",
    "_normalise_candidates",
    "extract_yes_no",
]


if __name__ == "__main__":
    scorer = EntroGraphAnswerEntropyScorer(device="cpu")
    candidates = [
        {"text": "no", "mean_cd_logprob": -0.2},
        {"text": "yes", "mean_cd_logprob": -1.2},
        {"text": "no", "mean_cd_logprob": -0.3},
        {"text": "no", "mean_cd_logprob": -0.4},
        {"text": "yes", "mean_cd_logprob": -1.4},
    ]
    result = scorer.score_and_select("Is there a dog?", candidates)
    print("\n=== Selection result ===")
    for i, cand in enumerate(candidates):
        score = result.scores[i]
        marker = "best" if i == result.best_index else "cand"
        print(f"[{marker}] cluster:{score.cluster} | ae:{score.AE:.4f} | {cand['text']}")
