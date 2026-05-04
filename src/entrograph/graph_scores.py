from __future__ import annotations

from .entropy import entropy_from_prob_list, self_information_scores


def cluster_masses_from_weights(cluster_ids: list[int], weights: list[float]) -> dict[int, float]:
    masses: dict[int, float] = {}
    for cluster_id, weight in zip(cluster_ids, weights):
        masses[int(cluster_id)] = masses.get(int(cluster_id), 0.0) + float(weight)
    return dict(sorted(masses.items()))


def binary_sindex_scores(cluster_ids: list[int], weights: list[float]) -> list[float]:
    return [
        sum(weight if cluster_ids[i] != cluster_ids[j] else 0.0 for j, weight in enumerate(weights))
        for i in range(len(cluster_ids))
    ]


def combine_entrograph_scores(
    answer_entropy_scores: list[float],
    sindex_scores: list[float],
    uncertainty_scores: list[float],
    ae_weight: float = 0.55,
    sindex_weight: float = 0.35,
    uncertainty_weight: float = 0.10,
) -> list[float]:
    return [
        ae_weight * ae + sindex_weight * sindex + uncertainty_weight * uncertainty
        for ae, sindex, uncertainty in zip(answer_entropy_scores, sindex_scores, uncertainty_scores)
    ]


def binary_graph_scores(
    cluster_ids: list[int],
    weights: list[float],
    confidence_scores: list[float],
) -> dict[str, object]:
    cluster_masses = cluster_masses_from_weights(cluster_ids, weights)
    answer_entropy = entropy_from_prob_list(cluster_masses.values(), normalise=True)
    ae_scores = self_information_scores(cluster_ids, cluster_masses)
    sindex_scores = binary_sindex_scores(cluster_ids, weights)
    uncertainty_scores = [1.0 - confidence for confidence in confidence_scores]
    entrograph_scores = combine_entrograph_scores(ae_scores, sindex_scores, uncertainty_scores)
    return {
        "cluster_masses": cluster_masses,
        "answer_entropy": answer_entropy,
        "ae_scores": ae_scores,
        "sindex_scores": sindex_scores,
        "entrograph_scores": entrograph_scores,
    }
