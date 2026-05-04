from __future__ import annotations

import re
from array import array
from collections import Counter
from dataclasses import dataclass
from math import ceil
from typing import Any, Dict, List

from src.entrograph.entropy import normalise_scores, safe_softmax_scores
from src.entrograph.graph_scores import binary_graph_scores


YES_NO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


@dataclass
class CandidateScore:
    index: int
    text: str
    cluster: int
    H_mem: float
    H_local: float
    H_cd: float
    D_vis: float
    D_vis_norm: float
    H_vis: float
    H_vis_abs: float
    H_vis_rel: float
    avg_logprob_cd: float
    avg_logprob_norm: float
    AE: float
    final_score: float


@dataclass
class RerankResult:
    best_text: str
    best_index: int
    worst_index: int
    H_cluster: float
    delta_AE: float
    scores: List[CandidateScore]
    clusters: List[int]
    mode: str
    risk_high: bool = False
    embedding_mode: str = "question_answer"


def extract_yes_no(text):
    matches = YES_NO_RE.findall(str(text).lower())
    if not matches:
        return None
    unique = set(matches)
    if len(unique) != 1:
        return None
    return matches[0]


def normalise_candidates(candidates, candidate_metadata=None):
    if not candidates:
        return [], []

    if isinstance(candidates[0], dict):
        texts = [str(candidate.get("text", "")) for candidate in candidates]
        metadata = [dict(candidate) for candidate in candidates]
    else:
        texts = [str(candidate) for candidate in candidates]
        metadata = [dict(item or {}) for item in (candidate_metadata or [{} for _ in texts])]

    if len(metadata) < len(texts):
        metadata.extend({} for _ in range(len(texts) - len(metadata)))

    return texts, metadata[: len(texts)]


def candidate_confidence(metadata):
    scores = []
    for item in metadata:
        value = item.get("mean_cd_logprob")
        if value is None:
            value = item.get("mean_positive_logprob")
        try:
            scores.append(float(value))
        except (TypeError, ValueError):
            scores.append(float("-inf"))
    return safe_softmax_scores(scores), normalise_scores(scores)


class MHCDScorer:
    """
    Legacy sample-majority scorer.

    This is the old mhcd-ae behavior kept under the new method name
    sample_majority.
    """

    YES_NO_RE = YES_NO_RE

    def __init__(self, device="cuda", encoder_model="BAAI/bge-small-en-v1.5"):
        print("MHCDScorer: legacy sample_majority scorer with lazy embedding fallback.")
        self.device = device
        self.encoder_model = encoder_model
        self.encoder = None
        self.gamma = 0.3
        self.last_mode = None
        self.last_candidate_labels = None
        self.last_label_counts = None

    @classmethod
    def extract_yes_no(cls, text):
        return extract_yes_no(text)

    def score_and_select(self, question, candidates):
        texts, _ = normalise_candidates(candidates)
        if not texts:
            self.last_mode = "empty"
            self.last_candidate_labels = []
            self.last_label_counts = {}
            return "", array("d"), array("i")

        if len(texts) == 1:
            label = self.extract_yes_no(texts[0])
            self.last_mode = "single"
            self.last_candidate_labels = [label]
            self.last_label_counts = dict(Counter([label])) if label else {}
            return texts[0], array("d", [0.0]), array("i", [self._label_to_cluster(label)])

        binary_result = self._score_binary_candidates(texts)
        if binary_result is not None:
            return binary_result

        return self._score_with_embeddings(question, texts)

    def _score_binary_candidates(self, candidates):
        labels = [self.extract_yes_no(candidate) for candidate in candidates]
        known_labels = [label for label in labels if label in ("yes", "no")]
        if len(known_labels) < max(1, (len(candidates) // 2) + 1):
            return None

        counts = Counter(known_labels)
        majority_label, majority_count = counts.most_common(1)[0]
        anchor_label = labels[0]

        strong_agreement = majority_count >= max(2, int(ceil(0.8 * len(known_labels))))
        if strong_agreement or anchor_label not in ("yes", "no"):
            selected_label = majority_label
            mode = "binary_supermajority" if strong_agreement else "binary_majority"
        else:
            selected_label = anchor_label
            mode = "binary_anchor"

        best_idx = 0
        if labels[0] != selected_label:
            best_idx = next(i for i, label in enumerate(labels) if label == selected_label)

        scores = [1.0] * len(candidates)
        for i, label in enumerate(labels):
            if label in ("yes", "no"):
                label_confidence = counts[label] / len(known_labels)
                scores[i] = 1.0 - label_confidence
                if label != selected_label:
                    scores[i] += 1.0

        clusters = [self._label_to_cluster(label) for label in labels]
        self.last_mode = mode
        self.last_candidate_labels = labels
        self.last_label_counts = dict(counts)
        return candidates[best_idx], array("d", scores), array("i", clusters)

    @staticmethod
    def _label_to_cluster(label):
        if label == "yes":
            return 1
        if label == "no":
            return 0
        return -1

    def _ensure_encoder(self):
        if self.encoder is None:
            from sentence_transformers import SentenceTransformer

            self.encoder = SentenceTransformer(self.encoder_model, device=self.device)
        return self.encoder

    def _score_with_embeddings(self, question, candidates):
        import numpy as np
        from scipy.spatial.distance import pdist, squareform
        from sklearn.cluster import AgglomerativeClustering

        encoder = self._ensure_encoder()
        texts = [str(ans) for ans in candidates]
        embeddings = encoder.encode(texts, normalize_embeddings=True)
        dist_matrix = squareform(pdist(embeddings, metric="cosine"))

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.15,
            metric="precomputed",
            linkage="average",
        )
        labels = clustering.fit_predict(dist_matrix)

        k = len(candidates)
        ae_scores = np.zeros(k)
        for i in range(k):
            my_cluster = labels[i]
            cluster_indices = np.where(labels == my_cluster)[0]
            cluster_size = len(cluster_indices)

            if cluster_size > 1:
                other_indices = cluster_indices[cluster_indices != i]
                h_local = np.mean(dist_matrix[i, other_indices])
            else:
                h_local = 1.0

            h_mem = 1.0 - (cluster_size / k)
            ae_scores[i] = 0.7 * h_mem + 0.3 * h_local

        i_plus = np.argmin(ae_scores)
        i_minus = np.argmax(ae_scores)
        delta_h = ae_scores[i_minus] - ae_scores[i_plus]
        num_clusters = len(set(labels))

        self.last_mode = "embedding_cluster"
        self.last_candidate_labels = [None for _ in candidates]
        self.last_label_counts = {}

        if num_clusters == 1 or delta_h < 0.1:
            return candidates[i_plus], ae_scores, labels

        final_scores = np.zeros(k)
        for i in range(k):
            cos_to_plus = 1.0 - dist_matrix[i, i_plus]
            cos_to_minus = 1.0 - dist_matrix[i, i_minus]
            final_scores[i] = -ae_scores[i] + self.gamma * (cos_to_plus - cos_to_minus)

        best_idx = np.argmax(final_scores)
        return candidates[best_idx], ae_scores, labels


class EGAnswerEntropyScorer:
    """
    EG-MHCD-AE v2 reranker with SINDex/Answer Entropy/EntroGraph scores.
    """

    def __init__(
        self,
        encoder_model="BAAI/bge-small-en-v1.5",
        device="cuda",
        distance_threshold=0.15,
        gamma=0.3,
        mu=0.1,
        tau=0.1,
    ):
        print("EGAnswerEntropyScorer: SINDex + Answer Entropy + EntroGraph rerank.")
        self.encoder_model = encoder_model
        self.device = device
        self.distance_threshold = float(distance_threshold)
        self.gamma = float(gamma)
        self.mu = float(mu)
        self.tau = float(tau)
        self.encoder = None
        self.last_mode = None
        self.last_candidate_labels = None
        self.last_label_counts = None
        self.last_candidate_weights = None
        self.last_answer_entropy = None
        self.last_sindex_scores = None
        self.last_entrograph_scores = None
        self.last_cluster_masses = None
        self.last_ae_scores = None
        self.last_final_scores = None
        self.last_delta_ae = None
        self.last_risk_high = None
        self.last_embedding_mode = None

    def score_and_select(self, question: str, candidates: List[Dict]) -> RerankResult:
        texts, metadata = normalise_candidates(candidates)
        if not texts:
            result = RerankResult(
                best_text="",
                best_index=-1,
                worst_index=-1,
                H_cluster=0.0,
                delta_AE=0.0,
                scores=[],
                clusters=[],
                mode="empty",
            )
            self._record_result(result, [], {})
            return result

        return self._score_semantic_candidates(question, texts, metadata)

    def _ensure_encoder(self):
        if self.encoder is None:
            from sentence_transformers import SentenceTransformer

            self.encoder = SentenceTransformer(self.encoder_model, device=self.device)
        return self.encoder

    def _score_semantic_candidates(
        self,
        question: str,
        texts: list[str],
        metadata: list[dict[str, Any]],
    ) -> RerankResult:
        import math

        import numpy as np

        eps = 1e-12
        rho_min = 1e-3
        k = len(texts)
        embedding_texts = [
            f"{question.strip()} [SEP] {answer.strip()}"
            for answer in texts
        ]
        embeddings = self._encode_texts(embedding_texts)
        cosine = np.clip(embeddings @ embeddings.T, -1.0, 1.0)
        distance = np.clip(1.0 - cosine, 0.0, 2.0)
        clusters = self._cluster_average_linkage(distance, self.distance_threshold)
        cluster_ids = sorted(set(int(cluster_id) for cluster_id in clusters))
        m = len(cluster_ids)
        cluster_to_col = {cluster_id: col for col, cluster_id in enumerate(cluster_ids)}

        centroids = self._cluster_centroids(embeddings, clusters, cluster_ids)
        centroid_distance = np.clip(1.0 - embeddings @ centroids.T, 0.0, 2.0)
        memberships = self._softmax(-centroid_distance / max(self.tau, eps), axis=1)

        h_mem = np.zeros(k, dtype=float)
        if m > 1:
            h_mem = self._row_entropy(memberships) / math.log(m)

        h_local = np.zeros(k, dtype=float)
        for i in range(k):
            member_indices = np.where(clusters == clusters[i])[0]
            other_indices = member_indices[member_indices != i]
            if len(other_indices) == 0:
                h_local[i] = 1.0
            else:
                mean_cos = float(np.mean(cosine[i, other_indices]))
                h_local[i] = 1.0 - self._clamp(mean_cos, 0.0, 1.0)

        cluster_weights = []
        cluster_masses = {}
        for cluster_id in cluster_ids:
            member_indices = np.where(clusters == cluster_id)[0]
            p_c = len(member_indices) / k
            cluster_masses[int(cluster_id)] = float(p_c)
            if len(member_indices) > 1:
                sub_cos = cosine[np.ix_(member_indices, member_indices)]
                tri = sub_cos[np.triu_indices(len(member_indices), k=1)]
                rho_c = float(np.mean(tri)) if len(tri) else rho_min
            else:
                rho_c = rho_min
            cluster_weights.append(p_c * max(rho_c, rho_min) + eps)

        pbar = np.asarray(cluster_weights, dtype=float)
        pbar = pbar / max(float(np.sum(pbar)), eps)
        h_cluster = 0.0
        if m > 1:
            h_cluster = float(self._entropy(pbar) / math.log(m))
            h_cluster = self._clamp(h_cluster, 0.0, 1.0)

        h_cd = np.asarray([self._finite_float(item.get("H_cd", 0.0), 0.0) for item in metadata], dtype=float)
        h_cd = np.clip(h_cd, 0.0, 1.0)
        d_vis = np.asarray([self._finite_float(item.get("D_vis", 0.0), 0.0) for item in metadata], dtype=float)
        d_vis_norm = np.asarray(
            [
                self._finite_float(item.get("D_vis_norm", d_vis[i] / math.log(2.0)), 0.0)
                for i, item in enumerate(metadata)
            ],
            dtype=float,
        )
        d_vis_norm = np.clip(d_vis_norm, 0.0, 1.0)
        h_vis_abs = 1.0 - d_vis_norm
        h_vis_rel = 1.0 - self._minmax(d_vis)
        avg_logprob = np.asarray(
            [
                self._finite_float(item.get("avg_logprob_cd", item.get("mean_cd_logprob", 0.0)), 0.0)
                for item in metadata
            ],
            dtype=float,
        )
        avg_logprob_norm = np.asarray(normalise_scores(avg_logprob, default=0.5), dtype=float)

        ae = 0.30 * h_mem + 0.20 * h_local + 0.25 * h_cd + 0.25 * h_vis_abs
        i_plus = int(np.argmin(ae))
        i_minus = int(np.argmax(ae))
        delta_ae = float(ae[i_minus] - ae[i_plus])

        final_scores = np.zeros(k, dtype=float)
        for i in range(k):
            cos_to_plus = float(cosine[i, i_plus])
            cos_to_minus = float(cosine[i, i_minus])
            final_scores[i] = -ae[i] + self.gamma * (cos_to_plus - cos_to_minus) + self.mu * avg_logprob_norm[i]

        risk_high = False
        if m == 1 or h_cluster < 0.25:
            best_idx = i_plus
            mode = "low_cluster_entropy"
        else:
            best_idx = int(np.argmax(final_scores))
            mode = "eg_answer_entropy"
            if h_cluster > 0.65 and delta_ae < 0.10:
                mode = "high_uncertainty"
                risk_high = True

        score_rows = [
            CandidateScore(
                index=i,
                text=texts[i],
                cluster=int(clusters[i]),
                H_mem=float(h_mem[i]),
                H_local=float(h_local[i]),
                H_cd=float(h_cd[i]),
                D_vis=float(d_vis[i]),
                D_vis_norm=float(d_vis_norm[i]),
                H_vis=float(h_vis_abs[i]),
                H_vis_abs=float(h_vis_abs[i]),
                H_vis_rel=float(h_vis_rel[i]),
                avg_logprob_cd=float(avg_logprob[i]),
                avg_logprob_norm=float(avg_logprob_norm[i]),
                AE=float(ae[i]),
                final_score=float(final_scores[i]),
            )
            for i in range(k)
        ]
        result = RerankResult(
            best_text=texts[best_idx],
            best_index=best_idx,
            worst_index=i_minus,
            H_cluster=float(h_cluster),
            delta_AE=delta_ae,
            scores=score_rows,
            clusters=[int(cluster_id) for cluster_id in clusters.tolist()],
            mode=mode,
            risk_high=risk_high,
            embedding_mode="question_answer",
        )
        self._record_result(result, metadata, cluster_masses)
        return result

    def _encode_texts(self, texts):
        import numpy as np

        encoder = self._ensure_encoder()
        safe_texts = [text if str(text).strip() else "<empty>" for text in texts]
        embeddings = encoder.encode(safe_texts, normalize_embeddings=True)
        embeddings = np.asarray(embeddings, dtype=float)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        return self._l2_normalize(embeddings)

    @staticmethod
    def _l2_normalize(values):
        import numpy as np

        norms = np.linalg.norm(values, axis=1, keepdims=True)
        return values / np.maximum(norms, 1e-12)

    @staticmethod
    def _cluster_average_linkage(distance, threshold):
        import numpy as np

        clusters = [[i] for i in range(distance.shape[0])]
        while len(clusters) > 1:
            best_pair = None
            best_distance = float("inf")
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    avg_distance = float(np.mean(distance[np.ix_(clusters[i], clusters[j])]))
                    if avg_distance < best_distance:
                        best_distance = avg_distance
                        best_pair = (i, j)
            if best_pair is None or best_distance > threshold:
                break
            left, right = best_pair
            clusters[left] = clusters[left] + clusters[right]
            del clusters[right]

        labels = np.zeros(distance.shape[0], dtype=int)
        for label, member_indices in enumerate(clusters):
            for index in member_indices:
                labels[index] = label
        return labels

    @staticmethod
    def _cluster_centroids(embeddings, clusters, cluster_ids):
        import numpy as np

        centroids = []
        for cluster_id in cluster_ids:
            centroid = np.mean(embeddings[clusters == cluster_id], axis=0)
            norm = np.linalg.norm(centroid)
            centroids.append(centroid / max(norm, 1e-12))
        return np.asarray(centroids, dtype=float)

    @staticmethod
    def _softmax(values, axis=-1):
        import numpy as np

        shifted = values - np.max(values, axis=axis, keepdims=True)
        exp_values = np.exp(shifted)
        denom = np.maximum(np.sum(exp_values, axis=axis, keepdims=True), 1e-12)
        return exp_values / denom

    @staticmethod
    def _row_entropy(probs):
        import numpy as np

        safe = np.clip(probs, 1e-12, 1.0)
        return -np.sum(safe * np.log(safe), axis=1)

    @staticmethod
    def _entropy(probs):
        import numpy as np

        safe = np.clip(probs, 1e-12, 1.0)
        return -float(np.sum(safe * np.log(safe)))

    @staticmethod
    def _minmax(values):
        import numpy as np

        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return values
        low = float(np.min(values))
        high = float(np.max(values))
        if high - low <= 1e-12:
            return np.zeros_like(values)
        return (values - low) / (high - low)

    @staticmethod
    def _finite_float(value, default):
        import math

        try:
            out = float(value)
        except (TypeError, ValueError):
            return float(default)
        return out if math.isfinite(out) else float(default)

    @staticmethod
    def _clamp(value, low, high):
        return min(max(float(value), float(low)), float(high))

    def _record_result(self, result: RerankResult, metadata, cluster_masses):
        labels = [extract_yes_no(score.text) for score in result.scores]
        weights, _ = candidate_confidence(metadata) if metadata else ([], [])
        self.last_mode = result.mode
        self.last_candidate_labels = labels
        self.last_label_counts = dict(Counter(label for label in labels if label in ("yes", "no")))
        self.last_candidate_weights = list(weights)
        self.last_answer_entropy = result.H_cluster
        self.last_sindex_scores = [score.H_local for score in result.scores]
        self.last_entrograph_scores = [score.final_score for score in result.scores]
        self.last_cluster_masses = dict(cluster_masses)
        self.last_ae_scores = [score.AE for score in result.scores]
        self.last_final_scores = [score.final_score for score in result.scores]
        self.last_delta_ae = result.delta_AE
        self.last_risk_high = result.risk_high
        self.last_embedding_mode = result.embedding_mode


EntroGraphAnswerEntropyScorer = EGAnswerEntropyScorer
