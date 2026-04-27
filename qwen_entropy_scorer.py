import re
from array import array
from collections import Counter
from math import ceil


class MHCDScorer:
    """
    Selects one answer from MHCD candidates.

    For POPE-style yes/no evaluation, the scorer uses label-level evidence
    instead of whole question-answer embeddings. Embedding clustering remains as
    a lazy fallback for non-binary answers.
    """

    YES_NO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)

    def __init__(self, device="cuda", encoder_model="BAAI/bge-small-en-v1.5"):
        print("MHCDScorer: using binary yes/no adjudication with lazy embedding fallback.")
        self.device = device
        self.encoder_model = encoder_model
        self.encoder = None
        self.gamma = 0.3
        self.last_mode = None
        self.last_candidate_labels = None
        self.last_label_counts = None

    @classmethod
    def extract_yes_no(cls, text):
        matches = cls.YES_NO_RE.findall(str(text).lower())
        if not matches:
            return None
        unique = set(matches)
        if len(unique) != 1:
            return None
        return matches[0]

    def score_and_select(self, question, candidates):
        """
        Return (best_answer, scores, clusters).

        In binary mode:
        - clusters are label ids: no=0, yes=1, unknown=-1.
        - scores are lower for the selected/stronger label.
        """
        if not candidates:
            self.last_mode = "empty"
            self.last_candidate_labels = []
            self.last_label_counts = {}
            return "", array("d"), array("i")

        if len(candidates) == 1:
            label = self.extract_yes_no(candidates[0])
            self.last_mode = "single"
            self.last_candidate_labels = [label]
            self.last_label_counts = dict(Counter([label])) if label else {}
            return candidates[0], array("d", [0.0]), array("i", [self._label_to_cluster(label)])

        binary_result = self._score_binary_candidates(candidates)
        if binary_result is not None:
            return binary_result

        return self._score_with_embeddings(question, candidates)

    def _score_binary_candidates(self, candidates):
        labels = [self.extract_yes_no(candidate) for candidate in candidates]
        known_labels = [label for label in labels if label in ("yes", "no")]
        if len(known_labels) < max(1, (len(candidates) // 2) + 1):
            return None

        counts = Counter(known_labels)
        most_common = counts.most_common()
        majority_label, majority_count = most_common[0]
        anchor_label = labels[0]

        # Use the low-temperature first path as the anchor unless the other
        # paths show strong agreement. This avoids letting stochastic sampling
        # turn a weak yes bias into extra false positives.
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
                # Lower is better. Candidates from the selected label group
                # receive the smallest score; minority labels are penalized.
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
        # Encode answers only. Including the full question makes yes/no-like
        # answers collapse into a single cluster because the question dominates.
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


if __name__ == "__main__":
    scorer = MHCDScorer(device="cpu")
    q = "Is there a dog in the image? Please answer yes or no."
    simulated_candidates = ["no", "yes", "no", "no", "no"]
    best_ans, ae, clusters = scorer.score_and_select(q, simulated_candidates)
    print("\n=== Selection result ===")
    for i, ans in enumerate(simulated_candidates):
        marker = "best" if ans == best_ans else "cand"
        print(f"[{marker}] cluster:{clusters[i]} | score:{ae[i]:.4f} | {ans}")
