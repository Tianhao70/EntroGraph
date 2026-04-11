import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform

class MHCDScorer:
    """
    MHCD-AE 阶段四：单答案熵计算与对比重排引擎
    """
    def __init__(self, device="cuda"):
        print("🧠 正在加载轻量级语义裁决官 (BGE-Small)...")
        # 使用 BGE-Small，它极小极快，几乎不吃显存，完美适合做裁判
        self.encoder = SentenceTransformer('BAAI/bge-small-zh-v1.5', device=device)
        self.gamma = 0.3  # 导师推荐的对比重排强度参数
        
    def score_and_select(self, question, candidates):
        """
        核心逻辑：聚类 -> 算熵 -> 对比重排 -> 选出最终答案
        candidates: List of strings (K=5)
        """
        K = len(candidates)
        if K < 2:
            return candidates[0]

        # 1. 构造拼接文本并提取句向量 (q ⊕ [SEP] ⊕ a_i)
        texts = [f"{question} [SEP] {ans}" for ans in candidates]
        embeddings = self.encoder.encode(texts, normalize_embeddings=True) # (K, D)
        
        # 2. 计算距离矩阵与凝聚聚类
        # 余弦距离矩阵
        dist_matrix = squareform(pdist(embeddings, metric='cosine'))
        
        # 聚类 (距离阈值设为0.15，可根据实际语料微调)
        clustering = AgglomerativeClustering(
            n_clusters=None, 
            distance_threshold=0.15, 
            metric='precomputed', 
            linkage='average'
        )
        labels = clustering.fit_predict(dist_matrix)
        
        # 3. 计算 Answer Entropy (黑盒简化版)
        AE_scores = np.zeros(K)
        for i in range(K):
            my_cluster = labels[i]
            cluster_indices = np.where(labels == my_cluster)[0]
            cluster_size = len(cluster_indices)
            
            # H_local: 局部不一致度 (自己与本簇其他成员的平均距离)
            if cluster_size > 1:
                other_indices = cluster_indices[cluster_indices != i]
                h_local = np.mean(dist_matrix[i, other_indices])
            else:
                h_local = 1.0 # 孤立点，局部不确定性极高
                
            # H_mem: 软归属熵 (所在簇越小，越像离群值，惩罚越大)
            h_mem = 1.0 - (cluster_size / K)
            
            # 计算单答案熵 AE_i
            AE_scores[i] = 0.7 * h_mem + 0.3 * h_local

        # 4. 找到最优 (i+) 和最差 (i-) 答案
        i_plus = np.argmin(AE_scores)
        i_minus = np.argmax(AE_scores)
        
        delta_H = AE_scores[i_minus] - AE_scores[i_plus]
        
        # 5. 决策路由
        # 导师逻辑：如果大家都很一致 (簇少，或 Delta H 极小)，没必要重排，直接返回最好的
        num_clusters = len(set(labels))
        if num_clusters == 1 or delta_H < 0.1:
            return candidates[i_plus], AE_scores, labels
            
        # 6. 对比重排 (Contrastive Reranking)
        # S_i = -AE_i + gamma * (cos(z_i, z_{i^+}) - cos(z_i, z_{i^-}))
        final_scores = np.zeros(K)
        for i in range(K):
            cos_to_plus = 1.0 - dist_matrix[i, i_plus]
            cos_to_minus = 1.0 - dist_matrix[i, i_minus]
            final_scores[i] = -AE_scores[i] + self.gamma * (cos_to_plus - cos_to_minus)
            
        best_idx = np.argmax(final_scores)
        return candidates[best_idx], AE_scores, labels

if __name__ == "__main__":
    # 模拟测试：假设生成器输出了 5 个答案，其中 1 个是幻觉
    scorer = MHCDScorer()
    q = "图中穿红色衣服的人在做什么？"
    simulated_candidates = [
        "穿红色衣服的人正在踢足球。",             # 稳妥
        "图中红色衣服的男子在草地上踢球。",        # 稳妥
        "一个红衣人正在绿茵场上射门。",            # 稳妥
        "红衣服的人正和一只狗在玩飞盘。",          # 严重幻觉！(距离远，自成一簇)
        "那个穿红色衣服的人在奔跑着踢足球。"        # 稳妥
    ]
    
    best_ans, ae, clusters = scorer.score_and_select(q, simulated_candidates)
    print("\n=== 裁决结果 ===")
    for i, ans in enumerate(simulated_candidates):
        print(f"[{'最佳' if ans==best_ans else '候选'}] 簇:{clusters[i]} | AE熵:{ae[i]:.4f} | {ans}")