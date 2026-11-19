"""
algorithms/butina_clustering.py
Executes Butina unsupervised clustering and evaluates cluster stability across
fingerprint representations and bootstrap resamples.
"""

from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from sklearn.metrics import adjusted_rand_score


def _fast_butina_from_adj(adj: np.ndarray) -> np.ndarray:
    """
    Vectorized Taylor-Butina clustering from an (N, N) boolean adjacency matrix.
    Matches RDKit Butina.ClusterData greedy neighbor-count order and returns
    1-indexed integer cluster labels of shape (N,).
    """
    n = adj.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int32)

    work_adj = adj.copy()
    np.fill_diagonal(work_adj, True)

    neighbor_counts = work_adj.sum(axis=1)
    # Stable descending sort by neighbor count
    order = np.argsort(-neighbor_counts, kind="stable")

    unassigned = np.ones(n, dtype=bool)
    labels = np.zeros(n, dtype=np.int32)
    cluster_id = 1

    for idx in order:
        if not unassigned[idx]:
            continue
        members = np.where(work_adj[idx] & unassigned)[0]
        labels[members] = cluster_id
        unassigned[members] = False
        cluster_id += 1

    return labels


class ButinaClusterer:
    """
    Executes Butina unsupervised clustering on molecular fingerprint representations.
    """

    @staticmethod
    def compute_distance_vector(fp_list: List[Any]) -> List[float]:
        """
        Computes the 1D lower-triangular Tanimoto distance vector required by
        rdkit.ML.Cluster.Butina.ClusterData.
        """
        n = len(fp_list)
        dists = []
        for i in range(1, n):
            sims = DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list[:i])
            dists.extend((1.0 - np.asarray(sims, dtype=np.float64)).tolist())
        return dists

    @staticmethod
    def compute_distance_matrix(fp_list: List[Any]) -> np.ndarray:
        """
        Computes a symmetric (N, N) float32 Tanimoto distance matrix once.
        """
        n = len(fp_list)
        dist_mat = np.zeros((n, n), dtype=np.float32)
        for i in range(1, n):
            sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list[:i]), dtype=np.float32)
            d = 1.0 - sims
            dist_mat[i, :i] = d
            dist_mat[:i, i] = d
        return dist_mat

    @classmethod
    def cluster_fingerprints(
        cls,
        fp_list: List[Any],
        dist_cutoff: float = 0.3
    ) -> Tuple[Tuple[int, ...], ...]:
        """
        Runs Butina clustering for a list of bitvectors at a specified distance cutoff
        (dist_cutoff = 1.0 - Tanimoto similarity threshold).
        """
        n = len(fp_list)
        if n == 0:
            return ()
        dists = cls.compute_distance_vector(fp_list)
        clusters = Butina.ClusterData(dists, n, dist_cutoff, isDistData=True)
        return clusters

    @classmethod
    def cluster_dataset(
        cls,
        dataset_fps: Dict[str, Dict[str, Any]],
        fp_name: str = "ecfp4",
        cutoffs: Optional[List[float]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Clusters a dataset dictionary for a given fingerprint type across multiple
        distance cutoffs (computes the pairwise distance vector ONCE across all cutoffs).
        """
        if cutoffs is None:
            cutoffs = [0.2, 0.3, 0.4, 0.5]

        cmpd_ids = list(dataset_fps.keys())
        fp_list = [dataset_fps[cid][fp_name] for cid in cmpd_ids]
        n = len(fp_list)

        # Precompute lower-triangular distances once for all cutoffs
        dists = cls.compute_distance_vector(fp_list) if n > 0 else []

        summary_records = []
        mapping_df = pd.DataFrame({"compound_id": cmpd_ids})

        for cutoff in cutoffs:
            clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True) if n > 0 else ()
            num_clusters = len(clusters)
            singletons = sum(1 for c in clusters if len(c) == 1)
            largest_cluster = max((len(c) for c in clusters), default=0)

            summary_records.append({
                "fingerprint": fp_name,
                "distance_cutoff": cutoff,
                "similarity_threshold": round(1.0 - cutoff, 2),
                "total_clusters": num_clusters,
                "singletons": singletons,
                "largest_cluster_size": largest_cluster,
                "mean_cluster_size": round(len(cmpd_ids) / num_clusters, 2) if num_clusters > 0 else 0
            })

            # Create cluster ID mapping and centroid flags
            cluster_id_map = {}
            centroid_map = {}
            for cluster_idx, cluster_tuple in enumerate(clusters, start=1):
                centroid_idx = cluster_tuple[0]
                for idx in cluster_tuple:
                    cid = cmpd_ids[idx]
                    cluster_id_map[cid] = cluster_idx
                    centroid_map[cid] = (idx == centroid_idx)

            col_cluster = f"cluster_{fp_name}_cutoff_{cutoff}"
            col_centroid = f"is_centroid_{fp_name}_cutoff_{cutoff}"
            mapping_df[col_cluster] = mapping_df["compound_id"].map(cluster_id_map)
            mapping_df[col_centroid] = mapping_df["compound_id"].map(centroid_map)

        summary_df = pd.DataFrame(summary_records)
        return summary_df, mapping_df


class ClusterStabilityAnalyzer:
    """
    Evaluates cluster stability across retained fingerprints and bootstrap resamples.
    """

    @staticmethod
    def evaluate_fingerprint_agreement(
        mapping_df: pd.DataFrame,
        fp_names: List[str],
        cutoff: float = 0.3
    ) -> pd.DataFrame:
        """
        Computes pairwise Adjusted Rand Index (ARI) between cluster assignments
        of different retained fingerprints at a specified distance cutoff.
        """
        records = []
        cols = [f"cluster_{fp}_cutoff_{cutoff}" for fp in fp_names if f"cluster_{fp}_cutoff_{cutoff}" in mapping_df.columns]

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                fp1 = cols[i].replace("cluster_", "").replace(f"_cutoff_{cutoff}", "")
                fp2 = cols[j].replace("cluster_", "").replace(f"_cutoff_{cutoff}", "")
                ari = adjusted_rand_score(mapping_df[cols[i]], mapping_df[cols[j]])
                records.append({
                    "fp_pair": f"{fp1} vs {fp2}",
                    "distance_cutoff": cutoff,
                    "ARI_agreement": round(float(ari), 4)
                })
        return pd.DataFrame(records)

    @staticmethod
    def bootstrap_resample_stability(
        fp_list: List[Any],
        n_bootstraps: int = 50,
        dist_cutoff: float = 0.3,
        random_state: int = 42
    ) -> Tuple[float, float]:
        """
        Evaluates Butina cluster stability under bootstrap resampling.
        Precomputes the (N, N) boolean neighbor adjacency matrix once and runs
        all bootstrap resamples in parallel using NumPy matrix slicing.
        Returns (mean_ARI, std_ARI).
        """
        n = len(fp_list)
        if n < 5:
            return 1.0, 0.0

        # 1. Compute full N x N distance matrix ONCE and build boolean adjacency matrix
        dist_mat = ButinaClusterer.compute_distance_matrix(fp_list)
        adj = dist_mat < dist_cutoff
        full_labels = _fast_butina_from_adj(adj)

        # 2. Pre-generate deterministic bootstrap index arrays matching original RNG sequence
        rng = np.random.RandomState(random_state)
        resample_indices = [rng.choice(n, size=n, replace=True) for _ in range(n_bootstraps)]

        # 3. Run bootstrap clustering and ARI scoring in parallel across all CPU cores
        def _run_one_bootstrap(resample_idx: np.ndarray) -> float:
            sub_adj = adj[np.ix_(resample_idx, resample_idx)]
            res_labels = _fast_butina_from_adj(sub_adj)
            return float(adjusted_rand_score(full_labels[resample_idx], res_labels))

        ari_scores = Parallel(n_jobs=-1)(
            delayed(_run_one_bootstrap)(idx) for idx in resample_indices
        )

        return round(float(np.mean(ari_scores)), 4), round(float(np.std(ari_scores)), 4)
