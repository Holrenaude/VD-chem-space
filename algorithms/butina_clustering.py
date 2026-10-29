"""
algorithms/butina_clustering.py
Executes Butina unsupervised clustering and evaluates cluster stability across
fingerprint representations and bootstrap resamples.
"""

from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from sklearn.metrics import adjusted_rand_score


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
            for s in sims:
                dists.append(1.0 - float(s))
        return dists

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
        distance cutoffs.
        """
        if cutoffs is None:
            cutoffs = [0.2, 0.3, 0.4, 0.5]

        cmpd_ids = list(dataset_fps.keys())
        fp_list = [dataset_fps[cid][fp_name] for cid in cmpd_ids]

        summary_records = []
        mapping_df = pd.DataFrame({"compound_id": cmpd_ids})

        for cutoff in cutoffs:
            clusters = cls.cluster_fingerprints(fp_list, dist_cutoff=cutoff)
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
        Returns (mean_ARI, std_ARI) comparing full dataset clustering against resampled subsets.
        """
        n = len(fp_list)
        if n < 5:
            return 1.0, 0.0

        dists = ButinaClusterer.compute_distance_vector(fp_list)
        full_clusters = Butina.ClusterData(dists, n, dist_cutoff, isDistData=True)

        full_labels = np.zeros(n, dtype=int)
        for cluster_idx, cluster_tuple in enumerate(full_clusters, start=1):
            for idx in cluster_tuple:
                full_labels[idx] = cluster_idx

        ari_scores = []
        rng = np.random.RandomState(random_state)

        for _ in range(n_bootstraps):
            resample_idx = rng.choice(n, size=n, replace=True)
            resampled_fps = [fp_list[i] for i in resample_idx]

            res_dists = ButinaClusterer.compute_distance_vector(resampled_fps)
            res_clusters = Butina.ClusterData(res_dists, n, dist_cutoff, isDistData=True)

            res_labels = np.zeros(n, dtype=int)
            for cluster_idx, cluster_tuple in enumerate(res_clusters, start=1):
                for idx in cluster_tuple:
                    res_labels[idx] = cluster_idx

            ari = adjusted_rand_score(full_labels[resample_idx], res_labels)
            ari_scores.append(ari)

        return round(float(np.mean(ari_scores)), 4), round(float(np.std(ari_scores)), 4)
