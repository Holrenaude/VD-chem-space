"""
algorithms/fingerprint_evaluator.py
Evaluates and selects optimal fingerprint representations based on Section 2.3:
1. Information Content: Bit density and Tanimoto pairwise similarity distributions
   (mean, standard deviation, min/max spread).
2. Category Agreement: Evaluates category recovery via Adjusted Rand Index (ARI)
   and Normalized Mutual Information (NMI) using hierarchical clustering.
3. Redundancy Reduction: Computes inter-fingerprint Spearman rank correlations across
   pairwise distance matrices to identify and group redundant representations.
"""

from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from rdkit import DataStructs
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import AgglomerativeClustering


class FingerprintEvaluator:
    """
    Evaluates fingerprint panel metrics (information content, category agreement,
    and redundancy) to select non-redundant, informative representations.
    """

    @staticmethod
    def compute_distance_matrix(fp_list: List[Any]) -> np.ndarray:
        """
        Computes a square 1 - Tanimoto distance matrix for a list of RDKit bitvectors.
        """
        n = len(fp_list)
        dist_matrix = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            sims = DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list)
            dist_matrix[i, :] = 1.0 - np.array(sims, dtype=np.float32)
        return dist_matrix

    @classmethod
    def evaluate_panel(
        cls,
        dataset_fps: Dict[str, Dict[str, Any]],
        category_labels: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray]]:
        """
        Evaluates all fingerprints in the dataset dictionary across information content,
        category agreement (ARI/NMI), and inter-fingerprint redundancy.
        """
        cmpd_ids = list(dataset_fps.keys())
        if not cmpd_ids:
            raise ValueError("Empty fingerprint dataset provided.")

        fp_names = list(next(iter(dataset_fps.values())).keys())

        dist_matrices: Dict[str, np.ndarray] = {}
        summary_records = []

        # 1. Individual Fingerprint Information Content & Category Agreement
        for fp_name in fp_names:
            fp_list = [dataset_fps[cid][fp_name] for cid in cmpd_ids]
            dmat = cls.compute_distance_matrix(fp_list)
            dist_matrices[fp_name] = dmat

            num_bits = fp_list[0].GetNumBits()
            mean_density = np.mean([fp.GetNumOnBits() / num_bits for fp in fp_list])

            triu_idx = np.triu_indices(len(cmpd_ids), k=1)
            sims = 1.0 - dmat[triu_idx]

            ari, nmi = np.nan, np.nan
            if category_labels is not None and len(set(category_labels)) > 1:
                n_clusters = len(set(category_labels))
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    metric='precomputed',
                    linkage='average'
                )
                cluster_preds = clustering.fit_predict(dmat)
                ari = adjusted_rand_score(category_labels, cluster_preds)
                nmi = normalized_mutual_info_score(category_labels, cluster_preds)

            summary_records.append({
                "fingerprint": fp_name,
                "bit_length": num_bits,
                "mean_bit_density": round(float(mean_density), 4),
                "mean_similarity": round(float(np.mean(sims)), 4),
                "std_similarity": round(float(np.std(sims)), 4),
                "min_similarity": round(float(np.min(sims)), 4),
                "max_similarity": round(float(np.max(sims)), 4),
                "ARI": round(float(ari), 4) if not np.isnan(ari) else np.nan,
                "NMI": round(float(nmi), 4) if not np.isnan(nmi) else np.nan
            })

        summary_df = pd.DataFrame(summary_records)

        # 2. Inter-Fingerprint Redundancy Matrix (Spearman Rank Correlation)
        n_fps = len(fp_names)
        corr_matrix = np.zeros((n_fps, n_fps), dtype=np.float32)
        triu_idx = np.triu_indices(len(cmpd_ids), k=1)

        for i, name1 in enumerate(fp_names):
            vec1 = dist_matrices[name1][triu_idx]
            for j, name2 in enumerate(fp_names):
                vec2 = dist_matrices[name2][triu_idx]
                corr, _ = spearmanr(vec1, vec2)
                corr_matrix[i, j] = round(float(corr), 4)

        redundancy_df = pd.DataFrame(corr_matrix, index=fp_names, columns=fp_names)

        return summary_df, redundancy_df, dist_matrices

    @staticmethod
    def select_retained_fingerprints(
        summary_df: pd.DataFrame,
        redundancy_df: pd.DataFrame,
        max_correlation: float = 0.90
    ) -> List[str]:
        """
        Selects non-redundant fingerprint representations from distinct families
        by filtering out highly correlated pairs (|r| > max_correlation), prioritizing
        fingerprints with higher information spread (std_similarity).
        """
        corr = redundancy_df.abs()

        # Sort fingerprints by similarity standard deviation descending
        sorted_df = summary_df.sort_values(by="std_similarity", ascending=False)
        candidate_fps = sorted_df["fingerprint"].tolist()

        retained = []
        for fp in candidate_fps:
            # Retain fp if it is not collinear (> max_correlation) with any already retained fp
            if not any(corr.loc[fp, r_fp] > max_correlation for r_fp in retained):
                retained.append(fp)

        return retained
