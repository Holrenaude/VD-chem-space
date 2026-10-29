"""
algorithms/descriptor_filtering.py
Data-driven 3-stage descriptor filtering engine.

Pipeline Stages:
1. Filter missing values (NaNs / uncomputable descriptors).
2. Filter low-variance / constant descriptors (variance thresholding).
3. Filter collinear descriptor pairs (|r| > 0.90 Spearman rank correlation).
"""

from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np


class DescriptorFilter:
    """
    Executes data-driven descriptor filtering to retain only informative,
    non-redundant 2D descriptors across the dataset.
    """

    @staticmethod
    def filter_missing_values(df: pd.DataFrame, meta_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        """
        Stage 1: Removes descriptors that contain NaNs or failed to compute
        for any compound in the library.
        """
        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in meta_cols]
        valid_cols = df[num_cols].dropna(axis=1).columns.tolist()
        dropped = [c for c in num_cols if c not in valid_cols]
        return df[meta_cols + valid_cols].copy(), dropped

    @staticmethod
    def filter_low_variance(
        df: pd.DataFrame,
        meta_cols: List[str],
        threshold: float = 1e-5
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Stage 2: Removes constant or near-zero variance descriptors across the dataset.
        """
        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in meta_cols]
        variances = df[num_cols].var()
        valid_cols = variances[variances > threshold].index.tolist()
        dropped = [c for c in num_cols if c not in valid_cols]
        return df[meta_cols + valid_cols].copy(), dropped

    @staticmethod
    def filter_collinear(
        df: pd.DataFrame,
        meta_cols: List[str],
        threshold: float = 0.90,
        method: str = "spearman"
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Stage 3: Constructs a pairwise correlation matrix and drops one descriptor
        from each pair showing high collinearity (|r| > threshold).
        """
        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in meta_cols]
        corr_matrix = df[num_cols].corr(method=method).abs()

        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        dropped = [col for col in upper_tri.columns if any(upper_tri[col] > threshold)]
        retained = [c for c in num_cols if c not in dropped]
        return df[meta_cols + retained].copy(), dropped

    @classmethod
    def run_full_filter_pipeline(
        cls,
        df: pd.DataFrame,
        meta_cols: Optional[List[str]] = None,
        corr_threshold: float = 0.90,
        var_threshold: float = 1e-5
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Runs all three filtering stages sequentially and returns the cleaned DataFrame
        alongside summary statistics.
        """
        if meta_cols is None:
            meta_cols = ["compound_id", "is_boron_analogue"]

        actual_meta = [c for c in meta_cols if c in df.columns]

        df_stage1, dropped_na = cls.filter_missing_values(df, actual_meta)
        df_stage2, dropped_var = cls.filter_low_variance(df_stage1, actual_meta, threshold=var_threshold)
        df_stage3, dropped_corr = cls.filter_collinear(df_stage2, actual_meta, threshold=corr_threshold)

        num_retained = len([c for c in df_stage3.columns if c not in actual_meta])

        stats = {
            "initial_numeric_count": len([c for c in df.select_dtypes(include=[np.number]).columns if c not in actual_meta]),
            "dropped_missing": len(dropped_na),
            "dropped_low_variance": len(dropped_var),
            "dropped_collinear": len(dropped_corr),
            "retained_descriptors": num_retained
        }

        return df_stage3, stats
