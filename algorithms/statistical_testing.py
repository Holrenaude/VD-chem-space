import pandas as pd
from scipy import stats
import statsmodels.stats.multitest as smm
import scikit_posthocs as sp


def compute_spearman_correlation(descriptors_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the Spearman rank correlation matrix for continuous descriptors.
    """
    corr, _ = stats.spearmanr(descriptors_df, axis=0)
    return pd.DataFrame(corr, index=descriptors_df.columns, columns=descriptors_df.columns)


def run_kruskal_dunn_tests(
    master_df: pd.DataFrame, feature_cols: list[str], group_col: str = "category"
) -> tuple[pd.DataFrame, dict]:
    """
    Performs global Kruskal-Wallis test across groups. If significant (p < 0.05), 
    performs Dunn's post hoc test with Benjamini-Hochberg FDR correction.
    
    Parameters:
    -----------
    master_df : pd.DataFrame
        Integrated dataset containing descriptors and grouping column.
    feature_cols : list[str]
        List of continuous descriptor column names to test.
    group_col : str
        The categorical column defining the groups (e.g., 'category').
        
    Returns:
    --------
    kw_results_df : pd.DataFrame
        Summary dataframe of global Kruskal-Wallis statistics and BH-corrected p-values.
    dunn_results : dict
        Dictionary of Dunn's pairwise p-value matrices (BH-corrected) for significant features.
    """
    kw_results = []
    dunn_results = {}
    
    print(f"[INFO] Running Kruskal-Wallis & Dunn's post-hoc across {len(feature_cols)} descriptors grouped by '{group_col}'...")

    for feature in feature_cols:
        # Extract non-null groups for the current feature
        groups = [group[feature].dropna().values for _, group in master_df.groupby(group_col)]
        
        # 1. Global Kruskal-Wallis Omnibus Test (needs at least 2 groups)
        if len(groups) > 1:
            kw_stat, kw_p = stats.kruskal(*groups)
            kw_results.append({
                'Feature': feature, 
                'Kruskal_Stat': kw_stat, 
                'Kruskal_pvalue_raw': kw_p
            })
            
            # 2. Dunn's Post-Hoc Test (Only if KW is significant globally)
            if kw_p < 0.05:
                # p_adjust='fdr_bh' applies the Benjamini-Hochberg correction directly to pairwise p-values
                dunn_df = sp.posthoc_dunn(master_df, val_col=feature, group_col=group_col, p_adjust='fdr_bh')
                dunn_results[feature] = dunn_df
                
    res_df = pd.DataFrame(kw_results)
    
    # 3. Apply BH FDR Correction to the Global KW p-values
    if not res_df.empty:
        reject, pvals_corrected, _, _ = smm.multipletests(
            res_df['Kruskal_pvalue_raw'], alpha=0.05, method='fdr_bh'
        )
        res_df['Kruskal_pvalue_FDR'] = pvals_corrected
        res_df['Significant_Global_Diff'] = reject
        # Sort by most significant features
        res_df = res_df.sort_values(by='Kruskal_pvalue_FDR').reset_index(drop=True)
    
    print(f"[INFO] Found {len(dunn_results)} features with significant global differences requiring Dunn's post-hoc.")
    
    return res_df, dunn_results


def run_statistical_analysis(master_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Orchestrates the statistical testing pipeline on the descriptor matrix."""
    
    # Identify non-descriptor columns to exclude
    metadata_cols = ['compound_id', 'category', 'std_smiles', 'raw_smiles', 
                     'figure_num', 'pub_year', 'decade', 'is_boron_analogue']
    
    descriptor_cols = [c for c in master_df.columns if c not in metadata_cols 
                       and not c.startswith('cluster_') and not c.startswith('is_centroid_')]
    
    # 1. Spearman Correlation
    print("[INFO] Computing Spearman correlation matrix...")
    X_descriptors = master_df[descriptor_cols]
    spearman_corr_df = compute_spearman_correlation(X_descriptors)
    
    # 2. Kruskal-Wallis & Dunn Tests
    kw_results_df, dunn_matrices_dict = run_kruskal_dunn_tests(
        master_df, feature_cols=descriptor_cols, group_col='category'
    )
    
    return spearman_corr_df, kw_results_df, dunn_matrices_dict


if __name__ == "__main__":
    # Quick execution sanity check
    from algorithms.temporal_integration import load_and_integrate_vdr_data
    
    integrated_df = load_and_integrate_vdr_data(
        curated_path="vdr_curated_library.csv", 
        descriptors_path="vdr_descriptors_filtered.csv", 
        clusters_path="vdr_butina_clusters.csv",
        raw_path="VDR.csv"
    )
    
    spearman, kw_res, dunn_dict = run_statistical_analysis(integrated_df)
    print("\nGlobal Kruskal-Wallis Top 5 Significant Features (FDR Corrected):")
    print(kw_res.head())
