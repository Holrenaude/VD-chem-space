import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def run_pca_analysis(
    master_df: pd.DataFrame, n_components: int = 5
) -> tuple[pd.DataFrame, pd.DataFrame, PCA]:
    """Isolates numeric descriptors, applies standard scaling, and performs

    Principal Component Analysis (PCA).

    Parameters:
    -----------
    master_df : pd.DataFrame
        Integrated master dataset containing compound_id and descriptors.
    n_components : int
        Number of principal components to calculate (default: 5).

    Returns:
    --------
    pca_coords_df : pd.DataFrame
        DataFrame with compound_id as index and PC coordinates as columns.
    pca_variance_df : pd.DataFrame
        DataFrame detailing explained variance ratio and cumulative variance per PC.
    pca_obj : PCA
        Fitted scikit-learn PCA object.
    """
    # 1. Identify non-descriptor columns to exclude
    metadata_cols = [
        "compound_id",
        "category",
        "std_smiles",
        "raw_smiles",
        "figure_num",
        "pub_year",
        "decade",
        "is_boron_analogue",
    ]

    # Select only numeric descriptor columns
    descriptor_cols = [
        c
        for c in master_df.columns
        if c not in metadata_cols and not c.startswith("cluster_") and not c.startswith("is_centroid_")
    ]

    X = master_df[descriptor_cols].copy()

    # 2. Apply Standard Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Fit PCA
    pca_obj = PCA(n_components=n_components, random_state=42)
    pca_coords = pca_obj.fit_transform(X_scaled)

    # 4. Construct Coordinate DataFrame
    pc_colnames = [f"PCA_PC{i+1}" for i in range(n_components)]
    pca_coords_df = pd.DataFrame(pca_coords, columns=pc_colnames)
    pca_coords_df["compound_id"] = master_df["compound_id"].values

    # 5. Construct Variance Summary DataFrame
    exp_var = pca_obj.explained_variance_ratio_
    cum_var = exp_var.cumsum()

    pca_variance_df = pd.DataFrame(
        {
            "PC": [f"PC{i+1}" for i in range(n_components)],
            "Explained_Variance_Ratio": exp_var,
            "Cumulative_Variance_Ratio": cum_var,
        }
    )

    print(f"[INFO] PCA complete ({len(descriptor_cols)} descriptors transformed).")
    print(
        f"[INFO] PC1 Variance: {exp_var[0]:.2%}, PC2 Variance: {exp_var[1]:.2%}, Total Explained (PC1-2): {cum_var[1]:.2%}"
    )

    return pca_coords_df, pca_variance_df, pca_obj


if __name__ == "__main__":
    # Quick execution sanity check using Module 1 output
    from algorithms.temporal_integration import load_and_integrate_vdr_data

    integrated_df = load_and_integrate_vdr_data(
        curated_path="vdr_curated_library.csv",
        descriptors_path="vdr_descriptors_filtered.csv",
        clusters_path="vdr_butina_clusters.csv",
        raw_path="VDR.csv",
    )
    coords_df, var_df, _ = run_pca_analysis(integrated_df)
    print("\nPCA Coordinates Preview:")
    print(coords_df.head())
    print("\nPCA Variance Summary:")
    print(var_df)
