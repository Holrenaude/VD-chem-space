import os
import pandas as pd

# Import from our modular algorithms
from algorithms.temporal_integration import load_and_integrate_vdr_data
from algorithms.pca_analysis import run_pca_analysis
from algorithms.nonlinear_projections import run_nonlinear_projections
from algorithms.statistical_testing import run_statistical_analysis
from algorithms.visualisation import (
    plot_chemical_space_maps,
    plot_spearman_heatmap,
    plot_top_descriptor_distributions
)

def execute_step04():
    print("=" * 60)
    print("[STEP 4] Chemical Space Visualisation & Correlation Analysis")
    print("=" * 60)
    
    # ---------------------------------------------------------
    # Module 1: Data Integration
    # ---------------------------------------------------------
    print("\n--- 1. Loading & Integrating Data ---")
    master_df = load_and_integrate_vdr_data(
        curated_path="data/processed/vdr_curated_library.csv",
        descriptors_path="data/processed/vdr_descriptors_filtered.csv",
        clusters_path="data/processed/vdr_butina_clusters.csv",
        raw_path="data/raw/VDR.csv"
    )
    
    # Identify the primary cluster column dynamically (defaults to the 0.4 cutoff if present)
    cluster_cols = [c for c in master_df.columns if c.startswith("cluster_")]
    primary_cluster_col = "cluster_rdkit_topo_cutoff_0.4" if "cluster_rdkit_topo_cutoff_0.4" in cluster_cols else cluster_cols[0]

    # ---------------------------------------------------------
    # Module 2 & 3: Spatial Projections
    # ---------------------------------------------------------
    print("\n--- 2. Computing Spatial Projections ---")
    pca_coords_df, pca_variance_df, _ = run_pca_analysis(master_df, n_components=5)
    nonlinear_coords_df = run_nonlinear_projections(master_df)
    
    # Merge all coordinates with critical metadata
    metadata_cols = ["compound_id", "category", primary_cluster_col, "figure_num"]
    coords_master = master_df[metadata_cols].merge(
        pca_coords_df, on="compound_id", how="inner"
    ).merge(
        nonlinear_coords_df, on="compound_id", how="inner"
    )
    
    # ---------------------------------------------------------
    # Module 4: Statistical Testing
    # ---------------------------------------------------------
    print("\n--- 3. Running Statistical Tests ---")
    spearman_corr_df, kw_results_df, dunn_matrices_dict = run_statistical_analysis(master_df)
    
    # ---------------------------------------------------------
    # Module 5: Visualisation
    # ---------------------------------------------------------
    print("\n--- 4. Generating Figures ---")
    plot_chemical_space_maps(coords_master, cluster_col=primary_cluster_col, output_dir="data/figures")
    plot_spearman_heatmap(spearman_corr_df, output_dir="data/figures")
    plot_top_descriptor_distributions(master_df, kw_results_df, top_n=4, output_dir="data/figures")
    
    # ---------------------------------------------------------
    # Data Export
    # ---------------------------------------------------------
    print("\n--- 5. Exporting Results ---")
    os.makedirs("data/processed", exist_ok=True)
    
    # Core DataFrames
    coords_master.to_csv("data/processed/vdr_chemical_space_coordinates.csv", index=False)
    pca_variance_df.to_csv("data/processed/vdr_pca_variance.csv", index=False)
    spearman_corr_df.to_csv("data/processed/vdr_descriptor_spearman_matrix.csv")
    kw_results_df.to_csv("data/processed/vdr_kruskal_wallis_results.csv", index=False)
    
    # Dunn's Post-Hoc Pairwise Matrices
    dunn_dir = "data/processed/dunn_posthoc_matrices"
    os.makedirs(dunn_dir, exist_ok=True)
    for feature, df_dunn in dunn_matrices_dict.items():
        safe_name = "".join(c if c.isalnum() else "_" for c in feature)
        df_dunn.to_csv(os.path.join(dunn_dir, f"dunn_bh_{safe_name}.csv"))
    
    print(f"[SUCCESS] Step 4 complete. Saved core outputs to data/processed/ and {len(dunn_matrices_dict)} Dunn matrices to {dunn_dir}/")


if __name__ == "__main__":
    execute_step04()
