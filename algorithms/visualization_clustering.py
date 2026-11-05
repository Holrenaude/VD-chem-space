"""
algorithms/visualization_clustering.py
Generates chemical space clustering, fingerprint redundancy, and scaffold charts.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_clustering_visualizations(
    redundancy_csv: str = "data/processed/vdr_fingerprint_redundancy.csv",
    scaffolds_csv: str = "data/processed/vdr_scaffold_assignments.csv",
    clusters_csv: str = "data/processed/vdr_butina_clusters.csv",
    output_dir: str = "data/figures/step03"
):
    # Ensure the new step03 nested directory exists
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")
    
    # ---------------------------------------------------------
    # Chart 1: Fingerprint Redundancy Heatmap
    # ---------------------------------------------------------
    if os.path.exists(redundancy_csv):
        try:
            # Assuming the first column contains the fingerprint names
            red_df = pd.read_csv(redundancy_csv, index_col=0)
            
            plt.figure(figsize=(10, 8))
            # Use a hierarchically clustered heatmap to group similar fingerprints
            sns.clustermap(
                red_df, 
                cmap="vlag", 
                vmin=-1, vmax=1, 
                annot=True, 
                fmt=".2f",
                annot_kws={"size": 8},
                figsize=(10, 10)
            )
            plt.title("Fingerprint Correlation (Redundancy) Matrix", pad=20, fontsize=14, fontweight="bold")
            
            out_fp = os.path.join(output_dir, "fingerprint_redundancy_heatmap.png")
            plt.savefig(out_fp, dpi=300, bbox_inches='tight')
            plt.close('all')
            print(f"Saved: {out_fp}")
        except Exception as e:
            print(f"Failed to generate fingerprint heatmap: {e}")

    # ---------------------------------------------------------
    # Chart 2: Top 15 Bemis-Murcko Scaffolds
    # ---------------------------------------------------------
    if os.path.exists(scaffolds_csv):
        try:
            scaff_df = pd.read_csv(scaffolds_csv)
            
            # Find the scaffold column dynamically (usually contains 'scaffold' or 'bm')
            scaff_col = [c for c in scaff_df.columns if 'scaffold' in c.lower() or 'bm' in c.lower()][0]
            
            # Get top 15 scaffolds (ignoring empty/nan if any)
            top_scaffolds = scaff_df[scaff_col].value_counts().head(15)
            
            plt.figure(figsize=(12, 6))
            sns.barplot(x=top_scaffolds.values, y=top_scaffolds.index, palette="viridis")
            plt.title("Top 15 Most Frequent Bemis-Murcko Scaffolds", fontsize=14, fontweight="bold")
            plt.xlabel("Number of Compounds", fontsize=12)
            plt.ylabel("Bemis-Murcko Scaffold (SMILES)", fontsize=12)
            
            out_scaff = os.path.join(output_dir, "top_bemis_murcko_scaffolds.png")
            plt.savefig(out_scaff, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: {out_scaff}")
        except Exception as e:
            print(f"Failed to generate scaffold chart: {e}")

    # ---------------------------------------------------------
    # Chart 3: Cluster Size Distribution (Butina)
    # ---------------------------------------------------------
    if os.path.exists(clusters_csv):
        try:
            clust_df = pd.read_csv(clusters_csv)
            
            # Identify the cluster columns (e.g., cluster_ecfp4_cutoff_0.3)
            cluster_cols = [c for c in clust_df.columns if 'cluster' in c.lower() and 'cutoff' in c.lower()]
            
            if cluster_cols:
                plt.figure(figsize=(14, 6))
                
                # Calculate the sizes of clusters for each cutoff threshold
                cluster_sizes = []
                for col in cluster_cols:
                    sizes = clust_df[col].value_counts()
                    cutoff_name = col.split('cutoff_')[-1] if 'cutoff_' in col else col
                    temp_df = pd.DataFrame({'Cluster Size': sizes.values, 'Distance Cutoff': cutoff_name})
                    cluster_sizes.append(temp_df)
                    
                size_df = pd.concat(cluster_sizes, ignore_index=True)
                
                # Plot distribution using a boxplot with swarmplot overlay for granularity
                sns.boxplot(data=size_df, x="Distance Cutoff", y="Cluster Size", color="lightblue", showfliers=False)
                sns.stripplot(data=size_df, x="Distance Cutoff", y="Cluster Size", color="darkblue", alpha=0.5, jitter=True)
                
                plt.title("Butina Cluster Size Distribution Across Tanimoto Cutoffs", fontsize=14, fontweight="bold")
                plt.xlabel("Tanimoto Distance Cutoff", fontsize=12)
                plt.ylabel("Number of Compounds per Cluster", fontsize=12)
                # Use log scale if there are massive clusters
                if size_df['Cluster Size'].max() > 50:
                    plt.yscale('log')
                    plt.ylabel("Number of Compounds per Cluster (Log Scale)", fontsize=12)
                
                out_clust = os.path.join(output_dir, "cluster_size_distributions.png")
                plt.savefig(out_clust, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved: {out_clust}")
        except Exception as e:
            print(f"Failed to generate cluster size chart: {e}")

if __name__ == "__main__":
    plot_clustering_visualizations()
