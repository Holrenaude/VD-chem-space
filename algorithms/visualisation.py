import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication-ready aesthetics
sns.set_theme(style="ticks", context="paper", font_scale=1.2)

def plot_chemical_space_maps(
    coords_df: pd.DataFrame, 
    cluster_col: str, 
    output_dir: str = "data/figures"
):
    """
    Plots a 2x3 grid of chemical space maps (PCA, t-SNE, UMAP).
    Row 1: Colored by Category.
    Row 2: Colored by Cluster.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    projections = [
        ("PCA_PC1", "PCA_PC2", "PCA"),
        ("tSNE_1", "tSNE_2", "t-SNE"),
        ("UMAP_1", "UMAP_2", "UMAP")
    ]
    
    # Define consistent palette for categories
    categories = coords_df["category"].dropna().unique()
    cat_palette = sns.color_palette("Set2", n_colors=len(categories))
    
    for i, (x_col, y_col, title) in enumerate(projections):
        if x_col not in coords_df.columns or y_col not in coords_df.columns:
            continue
            
        # Row 0: Colored by Category
        sns.scatterplot(
            data=coords_df, x=x_col, y=y_col, hue="category", 
            palette=cat_palette, alpha=0.8, s=50, edgecolor="w", ax=axes[0, i]
        )
        axes[0, i].set_title(f"{title} - By Category")
        axes[0, i].legend(title="Category", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Row 1: Colored by Cluster (using a distinct palette, turning off legend if too many clusters)
        n_clusters = coords_df[cluster_col].nunique()
        clus_palette = sns.color_palette("tab20", n_colors=n_clusters)
        
        sns.scatterplot(
            data=coords_df, x=x_col, y=y_col, hue=cluster_col, 
            palette=clus_palette, alpha=0.8, s=50, edgecolor="w", ax=axes[1, i], legend=False
        )
        axes[1, i].set_title(f"{title} - By Cluster")
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, "chemical_space_maps.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved chemical space maps to {out_path}")


def plot_spearman_heatmap(
    corr_df: pd.DataFrame, 
    output_dir: str = "data/figures"
):
    """
    Plots a masked Spearman correlation heatmap for the descriptor matrix.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(
        corr_df, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
        square=True, linewidths=.5, cbar_kws={"shrink": .7}, ax=ax
    )
    
    ax.set_title("Descriptor Spearman Correlation Matrix", pad=20)
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, "descriptor_spearman_heatmap.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved Spearman heatmap to {out_path}")


def plot_top_descriptor_distributions(
    master_df: pd.DataFrame, 
    kw_results_df: pd.DataFrame, 
    top_n: int = 4, 
    output_dir: str = "data/figures"
):
    """
    Creates box plots overlaid with strip plots for the top N most 
    statistically significant descriptors across categories.
    """
    if kw_results_df.empty:
        print("[WARNING] No statistical results provided to plot distributions.")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Isolate the most significant features
    significant_features = kw_results_df[kw_results_df['Significant_Global_Diff'] == True]
    top_features = significant_features.head(top_n)['Feature'].tolist()
    
    if not top_features:
        print("[WARNING] No descriptors showed significant global differences.")
        return

    fig, axes = plt.subplots(1, len(top_features), figsize=(5 * len(top_features), 5))
    if len(top_features) == 1:
        axes = [axes] # Ensure iterable if only 1 feature
        
    categories = master_df["category"].dropna().unique()
    cat_palette = sns.color_palette("Set2", n_colors=len(categories))

    for ax, feature in zip(axes, top_features):
        sns.boxplot(
            data=master_df, x="category", y=feature, 
            color="white", showfliers=False, ax=ax
        )
        sns.stripplot(
            data=master_df, x="category", y=feature, 
            hue="category", palette=cat_palette, alpha=0.6, 
            jitter=True, size=4, legend=False, ax=ax
        )
        ax.set_title(feature)
        ax.set_xlabel("")
        ax.tick_params(axis='x', rotation=45)
        
    plt.tight_layout()
    out_path = os.path.join(output_dir, "top_descriptors_distribution.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved top descriptor distributions to {out_path}")


if __name__ == "__main__":
    print("[INFO] Visualisation module loaded successfully.")
