"""
algorithms/visualization_druglikeness.py
Generates Bickerton/Nature Chemistry style property histograms color-coded by VDR structural category.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_druglikeness_histograms(
    druglikeness_csv: str = "data/processed/vdr_druglikeness_np.csv",
    curated_csv: str = "data/processed/vdr_curated_library.csv",
    output_dir: str = "data/figures"
):
    if not os.path.exists(druglikeness_csv) or not os.path.exists(curated_csv):
        print("Required CSVs missing. Ensure Step 1 and Step 2 are completed.")
        return

    # Load data
    druglike_df = pd.read_csv(druglikeness_csv)
    curated_df = pd.read_csv(curated_csv)
    
    # Merge on compound_id to map the 'category' to the physicochemical properties
    df = pd.merge(druglike_df, curated_df[['compound_id', 'category']], on='compound_id')

    properties = ['MolWt', 'MolLogP', 'NumHDonors', 'NumHAcceptors', 'TPSA', 'NumRotatableBonds']
    titles = ['Molecular Weight', 'LogP', 'HBD', 'HBA', 'Polar Surface Area', 'Rotatable Bonds']
    
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, (prop, title) in enumerate(zip(properties, titles)):
        if prop not in df.columns:
            continue
            
        sns.histplot(
            data=df, 
            x=prop, 
            hue='category', 
            element='step', 
            stat='density',       
            common_norm=False,    
            ax=axes[i],
            alpha=0.4,
            linewidth=1.5
        )
        
        axes[i].set_title(title, fontsize=14, fontweight="bold")
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Density', fontsize=12)

    plt.tight_layout()
    
    png_path = os.path.join(output_dir, "druglikeness_histograms.png")
    pdf_path = os.path.join(output_dir, "druglikeness_histograms.pdf")
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved property distribution histograms to:\n - {png_path}\n - {pdf_path}")

if __name__ == "__main__":
    plot_druglikeness_histograms()
