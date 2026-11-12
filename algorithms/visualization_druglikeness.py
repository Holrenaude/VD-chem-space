"""
algorithms/visualization_druglikeness.py
Generates Bickerton/Nature Chemistry style overall property histograms across the VDR library.
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
    if not os.path.exists(druglikeness_csv):
        print(f"Required CSV missing at '{druglikeness_csv}'. Ensure Step 2 is completed.")
        return

    # Load data directly (no category merge needed)
    df = pd.read_csv(druglikeness_csv)

    properties = ['MolWt', 'MolLogP', 'NumHDonors', 'NumHAcceptors', 'TPSA', 'NumRotatableBonds']
    titles = ['Molecular Weight', 'LogP', 'HBD', 'HBA', 'Polar Surface Area', 'Rotatable Bonds']

    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, (prop, title) in enumerate(zip(properties, titles)):
        if prop not in df.columns:
            continue

        data = df[prop].dropna()

        sns.histplot(
            data=df,
            x=prop,
            stat='density',
            kde=True,
            color="#2b6cb0",
            edgecolor="black",
            alpha=0.55,
            linewidth=1.2,
            ax=axes[i]
        )

        # Add Mean & Median reference lines
        mean_val = data.mean()
        median_val = data.median()
        axes[i].axvline(mean_val, color="#dd6b20", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.2f}")
        axes[i].axvline(median_val, color="#2f855a", linestyle=":", linewidth=1.5, label=f"Median: {median_val:.2f}")

        axes[i].set_title(title, fontsize=14, fontweight="bold")
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Density', fontsize=12)
        axes[i].legend(fontsize=10)

    plt.tight_layout()

    png_path = os.path.join(output_dir, "druglikeness_histograms.png")
    pdf_path = os.path.join(output_dir, "druglikeness_histograms.pdf")

    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nSaved property distribution histograms to:\n - {png_path}\n - {pdf_path}")


if __name__ == "__main__":
    plot_druglikeness_histograms()
