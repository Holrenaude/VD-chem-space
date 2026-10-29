import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.manifold import TSNE
import umap


def generate_morgan_fingerprints(
    smiles_series: pd.Series, radius: int = 2, n_bits: int = 1024
) -> np.ndarray:
    """Generates a binary Morgan fingerprint matrix (ECFP4) from SMILES series."""
    fps = []
    for smi in smiles_series:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            fps.append(np.array(fp))
        else:
            fps.append(np.zeros((n_bits,), dtype=int))
    return np.array(fps)


def run_tsne_projection(
    fps_array: np.ndarray,
    compound_ids: pd.Series,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """Runs t-SNE manifold projection on binary fingerprint matrix using Jaccard (Tanimoto) distance."""
    tsne = TSNE(
        n_components=2,
        metric="jaccard",
        perplexity=perplexity,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    coords = tsne.fit_transform(fps_array)

    tsne_df = pd.DataFrame(coords, columns=["tSNE_1", "tSNE_2"])
    tsne_df["compound_id"] = compound_ids.values
    return tsne_df


def run_umap_projection(
    fps_array: np.ndarray,
    compound_ids: pd.Series,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> pd.DataFrame:
    """Runs UMAP manifold projection on binary fingerprint matrix using Jaccard (Tanimoto) distance."""
    reducer = umap.UMAP(
        n_components=2,
        metric="jaccard",
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    coords = reducer.fit_transform(fps_array)

    umap_df = pd.DataFrame(coords, columns=["UMAP_1", "UMAP_2"])
    umap_df["compound_id"] = compound_ids.values
    return umap_df


def run_nonlinear_projections(master_df: pd.DataFrame) -> pd.DataFrame:
    """Orchestrates ECFP4 fingerprint generation and fits both t-SNE and UMAP projections."""
    print("[INFO] Generating ECFP4 (Morgan radius=2, 1024 bits) fingerprints...")
    fps_array = generate_morgan_fingerprints(master_df["std_smiles"])

    print("[INFO] Computing t-SNE projection (Jaccard distance)...")
    tsne_df = run_tsne_projection(fps_array, master_df["compound_id"])

    print("[INFO] Computing UMAP projection (Jaccard distance)...")
    umap_df = run_umap_projection(fps_array, master_df["compound_id"])

    # Combine non-linear projections
    nonlinear_coords_df = tsne_df.merge(umap_df, on="compound_id", how="inner")
    print(f"[INFO] Non-linear projections complete. Shape: {nonlinear_coords_df.shape}")

    return nonlinear_coords_df


if __name__ == "__main__":
    # Quick execution sanity check using Module 1 output
    from algorithms.temporal_integration import load_and_integrate_vdr_data

    integrated_df = load_and_integrate_vdr_data(
        curated_path="vdr_curated_library.csv",
        descriptors_path="vdr_descriptors_filtered.csv",
        clusters_path="vdr_butina_clusters.csv",
        raw_path="VDR.csv",
    )
    nonlinear_df = run_nonlinear_projections(integrated_df)
    print("\nNon-linear Coordinates Preview:")
    print(nonlinear_df.head())
