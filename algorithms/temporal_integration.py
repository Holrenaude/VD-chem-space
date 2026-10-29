import os
import pandas as pd


def load_and_integrate_vdr_data(
    curated_path: str = "data/processed/vdr_curated_library.csv",
    descriptors_path: str = "data/processed/vdr_descriptors_filtered.csv",
    clusters_path: str = "data/processed/vdr_butina_clusters.csv",
    raw_path: str = "data/raw/VDR.csv",
) -> pd.DataFrame:
    """
    Loads processed datasets from Steps 1-3, checks raw data for temporal metadata,
    calculates decade attributes, and returns a unified master DataFrame merged on compound_id.
    """
    # 1. Verify existence of required processed files
    for path, label in [
        (curated_path, "Curated library"),
        (descriptors_path, "Filtered descriptors"),
        (clusters_path, "Butina clusters"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} not found at: {path}")

    # 2. Load Processed Datasets
    curated_df = pd.read_csv(curated_path)
    descriptors_df = pd.read_csv(descriptors_path)
    clusters_df = pd.read_csv(clusters_path)

    # 3. Handle Temporal Metadata (Publication Year & Decade)
    if os.path.exists(raw_path):
        raw_df = pd.read_csv(raw_path)
        year_col = next((col for col in raw_df.columns if "year" in col.lower()), None)

        if year_col and "compound_id" in raw_df.columns:
            temporal_df = raw_df[["compound_id", year_col]].rename(columns={year_col: "pub_year"})
            curated_df = curated_df.merge(temporal_df, on="compound_id", how="left")
            valid_years = pd.to_numeric(curated_df["pub_year"], errors="coerce")
            curated_df["decade"] = (valid_years // 10 * 10).dropna().astype(int).astype(str) + "s"
            curated_df["decade"] = curated_df["decade"].fillna("Unknown")
        else:
            curated_df["pub_year"] = pd.NA
            curated_df["decade"] = "Unknown"
    else:
        curated_df["pub_year"] = pd.NA
        curated_df["decade"] = "Unknown"

    # 4. Extract Core Metadata Columns
    meta_cols = ["compound_id", "category", "std_smiles", "pub_year", "decade", "figure_num"]
    meta_cols = [c for c in meta_cols if c in curated_df.columns]
    master_df = curated_df[meta_cols].copy()

    # 5. Merge Descriptors and Cluster Assignments via compound_id
    master_df = master_df.merge(descriptors_df, on="compound_id", how="inner")
    master_df = master_df.merge(clusters_df, on="compound_id", how="inner")

    print(
        f"[INFO] Temporal integration complete. Unified master shape: {master_df.shape} ({len(master_df)} compounds)"
    )
    return master_df


if __name__ == "__main__":
    # Test Module 1 locally
    integrated_df = load_and_integrate_vdr_data(
        curated_path="vdr_curated_library.csv",
        descriptors_path="vdr_descriptors_filtered.csv",
        clusters_path="vdr_butina_clusters.csv",
        raw_path="VDR.csv",
    )
    print("Integration Preview:")
    print(integrated_df[["compound_id", "category", "decade"]].head())
