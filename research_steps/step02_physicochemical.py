"""
research_steps/step02_physicochemical.py
Executes Step 2 of the VDR Chemical Space Profiling pipeline:
1. Loads 'data/processed/vdr_curated_library.csv'.
2. Calculates full 200+ RDKit 2D descriptors via RDKitDescriptorEngine -> exports 'vdr_descriptors_full.csv'.
3. Filters descriptors via 3-stage DescriptorFilter (missing, low-variance, collinear |r| > 0.90) -> exports 'vdr_descriptors_filtered.csv'.
4. Calculates Lipinski, Veber, QED, and NP-likeness scores via DrugLikenessProfiler -> exports 'vdr_druglikeness_np.csv'.
5. Prints a comprehensive execution audit summary.
"""

import os
import pandas as pd
from algorithms.rdkit_descriptors import RDKitDescriptorEngine
from algorithms.descriptor_filtering import DescriptorFilter
from algorithms.drug_likeness_np import DrugLikenessProfiler


def run_physicochemical_step(
    input_csv_path: str = "data/processed/vdr_curated_library.csv",
    out_full_desc_path: str = "data/processed/vdr_descriptors_full.csv",
    out_filt_desc_path: str = "data/processed/vdr_descriptors_filtered.csv",
    out_druglikeness_path: str = "data/processed/vdr_druglikeness_np.csv"
):
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Curated library not found at '{input_csv_path}'. Please run Step 1 first.")

    print(f"Loading curated library from '{input_csv_path}'...")
    curated_df = pd.read_csv(input_csv_path)
    total_compounds = len(curated_df)

    # 1. Full 200+ RDKit 2D Descriptor Engine
    print("Computing 200+ RDKit 2D descriptors for all compounds...")
    desc_engine = RDKitDescriptorEngine()
    full_desc_df = desc_engine.profile_dataframe(curated_df, smiles_col="std_smiles", id_col="compound_id")

    os.makedirs(os.path.dirname(out_full_desc_path), exist_ok=True)
    full_desc_df.to_csv(out_full_desc_path, index=False)
    print(f"Saved full descriptor matrix: {out_full_desc_path}")

    # 2. 3-Stage Data-Driven Descriptor Filtering Pipeline
    print("Applying 3-stage descriptor filtering pipeline (Missing, Low Variance, Collinearity |r| > 0.90)...")
    filt_desc_df, filter_stats = DescriptorFilter.run_full_filter_pipeline(
        df=full_desc_df,
        meta_cols=["compound_id", "is_boron_analogue"],
        corr_threshold=0.90,
        var_threshold=1e-5
    )

    filt_desc_df.to_csv(out_filt_desc_path, index=False)
    print(f"Saved filtered descriptor matrix: {out_filt_desc_path}")

    # 3. Drug-Likeness and Natural Product Profiling
    print("Evaluating Lipinski, Veber, QED, and NP-likeness scores...")
    drug_profiler = DrugLikenessProfiler()
    druglikeness_df = drug_profiler.profile_dataframe(curated_df, smiles_col="std_smiles", id_col="compound_id")

    druglikeness_df.to_csv(out_druglikeness_path, index=False)
    print(f"Saved drug-likeness & NP-score matrix: {out_druglikeness_path}")

    # 4. Print Execution Audit Summary
    boron_count = full_desc_df["is_boron_analogue"].sum() if "is_boron_analogue" in full_desc_df.columns else 0

    print("\n" + "=" * 55)
    print("       STEP 2 PHYSICOCHEMICAL PROFILING AUDIT")
    print("=" * 55)
    print(f"Total Compounds Processed:       {total_compounds}")
    print(f"Carborane / Boron Analogues:     {boron_count}")
    print(f"Initial RDKit Descriptors:       {filter_stats['initial_numeric_count']}")
    print(f"Stage 1 Dropped (Uncomputable):  {filter_stats['dropped_missing']}")
    print(f"Stage 2 Dropped (Low Variance):  {filter_stats['dropped_low_variance']}")
    print(f"Stage 3 Dropped (Collinear |r|>0.9): {filter_stats['dropped_collinear']}")
    print(f"Retained Informative Descriptors:{filter_stats['retained_descriptors']}")
    print(f"Lipinski Rule Pass Rate:         {(druglikeness_df['Lipinski_Violations'] == 0).mean() * 100:.1f}%")
    print(f"Veber Criteria Pass Rate:        {druglikeness_df['Veber_Pass'].mean() * 100:.1f}%")
    print(f"Mean QED Score:                  {druglikeness_df['QED'].mean():.3f}")
    print(f"Mean NP-Likeness Score:          {druglikeness_df['NP_Score'].mean():.3f}")
    print("=" * 55)


if __name__ == "__main__":
    run_physicochemical_step()
