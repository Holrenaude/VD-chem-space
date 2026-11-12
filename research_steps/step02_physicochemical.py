"""
research_steps/step02_physicochemical.py
Executes Step 2 of the VDR Chemical Space Profiling pipeline:
1. Loads 'data/processed/vdr_curated_library.csv'.
2. Calculates full 200+ RDKit 2D descriptors via RDKitDescriptorEngine -> exports 'vdr_descriptors_full.csv'.
3. Filters descriptors via 3-stage DescriptorFilter (missing, low-variance, collinear |r| > 0.90) -> exports 'vdr_descriptors_filtered.csv'.
4. Calculates Lipinski, Veber, QED, NP-likeness scores, and Structural Alerts -> exports 'vdr_druglikeness_np.csv'.
5. Prints and saves a comprehensive execution audit summary to 'data/reports/step2_report.txt'.
"""

import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import FilterCatalog

from algorithms.rdkit_descriptors import RDKitDescriptorEngine
from algorithms.descriptor_filtering import DescriptorFilter
from algorithms.drug_likeness_np import DrugLikenessProfiler
from algorithms.visualization_druglikeness import plot_druglikeness_histograms


def compute_structural_alert_flags(smiles_series: pd.Series) -> pd.DataFrame:
    """
    Annotates molecules with PAINS, Brenk, and NIH structural alert flags
    without removing them from the curated VDR dataset.
    """
    def _make_cat(cat_enum):
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(cat_enum)
        return FilterCatalog.FilterCatalog(params)

    pains_cat = _make_cat(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    brenk_cat = _make_cat(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
    nih_cat = _make_cat(FilterCatalog.FilterCatalogParams.FilterCatalogs.NIH)

    records = []
    for smi in smiles_series:
        mol = Chem.MolFromSmiles(str(smi)) if pd.notna(smi) else None
        if mol is None:
            records.append({
                "PAINS_Flag": None,
                "Brenk_Flag": None,
                "NIH_Flag": None,
                "Alert_Descriptions": None,
                "Is_Alert_Free": None,
            })
            continue

        pains_matches = [m.GetDescription() for m in pains_cat.GetMatches(mol)]
        brenk_matches = [m.GetDescription() for m in brenk_cat.GetMatches(mol)]
        nih_matches = [m.GetDescription() for m in nih_cat.GetMatches(mol)]
        all_descs = sorted(set(pains_matches + brenk_matches + nih_matches))

        records.append({
            "PAINS_Flag": len(pains_matches) > 0,
            "Brenk_Flag": len(brenk_matches) > 0,
            "NIH_Flag": len(nih_matches) > 0,
            "Alert_Descriptions": "; ".join(all_descs) if all_descs else "Clean",
            "Is_Alert_Free": len(all_descs) == 0,
        })

    return pd.DataFrame(records)


def run_physicochemical_step(
    input_csv_path: str = "data/processed/vdr_curated_library.csv",
    out_full_desc_path: str = "data/processed/vdr_descriptors_full.csv",
    out_filt_desc_path: str = "data/processed/vdr_descriptors_filtered.csv",
    out_druglikeness_path: str = "data/processed/vdr_druglikeness_np.csv",
    out_report_path: str = "data/reports/step2_report.txt"
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

    # 3. Drug-Likeness, Natural Product Profiling, and Structural Alerts
    print("Evaluating Lipinski, Veber, QED, NP-likeness scores, and Structural Alerts...")
    drug_profiler = DrugLikenessProfiler()
    druglikeness_df = drug_profiler.profile_dataframe(curated_df, smiles_col="std_smiles", id_col="compound_id")

    print("Checking for PAINS, Brenk, and NIH structural alerts...")
    alert_df = compute_structural_alert_flags(curated_df["std_smiles"])
    druglikeness_df = pd.concat([druglikeness_df.reset_index(drop=True), alert_df.reset_index(drop=True)], axis=1)

    druglikeness_df.to_csv(out_druglikeness_path, index=False)
    print(f"Saved drug-likeness & NP-score matrix: {out_druglikeness_path}")

    # 4. Build, Print, and Save Execution Audit Summary
    boron_count = full_desc_df["is_boron_analogue"].sum() if "is_boron_analogue" in full_desc_df.columns else 0

    report_lines = [
        "=" * 55,
        "        STEP 2 PHYSICOCHEMICAL PROFILING AUDIT",
        "=" * 55,
        f"Total Compounds Processed:       {total_compounds}",
        f"Carborane / Boron Analogues:     {boron_count}",
        f"Initial RDKit Descriptors:       {filter_stats['initial_numeric_count']}",
        f"Stage 1 Dropped (Uncomputable):  {filter_stats['dropped_missing']}",
        f"Stage 2 Dropped (Low Variance):  {filter_stats['dropped_low_variance']}",
        f"Stage 3 Dropped (Collinear |r|>0.9): {filter_stats['dropped_collinear']}",
        f"Retained Informative Descriptors:{filter_stats['retained_descriptors']}",
        f"Lipinski Rule Pass Rate:         {(druglikeness_df['Lipinski_Violations'] == 0).mean() * 100:.1f}%",
        f"Veber Criteria Pass Rate:        {druglikeness_df['Veber_Pass'].mean() * 100:.1f}%",
        f"Mean QED Score:                  {druglikeness_df['QED'].mean():.3f}",
        f"Mean NP-Likeness Score:          {druglikeness_df['NP_Score'].mean():.3f}",
        "-" * 55,
        f"Completely Alert-Free:           {druglikeness_df['Is_Alert_Free'].mean() * 100:.1f}%",
        f"PAINS-Free Pass Rate:            {(~druglikeness_df['PAINS_Flag']).mean() * 100:.1f}%",
        f"Brenk-Free Pass Rate:            {(~druglikeness_df['Brenk_Flag']).mean() * 100:.1f}%",
        "=" * 55,
    ]
    report_text = "\n".join(report_lines)

    print("\n" + report_text)

    os.makedirs(os.path.dirname(out_report_path), exist_ok=True)
    with open(out_report_path, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")
    print(f"Saved Step 2 audit report: {out_report_path}")

    print("Generating property distribution histograms...")
    plot_druglikeness_histograms(
        druglikeness_csv=out_druglikeness_path,
        curated_csv=input_csv_path
    )


if __name__ == "__main__":
    run_physicochemical_step()
