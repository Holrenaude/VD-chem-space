"""
ML_generation/validate_generation.py
Unified validation and benchmarking suite for de novo VDR ligand generators
(REINVENT 4, Transmol, Mol2Mol, etc.).

Evaluates:
1. Core Generative Quality: Validity, Uniqueness, Novelty, and Internal Diversity
2. Out-of-Sample Regeneration Rate: Exact 2D, Stereoisomer-aware, and Analogue Recovery (@0.70 / @0.85)
3. Rediscovery Profiling:
   - GuacaMol-style continuous rediscovery score (mean max Tanimoto to held-out targets)
   - Structural-class stratified rediscovery (Aliphatic Secosteroids vs. Aromatic Non-Secosteroids)
   - Likelihood/NLL-ranked Top-100 & Top-500 hit recovery
   - Exports exact rediscovered SMILES to CSV for visual inspection
4. Bemis-Murcko Scaffold Diversity & Novelty
5. Structural Alerts (PAINS, Brenk, NIH) & Aromatic Bond Alignment vs. Known VDR Baseline
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, Descriptors, FilterCatalog
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")


# =============================================================================
# MOLECULAR & FINGERPRINT HELPERS
# =============================================================================

def canonicalize(smiles_list: list[str], keep_stereo: bool = False) -> list[str]:
    valid = []
    for smi in smiles_list:
        if not isinstance(smi, str) or not smi.strip():
            continue
        mol = Chem.MolFromSmiles(smi.strip())
        if mol is not None:
            valid.append(Chem.MolToSmiles(mol, isomericSmiles=keep_stereo))
    return valid


def load_smi_file(path: str, keep_stereo: bool = False) -> set[str]:
    if not path or not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip().split()[0] for line in f if line.strip()]
    return set(canonicalize(lines, keep_stereo=keep_stereo))


def compute_morgan_fps(smiles_list: list[str], radius: int = 2, n_bits: int = 2048):
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits))
    return fps


def extract_scaffolds(smiles_list: list[str]) -> set[str]:
    scaffolds = set()
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            if scaf:
                scaffolds.add(scaf)
    return scaffolds


# =============================================================================
# STRUCTURAL ALERTS & PHYSICOCHEMICAL / AROMATIC METRICS
# =============================================================================

def build_alert_catalog(catalogs=("PAINS", "BRENK", "NIH")):
    params = FilterCatalog.FilterCatalogParams()
    catalog_map = {
        "PAINS": FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS,
        "BRENK": FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK,
        "NIH": FilterCatalog.FilterCatalogParams.FilterCatalogs.NIH,
    }
    for name in catalogs:
        if name.upper() in catalog_map:
            params.AddCatalog(catalog_map[name.upper()])
    return FilterCatalog.FilterCatalog(params)


def evaluate_structural_alerts(smiles_list: list[str]) -> dict:
    pains_cat = build_alert_catalog(("PAINS",))
    brenk_cat = build_alert_catalog(("BRENK",))
    combined_cat = build_alert_catalog(("PAINS", "BRENK", "NIH"))

    total = len(smiles_list)
    if total == 0:
        return {"Alert_Free_%": 0.0, "PAINS_Free_%": 0.0, "Brenk_Free_%": 0.0}

    clean_all, clean_pains, clean_brenk = 0, 0, 0
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        if not combined_cat.HasMatch(mol):
            clean_all += 1
        if not pains_cat.HasMatch(mol):
            clean_pains += 1
        if not brenk_cat.HasMatch(mol):
            clean_brenk += 1

    return {
        "Alert_Free_%": round((clean_all / total) * 100, 2),
        "PAINS_Free_%": round((clean_pains / total) * 100, 2),
        "Brenk_Free_%": round((clean_brenk / total) * 100, 2),
    }


def compute_physchem_and_aromatic(smiles_list: list[str]) -> pd.DataFrame:
    records = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            total_bonds = mol.GetNumBonds()
            arom_bonds = sum(1 for b in mol.GetBonds() if b.GetIsAromatic())
            records.append({
                "MW": Descriptors.MolWt(mol),
                "LogP": Descriptors.MolLogP(mol),
                "HBD": Descriptors.NumHDonors(mol),
                "HBA": Descriptors.NumHAcceptors(mol),
                "TPSA": Descriptors.TPSA(mol),
                "NumAromaticBonds": arom_bonds,
                "AromaticBondFraction": (arom_bonds / total_bonds) if total_bonds > 0 else 0.0,
                "HasAromaticSystem": arom_bonds > 0,
            })
    return pd.DataFrame(records)


# =============================================================================
# REDISCOVERY PROFILING
# =============================================================================

def compute_rediscovery_metrics(
    gen_df: pd.DataFrame,
    smi_col: str,
    unique_2d: set[str],
    test_2d: set[str],
    all_ref_2d: set[str],
) -> dict:
    """
    Computes:
    - GuacaMol continuous rediscovery score (mean max Tanimoto to held-out targets)
    - Class-stratified exact rediscovery (Aliphatic Secosteroids vs. Aromatic Non-Secosteroids)
    - NLL/Score-ranked Top-100 & Top-500 hit counts
    - Exact rediscovered SMILES list
    """
    if not test_2d or not unique_2d:
        return {
            "GuacaMol_Rediscovery_Score": "N/A",
            "Secosteroid_Regen_2D_%": "N/A",
            "Aromatic_Regen_2D_%": "N/A",
            "Test_Seco_Hits": "N/A",
            "Test_Arom_Hits": "N/A",
            "Top100_Ref_Hits": "N/A",
            "Top500_Ref_Hits": "N/A",
            "Rediscovered_SMILES": [],
            "Max_Test_Sims": [],
        }

    gen_list = list(unique_2d)
    gen_fps = compute_morgan_fps(gen_list)

    # 1. Continuous GuacaMol Rediscovery Score
    test_list = list(test_2d)
    test_fps = compute_morgan_fps(test_list)
    max_sims = [max(DataStructs.BulkTanimotoSimilarity(t_fp, gen_fps)) for t_fp in test_fps] if gen_fps else [0.0]
    guacamol_score = round(float(np.mean(max_sims)), 4)

    # 2. Structural-Class Stratified Rediscovery (Aliphatic vs. Aromatic)
    test_seco, test_arom = set(), set()
    for smi in test_2d:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        has_arom = any(b.GetIsAromatic() for b in mol.GetBonds())
        if has_arom:
            test_arom.add(smi)
        else:
            test_seco.add(smi)

    seco_hits = unique_2d & test_seco
    arom_hits = unique_2d & test_arom
    seco_regen = (len(seco_hits) / len(test_seco) * 100) if test_seco else 0.0
    arom_regen = (len(arom_hits) / len(test_arom) * 100) if test_arom else 0.0

    # 3. Likelihood-Ranked Top-K Rediscovery (if NLL / Score / Loss column is present)
    score_col = next((c for c in gen_df.columns if c.lower() in {"nll", "score", "loss"}), None)
    top100_hits, top500_hits = "N/A", "N/A"
    if score_col is not None:
        ascending = score_col.lower() in {"nll", "loss"}
        ranked_df = gen_df.dropna(subset=[smi_col, score_col]).sort_values(by=score_col, ascending=ascending)
        ranked_canon = []
        seen = set()
        for raw in ranked_df[smi_col]:
            c = canonicalize([raw], keep_stereo=False)
            if c and c[0] not in seen:
                seen.add(c[0])
                ranked_canon.append(c[0])
        top100_hits = sum(1 for s in ranked_canon[:100] if s in all_ref_2d)
        top500_hits = sum(1 for s in ranked_canon[:500] if s in all_ref_2d)

    rediscovered = sorted(unique_2d & test_2d)

    return {
        "GuacaMol_Rediscovery_Score": guacamol_score,
        "Secosteroid_Regen_2D_%": round(seco_regen, 2),
        "Aromatic_Regen_2D_%": round(arom_regen, 2),
        "Test_Seco_Hits": f"{len(seco_hits)}/{len(test_seco)}",
        "Test_Arom_Hits": f"{len(arom_hits)}/{len(test_arom)}",
        "Top100_Ref_Hits": top100_hits,
        "Top500_Ref_Hits": top500_hits,
        "Rediscovered_SMILES": rediscovered,
        "Max_Test_Sims": max_sims,
    }


# =============================================================================
# DIAGNOSTIC PLOTTING
# =============================================================================

def plot_validation_diagnostics(
    model_name: str,
    nn_sims: list[float],
    max_test_sims: list[float],
    gen_props: pd.DataFrame,
    ref_props: pd.DataFrame,
    fig_path: str,
):
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Panel 1: Nearest-Neighbor Tanimoto (Generated -> VDR) & Held-Out Target Recovery (Test -> Generated)
    axes[0, 0].hist(nn_sims, bins=30, density=True, alpha=0.6, color="#2b6cb0", edgecolor="black", label="Gen -> VDR Space")
    if max_test_sims:
        axes[0, 0].hist(max_test_sims, bins=25, density=True, alpha=0.5, color="#38a169", edgecolor="black", label="Held-Out Target Recovery")
    axes[0, 0].axvline(0.70, color="#d69e2e", linestyle="--", linewidth=1.3, label="Analogue Threshold (0.70)")
    axes[0, 0].axvline(0.85, color="#c53030", linestyle="--", linewidth=1.3, label="Close Match (0.85)")
    axes[0, 0].set_title(f"{model_name}: Similarity & Target Rediscovery Profile", fontweight="bold")
    axes[0, 0].set_xlabel("Max ECFP4 Tanimoto Similarity")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].legend(fontsize=8)

    # Panel 2: MW vs cLogP Chemical Space Overlap
    axes[0, 1].scatter(ref_props["MW"], ref_props["LogP"], alpha=0.45, s=22, color="#718096", label="Known VDR Ligands")
    axes[0, 1].scatter(gen_props["MW"], gen_props["LogP"], alpha=0.45, s=22, color="#dd6b20", label=f"{model_name} Generated")
    axes[0, 1].set_title(f"{model_name}: Physicochemical Space (MW vs. LogP)", fontweight="bold")
    axes[0, 1].set_xlabel("Molecular Weight (Da)")
    axes[0, 1].set_ylabel("cLogP")
    axes[0, 1].legend(fontsize=8)

    # Panel 3: Aromatic Bond Count Distribution (Secosteroid vs Non-Secosteroid Balance)
    max_arom = int(max(ref_props["NumAromaticBonds"].max(), gen_props["NumAromaticBonds"].max(), 1))
    bins = np.arange(0, max_arom + 2) - 0.5
    axes[1, 0].hist(ref_props["NumAromaticBonds"], bins=bins, density=True, alpha=0.55, color="#718096", edgecolor="black", label="Known VDR Ligands")
    axes[1, 0].hist(gen_props["NumAromaticBonds"], bins=bins, density=True, alpha=0.55, color="#319795", edgecolor="black", label=f"{model_name} Generated")
    axes[1, 0].set_title(f"{model_name}: Aromatic Bond Count Distribution", fontweight="bold")
    axes[1, 0].set_xlabel("Number of Aromatic Bonds")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].legend(fontsize=8)

    # Panel 4: TPSA Alignment
    axes[1, 1].hist(ref_props["TPSA"], bins=25, density=True, alpha=0.55, color="#718096", edgecolor="black", label="Known VDR Ligands")
    axes[1, 1].hist(gen_props["TPSA"], bins=25, density=True, alpha=0.55, color="#805ad5", edgecolor="black", label=f"{model_name} Generated")
    axes[1, 1].set_title(f"{model_name}: Polar Surface Area (TPSA) Alignment", fontweight="bold")
    axes[1, 1].set_xlabel("TPSA (Å²)")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()


# =============================================================================
# MAIN VALIDATION ORCHESTRATOR
# =============================================================================

def validate_model(
    model_name: str,
    generated_csv: str,
    train_smi: str = "data/processed/vdr_train.smi",
    test_smi: str = "data/processed/vdr_test_holdout.smi",
    report_csv: str = "data/processed/ml_generation_benchmarks.csv",
) -> dict:
    if not os.path.exists(generated_csv):
        raise FileNotFoundError(f"Generated CSV not found: {generated_csv}")

    gen_df = pd.read_csv(generated_csv)
    smi_col = next((c for c in gen_df.columns if "smiles" in c.lower()), gen_df.columns[0])
    raw_gen = gen_df[smi_col].tolist()
    total_sampled = len(raw_gen)

    # 1. Canonicalize (2D graph-level and stereo-level)
    valid_2d = canonicalize(raw_gen, keep_stereo=False)
    valid_stereo = canonicalize(raw_gen, keep_stereo=True)
    unique_2d = set(valid_2d)
    unique_stereo = set(valid_stereo)

    # 2. Load Reference Sets
    train_2d = load_smi_file(train_smi, keep_stereo=False)
    test_2d = load_smi_file(test_smi, keep_stereo=False)
    test_stereo = load_smi_file(test_smi, keep_stereo=True)
    all_ref_2d = train_2d | test_2d

    # 3. Core Generative Quality
    validity = (len(valid_2d) / total_sampled * 100) if total_sampled else 0.0
    uniqueness = (len(unique_2d) / len(valid_2d) * 100) if valid_2d else 0.0
    novel_2d = unique_2d - all_ref_2d
    novelty = (len(novel_2d) / len(unique_2d) * 100) if unique_2d else 0.0

    # 4. Scaffold Metrics
    gen_scaffolds = extract_scaffolds(list(unique_2d))
    ref_scaffolds = extract_scaffolds(list(all_ref_2d))
    novel_scaffolds = gen_scaffolds - ref_scaffolds

    # 5. Fingerprints, Nearest-Neighbor Similarity & Internal Diversity
    gen_fps = compute_morgan_fps(list(unique_2d))
    ref_fps = compute_morgan_fps(list(all_ref_2d))

    nn_sims = [max(DataStructs.BulkTanimotoSimilarity(g_fp, ref_fps)) for g_fp in gen_fps] if (gen_fps and ref_fps) else [0.0]

    sample_fps = gen_fps[:1000]
    pw_sims = []
    for i in range(len(sample_fps)):
        pw_sims.extend(DataStructs.BulkTanimotoSimilarity(sample_fps[i], sample_fps[i + 1 :]))
    internal_div = 1.0 - float(np.mean(pw_sims)) if pw_sims else 0.0

    # 6. Train Recall vs. Out-of-Sample Regeneration Rate & Rediscovery
    train_recall = (len(unique_2d & train_2d) / len(train_2d) * 100) if train_2d else 0.0
    redisc = compute_rediscovery_metrics(gen_df, smi_col, unique_2d, test_2d, all_ref_2d)

    exact_regen_2d = np.nan
    exact_regen_stereo = np.nan
    analogue_regen_70 = np.nan
    analogue_regen_85 = np.nan
    exact_hits_count = 0

    if test_2d:
        hits_2d = unique_2d & test_2d
        hits_stereo = unique_stereo & test_stereo
        exact_hits_count = len(hits_2d)
        exact_regen_2d = (len(hits_2d) / len(test_2d)) * 100
        exact_regen_stereo = (len(hits_stereo) / len(test_stereo)) * 100

        max_test_sims = redisc["Max_Test_Sims"]
        analogue_regen_70 = (sum(s >= 0.70 for s in max_test_sims) / len(test_2d)) * 100
        analogue_regen_85 = (sum(s >= 0.85 for s in max_test_sims) / len(test_2d)) * 100

    # 7. Structural Alerts (PAINS, Brenk, NIH)
    gen_alerts = evaluate_structural_alerts(list(unique_2d))
    ref_alerts = evaluate_structural_alerts(list(all_ref_2d))

    # 8. Physicochemical & Aromatic Bond Alignment
    gen_props = compute_physchem_and_aromatic(list(unique_2d))
    ref_props = compute_physchem_and_aromatic(list(all_ref_2d))

    gen_arom_pct = round(gen_props["HasAromaticSystem"].mean() * 100, 2) if not gen_props.empty else 0.0
    ref_arom_pct = round(ref_props["HasAromaticSystem"].mean() * 100, 2) if not ref_props.empty else 0.0
    gen_mean_arom = round(gen_props["NumAromaticBonds"].mean(), 2) if not gen_props.empty else 0.0
    ref_mean_arom = round(ref_props["NumAromaticBonds"].mean(), 2) if not ref_props.empty else 0.0

    fig_path = f"data/figures/{model_name.lower()}_validation.png"
    if not gen_props.empty and not ref_props.empty:
        plot_validation_diagnostics(
            model_name,
            nn_sims,
            redisc["Max_Test_Sims"],
            gen_props,
            ref_props,
            fig_path,
        )

    # Export exact rediscovered SMILES to CSV if any were hit
    hits_csv_path = f"data/reports/{model_name.lower()}_rediscovered_hits.csv"
    if redisc["Rediscovered_SMILES"]:
        os.makedirs(os.path.dirname(hits_csv_path), exist_ok=True)
        pd.DataFrame({"rediscovered_canonical_smiles": redisc["Rediscovered_SMILES"]}).to_csv(hits_csv_path, index=False)

    metrics = {
        "Model": model_name,
        "Sampled": total_sampled,
        "Valid_Unique": len(unique_2d),
        "Validity_%": round(validity, 2),
        "Uniqueness_%": round(uniqueness, 2),
        "Novelty_%": round(novelty, 2),
        "Internal_Div": round(internal_div, 3),
        "Mean_NN_Tanimoto": round(float(np.mean(nn_sims)), 3),
        "Unique_Scaffolds": len(gen_scaffolds),
        "Novel_Scaffolds": len(novel_scaffolds),
        "Train_Recall_%": round(train_recall, 2),
        "Test_Hits_2D": exact_hits_count if test_2d else "N/A",
        "Exact_Regen_2D_%": round(exact_regen_2d, 2) if test_2d else "N/A",
        "Exact_Regen_Stereo_%": round(exact_regen_stereo, 2) if test_2d else "N/A",
        "GuacaMol_Rediscovery_Score": redisc["GuacaMol_Rediscovery_Score"],
        "Secosteroid_Regen_2D_%": redisc["Secosteroid_Regen_2D_%"],
        "Aromatic_Regen_2D_%": redisc["Aromatic_Regen_2D_%"],
        "Analogue_Regen_0.70_%": round(analogue_regen_70, 2) if test_2d else "N/A",
        "Analogue_Regen_0.85_%": round(analogue_regen_85, 2) if test_2d else "N/A",
        "Top100_Ref_Hits": redisc["Top100_Ref_Hits"],
        "Top500_Ref_Hits": redisc["Top500_Ref_Hits"],
        "Alert_Free_%": gen_alerts["Alert_Free_%"],
        "PAINS_Free_%": gen_alerts["PAINS_Free_%"],
        "Brenk_Free_%": gen_alerts["Brenk_Free_%"],
        "Ref_VDR_Alert_Free_%": ref_alerts["Alert_Free_%"],
        "Has_Aromatic_%": gen_arom_pct,
        "Ref_Has_Aromatic_%": ref_arom_pct,
        "Mean_Aromatic_Bonds": gen_mean_arom,
        "Ref_Mean_Aromatic_Bonds": ref_mean_arom,
    }

    # Format & Save Plain-Text Audit Report
    report_lines = [
        "=" * 70,
        f"  VALIDATION & REDISCOVERY BENCHMARK REPORT: {model_name}",
        "=" * 70,
        f"  {'Generated CSV':<30}: {generated_csv}",
        f"  {'Total Sampled':<30}: {metrics['Sampled']}",
        f"  {'Valid & Unique (2D)':<30}: {metrics['Valid_Unique']}",
        f"  {'Validity (%)':<30}: {metrics['Validity_%']}%",
        f"  {'Uniqueness (%)':<30}: {metrics['Uniqueness_%']}%",
        f"  {'Novelty (%)':<30}: {metrics['Novelty_%']}%",
        f"  {'Internal Diversity':<30}: {metrics['Internal_Div']}",
        f"  {'Mean NN Tanimoto to VDR':<30}: {metrics['Mean_NN_Tanimoto']}",
        "-" * 70,
        "  SCAFFOLD, REGENERATION & REDISCOVERY METRICS",
        "-" * 70,
        f"  {'Unique Murcko Scaffolds':<30}: {metrics['Unique_Scaffolds']}",
        f"  {'Novel Murcko Scaffolds':<30}: {metrics['Novel_Scaffolds']}",
        f"  {'Train Set Recall (%)':<30}: {metrics['Train_Recall_%']}%",
        f"  {'Held-Out Test Hits (2D)':<30}: {metrics['Test_Hits_2D']} / {len(test_2d) if test_2d else 0}",
        f"  {'Exact Regeneration 2D (%)':<30}: {metrics['Exact_Regen_2D_%']}%",
        f"  {'Exact Regen Stereo (%)':<30}: {metrics['Exact_Regen_Stereo_%']}%",
        f"  {'GuacaMol Rediscovery Score':<30}: {metrics['GuacaMol_Rediscovery_Score']}",
        f"  {'Secosteroid Regen 2D (%)':<30}: {metrics['Secosteroid_Regen_2D_%']}%  ({redisc['Test_Seco_Hits']})",
        f"  {'Aromatic Non-Seco Regen (%)':<30}: {metrics['Aromatic_Regen_2D_%']}%  ({redisc['Test_Arom_Hits']})",
        f"  {'Analogue Recovery (>=0.70)':<30}: {metrics['Analogue_Regen_0.70_%']}%",
        f"  {'Analogue Recovery (>=0.85)':<30}: {metrics['Analogue_Regen_0.85_%']}%",
        f"  {'Top-100 / Top-500 Ref Hits':<30}: {metrics['Top100_Ref_Hits']} / {metrics['Top500_Ref_Hits']}",
        "-" * 70,
        "  MEDICINAL CHEMISTRY & AROMATICITY ALIGNMENT",
        "-" * 70,
        f"  {'Alert-Free (PAINS/Brenk/NIH)':<30}: {metrics['Alert_Free_%']}%  (Ref VDR: {metrics['Ref_VDR_Alert_Free_%']}%)",
        f"  {'PAINS-Free Pass Rate':<30}: {metrics['PAINS_Free_%']}%",
        f"  {'Brenk-Free Pass Rate':<30}: {metrics['Brenk_Free_%']}%",
        f"  {'Molecules w/ Aromatic Bonds':<30}: {metrics['Has_Aromatic_%']}%  (Ref VDR: {metrics['Ref_Has_Aromatic_%']}%)",
        f"  {'Mean Aromatic Bonds / Mol':<30}: {metrics['Mean_Aromatic_Bonds']}  (Ref VDR: {metrics['Ref_Mean_Aromatic_Bonds']})",
        "-" * 70,
        f"  {'Diagnostic Plot':<30}: {fig_path}",
        f"  {'Rediscovered Hits CSV':<30}: {hits_csv_path if redisc['Rediscovered_SMILES'] else 'None'}",
        "=" * 70,
    ]
    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    txt_report_path = f"data/reports/{model_name.lower()}_validation.txt"
    os.makedirs(os.path.dirname(txt_report_path), exist_ok=True)
    with open(txt_report_path, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")
    print(f"Saved validation report: {txt_report_path}")

    # Append/update unified benchmark leaderboard CSV
    os.makedirs(os.path.dirname(report_csv), exist_ok=True)
    new_row = pd.DataFrame([metrics])
    if os.path.exists(report_csv):
        existing = pd.read_csv(report_csv)
        existing = existing[existing["Model"] != model_name]
        pd.concat([existing, new_row], ignore_index=True).to_csv(report_csv, index=False)
    else:
        new_row.to_csv(report_csv, index=False)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate generated VDR ligand libraries.")
    parser.add_argument("--model", required=True, help="Model identifier (e.g., REINVENT4_TL_v1)")
    parser.add_argument("--csv", required=True, help="Path to generated CSV file")
    parser.add_argument("--train", default="data/processed/vdr_train.smi", help="Path to training SMILES")
    parser.add_argument("--test", default="data/processed/vdr_test_holdout.smi", help="Path to held-out test SMILES")
    args = parser.parse_args()

    validate_model(
        model_name=args.model,
        generated_csv=args.csv,
        train_smi=args.train,
        test_smi=args.test,
    )
