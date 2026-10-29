"""
research_steps/step01_data_curation.py
Executes Step 1: Data Curation & Standardisation with 4-class structural category labeling.
"""

import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from algorithms.standardisation import ChemicalStandardiser


def classify_vdr_category(smiles: str, figure_num: str) -> str:
    """
    Classifies a VDR ligand into 4 distinct structural categories per Section 2.1:
    1. 'hybrid': Carborane or boron-containing analogues.
    2. 'metabolite': Natural physiological metabolites and primary derivatives (Figure 1).
    3. 'non-secosteroidal': Mimics lacking the conjugated 9,10-seco core (bis-phenyl, LG190178, etc.).
    4. 'analogue': Synthetic secosteroid derivatives with ring or side-chain modifications.
    """
    if not smiles or not isinstance(smiles, str) or pd.isna(smiles):
        return "analogue"

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "analogue"

    # 1. Boron/Carborane analogues -> 'hybrid'
    if any(atom.GetAtomicNum() == 5 for atom in mol.GetAtoms()):
        return "hybrid"

    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    scaff_smi = Chem.MolToSmiles(scaffold, canonical=True)
    generic = MurckoScaffold.MakeScaffoldGeneric(scaffold)
    gen_smi = Chem.MolToSmiles(generic, canonical=True)

    # 2. Physiological Natural Metabolites (Figure 001 compounds)
    if str(figure_num) in ["Figure001", "Figure1"]:
        return "metabolite"

    # 3. Non-secosteroidal mimics lacking 9,10-seco conjugated core
    is_secosteroid_core = ("C1CCCCC1CCC1CCCC2CCCC21" in gen_smi) or ("C1CCC(CCC2CCCC3CCCC32)CC1" in gen_smi)
    if not is_secosteroid_core:
        if "C1CCCCC1" in gen_smi or "c1ccccc1" in scaff_smi:
            return "non-secosteroidal"
        else:
            return "hybrid"

    # 4. Standard Secosteroid Synthetic Analogues
    return "analogue"


def run_curation_pipeline(
    input_csv_path: str = "data/raw/VDR.csv",
    output_csv_path: str = "data/processed/vdr_curated_library.csv",
    output_sdf_path: str = "data/processed/vdr_curated_library.sdf"
):
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Input file not found at '{input_csv_path}'.")

    print(f"Loading raw dataset from {input_csv_path}...")
    raw_df = pd.read_csv(input_csv_path)

    standardiser = ChemicalStandardiser()
    processed_records = []

    print("Running RDKit standardization and category classification pipeline...")
    for idx, row in raw_df.iterrows():
        cmpd_id = row.get("Filename", f"CMPD_{idx+1}")
        smiles = row.get("SMILES", None)

        if not smiles or pd.isna(smiles):
            continue

        std_result = standardiser.process_smiles(str(smiles))
        if std_result is not None:
            fig_num = "Figure000"
            if isinstance(cmpd_id, str) and "-" in cmpd_id:
                fig_num = cmpd_id.split("-")[0]

            std_smiles = std_result["std_smiles"]
            category = classify_vdr_category(std_smiles, fig_num)

            processed_records.append({
                "compound_id": cmpd_id,
                "figure_num": fig_num,
                "category": category,
                "raw_smiles": smiles,
                "std_smiles": std_smiles,
                "inchikey": std_result["inchikey"],
                "inchi": std_result["inchi"]
            })

    df_processed = pd.DataFrame(processed_records)
    df_curated = df_processed.drop_duplicates(subset=["inchikey"], keep="first").copy()

    # Export CSV with 'category' column
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df_curated.to_csv(output_csv_path, index=False)
    print(f"Saved standardized CSV with categories: {output_csv_path}")

    # Export SDF with Category property
    sdf_writer = Chem.SDWriter(output_sdf_path)
    for _, row in df_curated.iterrows():
        mol = Chem.MolFromSmiles(row["std_smiles"])
        if mol is not None:
            AllChem.Compute2DCoords(mol)
            mol.SetProp("_Name", str(row["compound_id"]))
            mol.SetProp("FigureNum", str(row["figure_num"]))
            mol.SetProp("Category", str(row["category"]))
            mol.SetProp("InChIKey", str(row["inchikey"]))
            mol.SetProp("InChI", str(row["inchi"]))
            sdf_writer.write(mol)
    sdf_writer.close()
    print(f"Saved standardized SDF with properties: {output_sdf_path}")

    return df_curated


if __name__ == "__main__":
    run_curation_pipeline()
