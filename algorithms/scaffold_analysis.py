"""
algorithms/scaffold_analysis.py
Bemis-Murcko scaffold extraction and generic framework analysis engine.

Features:
1. Extracts exact atomic Bemis-Murcko ring-and-linker scaffolds.
2. Converts atomic scaffolds to generic carbon frameworks (all heteroatoms -> C, all bonds -> single).
3. Computes scaffold frequencies, singleton ratios, and core diversity metrics.
4. Integrates BoronSurrogateTransformer for carborane/boron analogues.
"""

from typing import Dict, Tuple, List, Any, Optional
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from algorithms.boron_surrogate import BoronSurrogateTransformer


class ScaffoldAnalyzer:
    """
    Extracts Bemis-Murcko scaffolds and generic frameworks, and computes
    scaffold diversity statistics across the dataset.
    """

    def __init__(self):
        self.boron_transformer = BoronSurrogateTransformer()

    def extract_scaffolds_for_mol(self, mol: Optional[Chem.Mol]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Extracts (bm_scaffold_smiles, bm_scaffold_inchikey, generic_framework_smiles)
        for a given RDKit molecule.
        """
        if mol is None:
            return None, None, None

        # Handle boron / carborane analogues via carbon surrogate
        eval_mol, _ = self.boron_transformer.create_surrogate(mol)

        try:
            # 1. Atomic Bemis-Murcko Scaffold
            scaffold_mol = MurckoScaffold.GetScaffoldForMol(eval_mol)
            scaffold_smiles = Chem.MolToSmiles(scaffold_mol, canonical=True)
            scaffold_inchikey = Chem.MolToInchiKey(scaffold_mol) if scaffold_smiles else ""

            # 2. Generic Framework (heteroatoms -> Carbon, bonds -> single)
            generic_mol = MurckoScaffold.MakeScaffoldGeneric(scaffold_mol)
            generic_smiles = Chem.MolToSmiles(generic_mol, canonical=True)

            return scaffold_smiles, scaffold_inchikey, generic_smiles
        except Exception:
            return None, None, None

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        smiles_col: str = "std_smiles",
        id_col: str = "compound_id"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Analyzes an entire DataFrame of SMILES strings.

        Returns:
            cmpd_scaffolds_df: Compound-level scaffold assignment table.
            bm_counts_df: Bemis-Murcko scaffold frequency table.
            gen_counts_df: Generic framework frequency table.
            summary_stats: High-level scaffold diversity summary statistics.
        """
        records = []
        for _, row in df.iterrows():
            cmpd_id = str(row.get(id_col, "UNKNOWN"))
            smiles = str(row.get(smiles_col, ""))

            mol = Chem.MolFromSmiles(smiles) if smiles and isinstance(smiles, str) else None
            bm_smi, bm_key, gen_smi = self.extract_scaffolds_for_mol(mol)

            records.append({
                id_col: cmpd_id,
                "bm_scaffold_smiles": bm_smi,
                "bm_scaffold_inchikey": bm_key,
                "generic_framework_smiles": gen_smi,
                "has_scaffold": bool(bm_smi and len(bm_smi) > 0)
            })

        cmpd_scaffolds_df = pd.DataFrame(records)

        # 1. Bemis-Murcko Scaffold Frequency Table
        bm_counts_df = cmpd_scaffolds_df["bm_scaffold_smiles"].value_counts().reset_index()
        bm_counts_df.columns = ["bm_scaffold_smiles", "count"]
        bm_counts_df["percentage"] = (bm_counts_df["count"] / len(cmpd_scaffolds_df)) * 100

        # 2. Generic Framework Frequency Table
        gen_counts_df = cmpd_scaffolds_df["generic_framework_smiles"].value_counts().reset_index()
        gen_counts_df.columns = ["generic_framework_smiles", "count"]
        gen_counts_df["percentage"] = (gen_counts_df["count"] / len(cmpd_scaffolds_df)) * 100

        # 3. High-level Scaffold Diversity Summary
        total_compounds = len(cmpd_scaffolds_df)
        num_bm_scaffolds = cmpd_scaffolds_df["bm_scaffold_smiles"].nunique()
        num_gen_frameworks = cmpd_scaffolds_df["generic_framework_smiles"].nunique()
        bm_singletons = int((bm_counts_df["count"] == 1).sum())

        summary_stats = {
            "total_compounds": total_compounds,
            "unique_bm_scaffolds": num_bm_scaffolds,
            "unique_generic_frameworks": num_gen_frameworks,
            "bm_scaffold_diversity_ratio": round(num_bm_scaffolds / total_compounds, 4) if total_compounds > 0 else 0,
            "generic_framework_diversity_ratio": round(num_gen_frameworks / total_compounds, 4) if total_compounds > 0 else 0,
            "bm_singleton_scaffolds": bm_singletons,
            "bm_singleton_percentage": round((bm_singletons / num_bm_scaffolds) * 100, 2) if num_bm_scaffolds > 0 else 0
        }

        return cmpd_scaffolds_df, bm_counts_df, gen_counts_df, summary_stats
