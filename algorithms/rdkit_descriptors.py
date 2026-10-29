"""
algorithms/rdkit_descriptors.py
Engine for calculating RDKit 2D descriptors (200+) with Boron Dual-Reporting.

Calculates descriptors for:
1. Native molecular structures (reported under standard descriptor names).
2. Carbon-surrogate structures (reported under '{desc_name}_surrogate').
Flagged by 'is_boron_analogue' boolean column.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from algorithms.boron_surrogate import BoronSurrogateTransformer


class RDKitDescriptorEngine:
    """
    Engine for dynamically computing 2D RDKit descriptors with native vs surrogate dual-reporting.
    """
    def __init__(self):
        self.desc_list = Descriptors._descList
        self.boron_transformer = BoronSurrogateTransformer()

    def calculate_descriptors(self, smiles: str) -> Optional[Dict[str, Any]]:
        """
        Calculates RDKit 2D descriptors on native and surrogate structures separately.
        Returns a dictionary containing native and surrogate descriptor values.
        """
        if not smiles or not isinstance(smiles, str) or pd.isna(smiles):
            return None

        smiles_clean = smiles.strip()
        if not smiles_clean:
            return None

        mol = Chem.MolFromSmiles(smiles_clean)
        if mol is None:
            return None

        # 1. Obtain both native and carbon-surrogate molecules
        native_mol, surrogate_mol, is_boron = self.boron_transformer.get_native_and_surrogate(mol)

        res: Dict[str, Any] = {
            "is_boron_analogue": is_boron
        }

        # 2. Dual-reporting descriptor calculation
        for desc_name, desc_func in self.desc_list:
            # Native structure calculation
            try:
                val_native = desc_func(native_mol)
                res[desc_name] = float(val_native) if val_native is not None and np.isfinite(val_native) else np.nan
            except Exception:
                res[desc_name] = np.nan

            # Carbon-surrogate structure calculation
            try:
                val_surr = desc_func(surrogate_mol)
                res[f"{desc_name}_surrogate"] = float(val_surr) if val_surr is not None and np.isfinite(val_surr) else np.nan
            except Exception:
                res[f"{desc_name}_surrogate"] = np.nan

        return res

    def profile_dataframe(
        self,
        df: pd.DataFrame,
        smiles_col: str = "std_smiles",
        id_col: str = "compound_id"
    ) -> pd.DataFrame:
        """
        Calculates dual native/surrogate descriptors across a pandas DataFrame.
        """
        records = []
        for _, row in df.iterrows():
            cmpd_id = row.get(id_col, "UNKNOWN")
            smiles = row.get(smiles_col, "")

            res = self.calculate_descriptors(smiles)
            if res is not None:
                res[id_col] = cmpd_id
                records.append(res)

        out_df = pd.DataFrame(records)
        if id_col in out_df.columns:
            cols = [id_col] + [c for c in out_df.columns if c != id_col]
            out_df = out_df[cols]

        return out_df
