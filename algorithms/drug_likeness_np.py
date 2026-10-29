"""
algorithms/drug_likeness_np.py
Evaluates drug-likeness metrics and official Ertl & Roggo (2008) Natural Product-likeness scores
with dual native vs. surrogate reporting.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, rdMolDescriptors
from rdkit.Contrib.NP_Score import npscorer

from algorithms.boron_surrogate import BoronSurrogateTransformer


class NPScoreCalculator:
    """
    Calculates official Natural Product-likeness score per Ertl & Roggo (2008)
    using RDKit Contrib's fragment-based score model.
    """
    _fscore = None

    @classmethod
    def get_fscore_model(cls):
        if cls._fscore is None:
            cls._fscore = npscorer.readNPModel()
        return cls._fscore

    @classmethod
    def calculate_score(cls, mol: Optional[Chem.Mol]) -> float:
        if mol is None:
            return np.nan

        try:
            fscore = cls.get_fscore_model()
            score = npscorer.scoreMol(mol, fscore)
            return round(float(score), 4)
        except Exception:
            try:
                num_heavy = mol.GetNumHeavyAtoms()
                if num_heavy == 0:
                    return 0.0

                f_sp3 = rdMolDescriptors.CalcFractionCSP3(mol)
                num_chiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
                chiral_density = num_chiral / num_heavy

                ring_info = mol.GetRingInfo()
                num_rings = ring_info.NumRings()
                ring_density = num_rings / num_heavy

                proxy_score = (f_sp3 * 2.5) + (chiral_density * 3.0) + (ring_density * 1.5)
                return round(float(proxy_score), 4)
            except Exception:
                return np.nan


class DrugLikenessProfiler:
    """
    Evaluates Lipinski, Veber, QED, and official Ertl NP-score metrics for native and surrogate forms.
    """

    def __init__(self):
        self.boron_transformer = BoronSurrogateTransformer()
        self.np_calculator = NPScoreCalculator()

    def _evaluate_mol_properties(self, mol: Chem.Mol) -> Dict[str, Any]:
        if mol is None:
            return {}

        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        tpsa = Descriptors.TPSA(mol)
        rotb = Descriptors.NumRotatableBonds(mol)

        mw_pass = mw <= 500.0
        logp_pass = logp <= 5.0
        hbd_pass = hbd <= 5
        hba_pass = hba <= 10
        violations = sum([not mw_pass, not logp_pass, not hbd_pass, not hba_pass])
        veber_pass = (tpsa <= 140.0) and (rotb <= 10)

        try:
            qed_score = float(QED.qed(mol))
        except Exception:
            qed_score = np.nan

        np_score = self.np_calculator.calculate_score(mol)

        return {
            "MolWt": mw,
            "MolLogP": logp,
            "NumHDonors": hbd,
            "NumHAcceptors": hba,
            "TPSA": tpsa,
            "NumRotatableBonds": rotb,
            "Lipinski_MW_pass": mw_pass,
            "Lipinski_LogP_pass": logp_pass,
            "Lipinski_HBD_pass": hbd_pass,
            "Lipinski_HBA_pass": hba_pass,
            "Lipinski_Violations": violations,
            "Veber_Pass": veber_pass,
            "QED": qed_score,
            "NP_Score": np_score
        }

    def evaluate_molecule(self, smiles: str) -> Optional[Dict[str, Any]]:
        if not smiles or not isinstance(smiles, str) or pd.isna(smiles):
            return None

        smiles_clean = smiles.strip()
        if not smiles_clean:
            return None

        mol = Chem.MolFromSmiles(smiles_clean)
        if mol is None:
            return None

        native_mol, surrogate_mol, is_boron = self.boron_transformer.get_native_and_surrogate(mol)

        native_props = self._evaluate_mol_properties(native_mol)
        surrogate_props = self._evaluate_mol_properties(surrogate_mol)

        res: Dict[str, Any] = {"is_boron_analogue": is_boron}

        for k, v in native_props.items():
            res[k] = v

        for k, v in surrogate_props.items():
            res[f"{k}_surrogate"] = v

        return res

    def profile_dataframe(
        self,
        df: pd.DataFrame,
        smiles_col: str = "std_smiles",
        id_col: str = "compound_id"
    ) -> pd.DataFrame:
        records = []
        for _, row in df.iterrows():
            cmpd_id = row.get(id_col, "UNKNOWN")
            smiles = row.get(smiles_col, "")

            res = self.evaluate_molecule(smiles)
            if res is not None:
                res[id_col] = cmpd_id
                records.append(res)

        out_df = pd.DataFrame(records)
        if id_col in out_df.columns:
            cols = [id_col] + [c for c in out_df.columns if c != id_col]
            out_df = out_df[cols]

        return out_df
