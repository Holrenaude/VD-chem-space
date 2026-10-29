"""
algorithms/fingerprint_generator.py
Multi-family molecular fingerprint panel generation engine for VDR ligands.

Representation Families Generated (12 Fingerprints):
1. Path & Topological: RDKit Topological (2048-bit), Topological Torsion (2048-bit),
   Daylight-type Pattern (2048-bit), Layered Subgraph (2048-bit)
2. Atom Pair: Atom Pair (2048-bit)
3. Circular: ECFP4 (Morgan r=2, 2048-bit), ECFP6 (Morgan r=3, 2048-bit), FCFP4 (Feature-based r=2, 2048-bit)
4. Substructure Keys: MACCS Keys (167-bit), Avalon (512-bit), EState Substructure Keys (79-bit)
5. Pharmacophore: Gobbi 2D Pharmacophore (39972-bit)
"""

from typing import Dict, Optional, Any
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, MACCSkeys
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D, Generate
from rdkit.Chem.EState import AtomTypes
from rdkit.DataStructs import ExplicitBitVect

from algorithms.boron_surrogate import BoronSurrogateTransformer


class FingerprintGenerator:
    """
    Generates an expanded panel of 12 binary bitvector fingerprints across major
    representation families for molecular structures.
    """

    def __init__(self, fp_size: int = 2048):
        self.fp_size = fp_size
        self.boron_transformer = BoronSurrogateTransformer()

        # Initialize RDKit modern fingerprint generators
        self.rdkit_gen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=fp_size)
        self.tt_gen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=fp_size)
        self.ap_gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=fp_size)
        self.ecfp4_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fp_size)
        self.ecfp6_gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=fp_size)
        self.fcfp4_gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=2,
            fpSize=fp_size,
            atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
        )
        self.pharm_factory = Gobbi_Pharm2D.factory

        # Initialize EState SMARTS patterns if not built
        if not AtomTypes.esPatterns:
            AtomTypes.BuildPatts()

    def get_estate_keys(self, mol: Chem.Mol) -> ExplicitBitVect:
        """
        Generates 79-bit EState substructure presence keys.
        """
        ebv = ExplicitBitVect(len(AtomTypes.esPatterns))
        for i, (_, pat) in enumerate(AtomTypes.esPatterns):
            if mol.HasSubstructMatch(pat):
                ebv.SetBit(i)
        return ebv

    def generate_fingerprint_panel(self, smiles: str) -> Optional[Dict[str, Any]]:
        """
        Calculates all 12 fingerprint representations for a single SMILES string.
        Returns a dictionary mapping fingerprint names to RDKit BitVect objects,
        or None if molecule parsing fails.
        """
        if not smiles or not isinstance(smiles, str) or pd.isna(smiles):
            return None

        smiles_clean = smiles.strip()
        if not smiles_clean:
            return None

        mol = Chem.MolFromSmiles(smiles_clean)
        if mol is None:
            return None

        # Obtain carbon surrogate for carborane/boron analogues if present
        eval_mol, is_boron = self.boron_transformer.create_surrogate(mol)

        try:
            return {
                # 1. Path & Topological
                "rdkit_topo": self.rdkit_gen.GetFingerprint(eval_mol),
                "topological_torsion": self.tt_gen.GetFingerprint(eval_mol),
                "daylight_pattern": Chem.PatternFingerprint(eval_mol, fpSize=self.fp_size),
                "layered_path": Chem.LayeredFingerprint(eval_mol, fpSize=self.fp_size),
                
                # 2. Atom Pair
                "atom_pair": self.ap_gen.GetFingerprint(eval_mol),
                
                # 3. Circular
                "ecfp4": self.ecfp4_gen.GetFingerprint(eval_mol),
                "ecfp6": self.ecfp6_gen.GetFingerprint(eval_mol),
                "fcfp4": self.fcfp4_gen.GetFingerprint(eval_mol),
                
                # 4. Substructure Keys
                "maccs": MACCSkeys.GenMACCSKeys(eval_mol),
                "avalon": pyAvalonTools.GetAvalonFP(eval_mol, nBits=512),
                "estate_keys": self.get_estate_keys(eval_mol),
                
                # 5. Pharmacophore
                "pharm2d": Generate.Gen2DFingerprint(eval_mol, self.pharm_factory)
            }
        except Exception:
            return None

    def profile_dataframe(
        self,
        df: pd.DataFrame,
        smiles_col: str = "std_smiles",
        id_col: str = "compound_id"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generates fingerprint panels for all molecules in a pandas DataFrame.
        Returns a nested dictionary structure:
        { compound_id: { "ecfp4": BitVect, "maccs": BitVect, ... } }
        """
        dataset_fps: Dict[str, Dict[str, Any]] = {}

        for _, row in df.iterrows():
            cmpd_id = str(row.get(id_col, "UNKNOWN"))
            smiles = str(row.get(smiles_col, ""))

            fp_panel = self.generate_fingerprint_panel(smiles)
            if fp_panel is not None:
                dataset_fps[cmpd_id] = fp_panel

        return dataset_fps
