"""
algorithms/standardisation.py
Core chemical standardization module for VDR ligands.
Executes RDKit sanitisation, fragment parent extraction (salt/solvent removal),
charge neutralisation, aromaticity perception, canonical tautomer selection,
and InChIKey/InChI generation.
"""

from typing import Dict, Optional
import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


class ChemicalStandardiser:
    def __init__(self):
        self.uncharger = rdMolStandardize.Uncharger()
        self.tautomer_enumerator = rdMolStandardize.TautomerEnumerator()

    def process_smiles(self, smiles: str) -> Optional[Dict[str, str]]:
        """
        Processes a raw SMILES string through the standardization pipeline.
        Returns a dictionary with 'std_smiles', 'inchikey', and 'inchi',
        or None if parsing fails.
        """
        if not smiles or not isinstance(smiles, str) or pd.isna(smiles):
            return None

        smiles_clean = smiles.strip()
        if not smiles_clean:
            return None

        # 1. Parse raw SMILES
        mol = Chem.MolFromSmiles(smiles_clean)
        if mol is None:
            return None

        try:
            # 2. Basic Sanitisation
            Chem.SanitizeMol(mol)

            # 3. Salt and solvent fragment removal (retains largest fragment parent)
            mol = rdMolStandardize.FragmentParent(mol)

            # 4. Neutralisation (uncharge un-ionized forms where possible)
            mol = self.uncharger.uncharge(mol)

            # 5. Consistent Aromaticity Perception
            Chem.SanitizeMol(mol)

            # 6. Canonical Tautomer Perception
            mol = self.tautomer_enumerator.Canonicalize(mol)

            # 7. Generate canonical outputs
            std_smiles = Chem.MolToSmiles(mol, canonical=True)
            inchikey = Chem.MolToInchiKey(mol)
            inchi = Chem.MolToInchi(mol)

            return {
                "std_smiles": std_smiles,
                "inchikey": inchikey,
                "inchi": inchi
            }
        except Exception:
            return None
