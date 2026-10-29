"""
algorithms/boron_surrogate.py
Identifies Boron (B) atoms in carborane-containing analogues and constructs
a carbon-surrogate molecule (B -> C) so topological descriptors can be calculated
without RDKit errors.
"""

from typing import Tuple, Optional
from rdkit import Chem


class BoronSurrogateTransformer:
    """
    Handles detection and carbon-surrogate transformation for boron-containing ligands.
    """

    @staticmethod
    def has_boron(mol: Optional[Chem.Mol]) -> bool:
        """
        Checks if a molecule contains any Boron atoms (atomic number 5).
        """
        if mol is None:
            return False
        return any(atom.GetAtomicNum() == 5 for atom in mol.GetAtoms())

    @classmethod
    def create_surrogate(cls, mol: Optional[Chem.Mol]) -> Tuple[Optional[Chem.Mol], bool]:
        """
        Replaces Boron (B) atoms with Carbon (C) surrogates while preserving
        molecular topology and bond connectivity.
        Returns a tuple: (surrogate_mol, is_boron_flag).
        """
        if mol is None:
            return None, False

        is_boron = cls.has_boron(mol)
        if not is_boron:
            return mol, False

        rw_mol = Chem.RWMol(mol)
        for atom in rw_mol.GetAtoms():
            if atom.GetAtomicNum() == 5:
                atom.SetAtomicNum(6)  # Replace Boron with Carbon surrogate
                atom.UpdatePropertyCache(strict=False)  # Refresh RDKit valence cache

        surrogate_mol = rw_mol.GetMol()
        try:
            Chem.SanitizeMol(surrogate_mol)
        except Exception:
            pass

        return surrogate_mol, True

    @classmethod
    def get_native_and_surrogate(cls, mol: Optional[Chem.Mol]) -> Tuple[Optional[Chem.Mol], Optional[Chem.Mol], bool]:
        """
        Returns (native_mol, surrogate_mol, is_boron_flag) to allow dual-value
        tracking for native vs surrogate structure profiling.
        """
        if mol is None:
            return None, None, False

        surrogate_mol, is_boron = cls.create_surrogate(mol)
        return mol, surrogate_mol, is_boron
