"""
ML_generation/Mol2Mol/prepare_pairs.py
Builds vocabulary-verified matched molecular pairs (source -> target) from vdr_train.smi
for Mol2Mol Transfer Learning (90/10 train/val pair split), and prepares seed SMILES
from vdr_test_holdout.smi.
"""

from pathlib import Path
import re
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

TRAIN_SMI = Path("data/processed/vdr_train.smi")
TEST_SMI = Path("data/processed/vdr_test_holdout.smi")
OUT_DIR = Path("ML_generation/Mol2Mol/tl_v1")
TRAIN_PAIRS_FILE = OUT_DIR / "vdr_train_pairs.smi"
VAL_PAIRS_FILE = OUT_DIR / "vdr_val_pairs.smi"
SEEDS_FILE = OUT_DIR / "vdr_test_seeds.smi"

ALLOWED_ELEMENTS = {"C", "N", "O", "S", "F", "Cl", "Br", "I", "H"}
ALLOWED_BRACKETS = {
    "[C@H]", "[C@@H]", "[C@]", "[C@@]",
    "[nH]", "[n+]", "[N+]", "[N@+]", "[N@@+]", "[N@]",
    "[O-]", "[O]", "[S@]", "[S@@]",
}


def is_vocab_compatible(mol: Chem.Mol, smiles: str) -> bool:
    for atom in mol.GetAtoms():
        if atom.GetIsotope() != 0 or atom.GetSymbol() not in ALLOWED_ELEMENTS:
            return False
    brackets = re.findall(r"\[[^\]]+\]", smiles)
    return all(b in ALLOWED_BRACKETS for b in brackets)


def load_valid_smiles(path: Path) -> list[str]:
    smiles = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip().split()[0] if line.strip() else ""
            if not s or s.lower() == "smiles":
                continue
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                continue
            can_smi = Chem.MolToSmiles(mol, isomericSmiles=True)
            if not is_vocab_compatible(mol, can_smi):
                continue
            smiles.append(can_smi)
    return list(dict.fromkeys(smiles))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_smiles = load_valid_smiles(TRAIN_SMI)
    test_smiles = load_valid_smiles(TEST_SMI)

    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [morgan_gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in train_smiles]

    pairs = []
    for i, fp_i in enumerate(fps):
        sims = np.array(DataStructs.BulkTanimotoSimilarity(fp_i, fps))
        valid_idx = np.where((sims >= 0.45) & (sims < 0.99))[0]
        if len(valid_idx) == 0:
            continue
        top_k = valid_idx[np.argsort(-sims[valid_idx])[:5]]
        for j in top_k:
            pairs.append(f"{train_smiles[i]}\t{train_smiles[j]}")

    # Deterministic 90/10 split of training pairs for TL validation monitoring
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(pairs))
    n_val = max(1, int(0.1 * len(pairs)))
    val_pairs = [pairs[idx] for idx in perm[:n_val]]
    train_pairs = [pairs[idx] for idx in perm[n_val:]]

    with open(TRAIN_PAIRS_FILE, "w") as f:
        for p in train_pairs:
            f.write(p + "\n")

    with open(VAL_PAIRS_FILE, "w") as f:
        for p in val_pairs:
            f.write(p + "\n")

    with open(SEEDS_FILE, "w") as f:
        for s in test_smiles:
            f.write(s + "\n")

    print(f"Saved {len(train_pairs)} train pairs and {len(val_pairs)} val pairs")
    print(f"Saved {len(test_smiles)} held-out seed molecules to {SEEDS_FILE}")


if __name__ == "__main__":
    main()
