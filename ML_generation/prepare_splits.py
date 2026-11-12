"""
ML_generation/prepare_splits.py
Creates a reproducible 80/20 scaffold-aware train/test split for VDR ligands
so generative models can be evaluated on unseen held-out Regeneration Rate
and Rediscovery without scaffold leakage.
"""

import os
import random
import pandas as pd
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def get_canonical_smiles(smi: str, remove_stereo: bool = False) -> str | None:
    if not isinstance(smi, str) or not smi.strip():
        return None
    mol = Chem.MolFromSmiles(smi.strip())
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=not remove_stereo)


def create_scaffold_split(
    input_path: str = "data/processed/vdr_curated_library.csv",
    fallback_smi: str = "data/processed/vdr_binders.smi",
    train_out: str = "data/processed/vdr_train.smi",
    test_out: str = "data/processed/vdr_test_holdout.smi",
    test_ratio: float = 0.20,
    seed: int = 42,
):
    random.seed(seed)

    # Prefer curated CSV (preserves stereochemistry in std_smiles), fall back to .smi
    source_path = input_path if os.path.exists(input_path) else fallback_smi
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Neither '{input_path}' nor '{fallback_smi}' exists. Run Step 1 first.")

    if source_path.endswith(".csv"):
        df = pd.read_csv(source_path)
        smi_col = next((c for c in df.columns if "smiles" in c.lower()), df.columns[0])
        raw_smiles = df[smi_col].dropna().tolist()
    else:
        with open(source_path, "r", encoding="utf-8") as f:
            raw_smiles = [line.strip().split()[0] for line in f if line.strip()]

    # Canonicalize (retaining stereochemistry for Transfer Learning) and deduplicate by 2D graph
    seen_2d = {}
    for raw in raw_smiles:
        smi_2d = get_canonical_smiles(raw, remove_stereo=True)
        smi_iso = get_canonical_smiles(raw, remove_stereo=False)
        if smi_2d is not None and smi_2d not in seen_2d:
            seen_2d[smi_2d] = smi_iso

    # Group by Bemis-Murcko scaffold
    scaffold_groups = defaultdict(list)
    for smi_2d, smi_iso in seen_2d.items():
        mol = Chem.MolFromSmiles(smi_2d)
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        scaffold_groups[scaf].append(smi_iso)

    # Shuffle deterministically, then sort ascending so oversized scaffolds (e.g., core secosteroids) go to train
    buckets = list(scaffold_groups.values())
    random.shuffle(buckets)
    buckets.sort(key=len)

    total_unique = len(seen_2d)
    target_test_size = int(total_unique * test_ratio)
    train_smiles, test_smiles = [], []

    for bucket in buckets:
        if len(test_smiles) < target_test_size and len(bucket) <= target_test_size:
            test_smiles.extend(bucket)
        else:
            train_smiles.extend(bucket)

    os.makedirs(os.path.dirname(train_out), exist_ok=True)
    with open(train_out, "w", encoding="utf-8") as f:
        f.write("\n".join(train_smiles) + "\n")
    with open(test_out, "w", encoding="utf-8") as f:
        f.write("\n".join(test_smiles) + "\n")

    # Audit aromatic vs. aliphatic balance across both splits
    def _arom_pct(smi_list):
        arom = sum(
            1 for s in smi_list
            if (m := Chem.MolFromSmiles(s)) is not None and any(b.GetIsAromatic() for b in m.GetBonds())
        )
        return (arom / len(smi_list) * 100) if smi_list else 0.0

    print("\n" + "=" * 60)
    print("  SCAFFOLD-AWARE TRAIN / TEST SPLIT SUMMARY")
    print("=" * 60)
    print(f"  Source File          : {source_path}")
    print(f"  Total Unique Ligands : {total_unique} ({len(scaffold_groups)} Murcko scaffolds)")
    print(f"  Train Set ({len(train_smiles)/total_unique*100:.1f}%)    : {len(train_smiles)} ligands -> {train_out}")
    print(f"    - Aromatic ligands : {_arom_pct(train_smiles):.1f}%")
    print(f"  Test Holdout ({len(test_smiles)/total_unique*100:.1f}%) : {len(test_smiles)} ligands -> {test_out}")
    print(f"    - Aromatic ligands : {_arom_pct(test_smiles):.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    create_scaffold_split()
