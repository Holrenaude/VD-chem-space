import pandas as pd
import os
from rdkit import Chem

# Define file paths
input_path = os.path.join("data", "processed", "vdr_curated_library.csv")
output_path = os.path.join("data", "processed", "vdr_binders.smi")

# Load the dataset
df = pd.read_csv(input_path)

clean_smiles = set()

# Process the standardized SMILES column
for smi in df['std_smiles'].dropna():
    mol = Chem.MolFromSmiles(smi)
    if mol:
        # Canonicalize and remove stereochemistry for standard prior training
        clean_smiles.add(Chem.MolToSmiles(mol, isomericSmiles=False))

# Ensure the output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Save the cleaned SMILES to a .smi file
with open(output_path, 'w') as f:
    for smi in clean_smiles:
        f.write(f"{smi}\n")

print(f"Saved {len(clean_smiles)} unique, valid SMILES to {output_path}")
