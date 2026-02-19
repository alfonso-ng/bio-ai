from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator
import numpy as np

def featurize_smiles(df, radius=2, n_bits=2048):
    # Aplicar a tu DataFrame
    df['fingerprint'] = df['canonical_smiles'].apply(smiles_to_fp, radius=radius, n_bits=n_bits)
    
    # Eliminar posibles fallos
    df = df.dropna(subset=['fingerprint'])
    return np.stack(df["fingerprint"].values)
    

def smiles_to_fp(smiles, radius, n_bits):
    # 1. Convertir SMILES a objeto Mol de RDKit
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None # Por si hay algún SMILES corrupto
    
    # 2. Generar el Fingerprint de Morgan
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp = mfpgen.GetFingerprint(mol)
    
    # 3. Convertir a un array de NumPy para que la IA lo entienda
    arr = np.zeros((1,), dtype=int)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr