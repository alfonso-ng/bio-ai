import deepchem as dc
import joblib
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors



# 1. CARGAR EL MODELO GANADOR
print("Cargando modelo...")
model = joblib.load("../models/mejor_modelo_lrrk2_scaffold.pkl")

# 2. CARGAR DATASET ZINC15 (Versión actualizada)
# Nota: 'subset' puede ser '10K', '100K', etc. Vamos a por el 100K para que sea serio.
print("Cargando moléculas de ZINC15 via DeepChem...")
tasks, datasets, transformers = dc.molnet.load_zinc15(featurizer='Raw')
full_dataset = datasets[0] # Tomamos el primer bloque disponible

# 3. FEATURIZER (Morgan Fingerprints)
featurizer = dc.feat.CircularFingerprint(size=2048, radius=2)

# 4. BUCLE DE SCREENING OPTIMIZADO
print("Iniciando screening sobre 100,000 moléculas...")
hits = []
smiles_list = full_dataset.X


for i, sml in enumerate(smiles_list):
    mol = sml
    if mol:
        mw = Descriptors.MolWt(mol)
        tpsa = Descriptors.TPSA(mol)
        
        # Filtros BBB estrictos
        if mw < 450 and tpsa < 90:
            # Featurización
            fp = featurizer.featurize([mol])
            
            # Predicción
            score = model.predict(fp)[0]
            
            if score > 7.0: # Umbral de potencia
                hits.append({
                    'SMILES': Chem.MolToSmiles(mol, canonical=True),
                    'Score': round(float(score), 3),
                    'MW': round(mw, 2),
                    'TPSA': round(tpsa, 2)
                })
    
    if i % 10000 == 0:
        print(f"Progreso: {i} moléculas analizadas...")

# 5. RESULTADOS
df_hits = pd.DataFrame(hits).sort_values(by='Score', ascending=False)
df_hits.to_csv("hits_lrrk2_zinc15.csv", index=False)

print(f"\n¡Listo! Hemos encontrado {len(df_hits)} candidatos potenciales.")
print(df_hits.head(10))