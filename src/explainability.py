import joblib
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import SimilarityMaps, rdMolDraw2D
import matplotlib.pyplot as plt
import numpy as np

def explain_prediction(model, smiles, name):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return
    
    # Envoltorio para que el modelo prediga
    def get_probas(fp):
        fp_array = np.array([fp])
        return float(model.predict(fp_array)[0])

    # 1. Creamos el objeto de dibujo (Canvas)
    # 400x400 px suele ser ideal
    drawer = rdMolDraw2D.MolDraw2DCairo(400, 400) 
    
    # 2. Generamos los pesos (contribución de cada átomo)
    # Ahora pasamos el drawer como argumento
    weights = SimilarityMaps.GetAtomicWeightsForModel(mol, SimilarityMaps.GetMorganFingerprint, get_probas)
    
    # 3. Dibujamos el mapa de similitud en el drawer
    SimilarityMaps.GetSimilarityMapFromWeights(mol, weights, draw2d=drawer)
    drawer.FinishDrawing()
    
    # 4. Mostrar la imagen usando Matplotlib
    import io
    from PIL import Image
    bio = io.BytesIO(drawer.GetDrawingText())
    img = Image.open(bio)
    
    plt.figure(figsize=(8,8))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Mapa de Contribución: {name}\n(Verde = Mayor actividad, Rosa/Rojo = Menor)")
    plt.show()


import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem.Draw import SimilarityMaps

def get_total_contribution(model, smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return 0
    
    # Función envoltorio para la predicción
    def get_probas(fp):
        return float(model.predict(np.array([fp]))[0])

    # Obtenemos los pesos atómicos brutos
    weights = SimilarityMaps.GetAtomicWeightsForModel(
        mol, 
        SimilarityMaps.GetMorganFingerprint, 
        get_probas
    )
    
    # La suma de pesos nos da la contribución total
    return np.sum(weights)

# Ejecución para el Verubecestat
model = joblib.load("models/best_CHEMBL4822_model.pkl")
explain_prediction(model, "CN1C(N)=N[C@](C)(c2cc(NC(=O)c3ccc(F)cn3)ccc2F)CS1(=O)=O", "Verubecestat")

# Datos para la comparativa
moleculas = {
    "Verubecestat": "CN1C(N)=N[C@](C)(c2cc(NC(=O)c3ccc(F)cn3)ccc2F)CS1(=O)=O",
    "Atabecestat": "C[C@@]1(c2cc(NC(=O)c3ccc(C#N)cn3)ccc2F)C=CSC(N)=N1",
    "Risperidona": "Cc1nc2n(c(=O)c1CCN1CCC(c3noc4cc(F)ccc34)CC1)CCCC2",
    "Caffeine": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "Ibuprofeno": "CC(C)Cc1ccc(C(C)C(=O)O)cc1"
}

# Cálculo
nombres = list(moleculas.keys())
contribuciones = [get_total_contribution(model, s) for s in moleculas.values()]

# 2. Visualización
plt.figure(figsize=(10, 6))
colors = ['#2ecc71' if c > 5 else '#f1c40f' if c > 2 else '#e74c3c' for c in contribuciones]

plt.bar(nombres, contribuciones, color=colors)
plt.ylabel('Suma Total de Pesos de Importancia')
plt.title('Contribución Estructural Total al pChEMBL Predicho')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Añadimos los valores encima de las barras
for i, v in enumerate(contribuciones):
    plt.text(i, v + 0.1, f"{v:.2f}", ha='center', fontweight='bold')

plt.show()