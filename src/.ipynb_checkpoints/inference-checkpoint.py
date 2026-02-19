import joblib
import pandas as pd
import numpy as np
from features import featurize_smiles # Reutilizamos vuestra "fábrica"
import seaborn as sns
import matplotlib.pyplot as plt

def run_virtual_screening(model_path, candidates_csv):
    # 1. Cargar el modelo guardado
    model = joblib.load(model_path)
    # 2. Cargar candidatos (ej: fármacos de la FDA)
    # Supongamos que el CSV tiene columnas ['name', 'smiles']
    df_candidates = pd.read_csv(candidates_csv)
    print(df_candidates)
    
    print(f"🔬 Analizando {len(df_candidates)} moléculas...")
    
    # 3. Vectorización (IMPORTANTE: usar los mismos bits y radio)
    X_candidates = featurize_smiles(df_candidates)
    
    # 4. Predicción de pChEMBL
    # Vuestro modelo devuelve el logaritmo de la actividad
    predictions = model.predict(X_candidates)
    
    # 5. Organizar resultados
    df_candidates['predicted_pChEMBL'] = predictions
    
    # Ordenar de mayor a menor actividad
    df_results = df_candidates.sort_values(by='predicted_pChEMBL', ascending=False)
    
    return df_results


def plot_screening_results(results):
    plt.figure(figsize=(10, 6))
    sns.histplot(results['predicted_pChEMBL'], kde=True, color='teal')
    plt.axvline(x=7, color='red', linestyle='--', label='Umbral de Alta Potencia')
    plt.title('Distribución de Predicciones en la Librería de Candidatos')
    plt.xlabel('Valor pChEMBL Predicho')
    plt.ylabel('Número de Moléculas')
    plt.legend()
    plt.show()

# Ejecución rápida
if __name__ == "__main__":
    results = run_virtual_screening('models/best_CHEMBL4822_model.pkl', 'data/farmacos.csv')
    print("✨ Top 5 candidatos encontrados:")
    print(results.head(5))
    plot_screening_results(results)