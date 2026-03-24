import deepchem as dc
import pandas as pd
import optuna
import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import joblib

def run_pipeline():
    def objective(trial):
        # Definimos el espacio de búsqueda
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.8),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
            'n_jobs': -1,
            'random_state': 42
        }
    
        # Entrenar modelo
        model = xgb.XGBRegressor(**param)
        model.fit(X_train, y_train)
    
        # Predecir en el set de VALIDACIÓN (scaffolds nunca vistos)
        preds = model.predict(X_valid)
        r2 = r2_score(y_valid, preds)
        
        return r2 # Optuna intentará maximizar este valor

        
    # 1. Cargar datos
    df = pd.read_csv("data/CHEMBL1075104_nonredundant.csv")
    
    # 2. Featurización (Usando Morgan Fingerprints 2048)
    featurizer = dc.feat.CircularFingerprint(size=2048, radius=2)
    loader = dc.data.CSVLoader(tasks=["pchembl_value"], feature_field="canonical_smiles", featurizer=featurizer)
    dataset = loader.create_dataset("data/CHEMBL1075104_nonredundant.csv")
    
    # 3. Scaffold Splitting (80/10/10)
    splitter = dc.splits.ScaffoldSplitter()
    train_ds, valid_ds, test_ds = splitter.train_valid_test_split(dataset)
    
    # Convertir a formato compatible con XGBoost
    X_train, y_train = train_ds.X, train_ds.y.ravel()
    X_valid, y_valid = valid_ds.X, valid_ds.y.ravel()
    X_test, y_test = test_ds.X, test_ds.y.ravel()
    
    # Ejecutar optimización
    # Esto crea un archivo 'lrrk2_study.db' en vuestra carpeta actual
    study = optuna.create_study(
        study_name="lrrk2_optimization", 
        storage="sqlite:///studies/lrrk2_study.db", 
        load_if_exists=True,
        direction="maximize"
    )
    study.optimize(objective, n_trials=1)
    
    # 1. Crear una copia de los valores reales pero desordenados al azar
    y_train_random = np.random.permutation(y_train)
    
    # 2. Re-entrenar el modelo (con vuestros mejores parámetros de Optuna)
    # Usad los mismos parámetros que os dieron el 0.65
    model_random = xgb.XGBRegressor(**study.best_params)
    model_random.fit(X_train, y_train_random)
    
    # 3. Evaluar sobre el set de test REAL (el que no está desordenado)
    preds_random = model_random.predict(X_test)
    r2_rand = r2_score(y_test, preds_random)
    
    print(f"R2 con etiquetas aleatorias: {r2_rand:.4f}")
    
    if r2_rand <= 0:
        print("✅ ¡PRUEBA SUPERADA! El modelo original es legítimo.")
        
    else:
        print("⚠️ OJO: El modelo sigue prediciendo algo con datos al azar. Revisad el leakage.")
    
    
    # 1. Reentrenar con los mejores parámetros
    best_model = xgb.XGBRegressor(**study.best_params)
    best_model.fit(X_train, y_train)
    
    # 2. Guardar el modelo en el disco
    joblib.dump(best_model, "models/mejor_modelo_lrrk2_scaffold.pkl")
    print("Modelo guardado con éxito.")
    
    # 1. Métricas en el set de entrenamiento (qué tan bien se aprendió los datos)
    train_r2 = best_model.score(X_train, y_train)
    
    # 2. Métricas en el set de validación (el promedio que dio Optuna)
    val_r2 = study.best_value
    
    # 3. Métricas en el set de TEST (el mundo real)
    test_pred = best_model.predict(X_test)
    test_r2 = r2_score(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    # 4. Cálculo de la "Brecha de Generalización"
    gap = val_r2 - test_r2
    
    print("\n" + "="*30)
    print("   INFORME DE CALIDAD FINAL")
    print("="*30)
    print(f"R² Entrenamiento: {train_r2:.3f}")
    print(f"R² Validación (Optuna): {val_r2:.3f}")
    print(f"R² Test (Final): {test_r2:.3f}")
    print(f"RMSE Test: {test_rmse:.3f}")
    print("-"*30)
    
    if gap > 0.15:
        print("ALERTA: Overfitting detectado. El modelo rinde mucho mejor en validación que en test.")
    elif gap < 0:
        print("El modelo rinde mejor en test.")
    else:
        print("ROBUSTEZ: El modelo generaliza bien. La brecha es aceptable.")


if __name__ == "__main__":
    run_pipeline()


