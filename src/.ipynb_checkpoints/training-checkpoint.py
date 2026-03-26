import deepchem as dc
import pandas as pd
import optuna
import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import joblib
import torch
import dgl
import dgllife


# TRAINING MODEL WITH XGBOOST
def train_xgb(protein, train_ds, valid_ds, test_ds, trials):
    X_train, y_train = train_ds.X, train_ds.y
    X_valid, y_valid = valid_ds.X, valid_ds.y
    X_test, y_test = test_ds.X, test_ds.y
    
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

        
        
        # Train model
        model = xgb.XGBRegressor(**param)
        model.fit(X_train, y_train)
    
        # Evaluate model with validation set
        preds = model.predict(X_valid)
        r2 = r2_score(y_valid, preds)
        
        return r2

    study = optuna.create_study(
        study_name=f"{protein}_optimization", 
        storage=f"sqlite:///studies/{protein}_study_XGB.db", 
        load_if_exists=True,
        direction="maximize"
    )
    study.optimize(objective, n_trials=trials)

    
    # Train model with best parameters selected by optuna
    best_model = xgb.XGBRegressor(**study.best_params)
    best_model.fit(X_train, y_train)

    
    
    train_pred = best_model.predict(X_train)
    train_r2 = r2_score(y_train, train_pred)
        
    # 2. Métricas en el set de validación (el promedio que dio Optuna)
    valid_pred = best_model.predict(X_valid)
    valid_r2 = r2_score(y_valid, valid_pred)
    
    # 3. Métricas en el set de TEST (el mundo real)
    test_pred = best_model.predict(X_test)
    test_r2 = r2_score(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    gap = valid_r2 - test_r2
    
    print("\n" + "="*30)
    print("   FINAL QUALITY REPORT")
    print("="*30)
    print(f"Training R²: {train_r2:.3f}")
    print(f"Validation R²: {valid_r2:.3f}")
    print(f"Test R²: {test_r2:.3f}")
    print(f"RMSE Test: {test_rmse:.3f}")
    print("-"*30)
    
    if gap > 0.15:
        print("WARNING: Overfitting detected. The model performs significantly better in validation than in test.")
    else:
        print("Gap is acceptable, the model performs well.")
        model_path = f"models/{protein}_xgb.pkl"
        joblib.dump(best_model, model_path)
        print(f"Successfully saved model {model_path}")

# TRAINING WITH GRAPH ATTENTION NETWORK
def train_gat(protein, train_ds, valid_ds, test_ds, trials):
    def objective(trial):
        # Definimos rangos más estrictos para evitar el overfitting (0.93 vs 0.42)
        n_layers = trial.suggest_int('n_layers', 2, 3) # Menos capas = más generalización
        n_heads = trial.suggest_categorical('n_heads', [4, 8, 12])
        dropout = trial.suggest_float('dropout', 0.35, 0.55) # Forzamos dropout alto
        lr = trial.suggest_float('lr', 5e-5, 1e-3, log=True)
        
        layers = [16] * n_layers
        
        model = dc.models.GATModel(
            n_tasks=1,
            graph_attention_layers=layers,
            n_attention_heads=n_heads,
            dropout=dropout,
            learning_rate=lr,
            mode='regression',
        )
        
        # Entrenamiento controlado
        model.fit(train_ds, nb_epoch=100)
        
        metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)
        valid_score = model.evaluate(valid_ds, [metric])['pearson_r2_score']
        
        return valid_score
    
    # 2. Creamos o cargamos el estudio persistente
    study = optuna.create_study(
        study_name=f"{protein}_optimization", 
        storage=f"sqlite:///studies/{protein}_study_GAT.db", 
        direction="maximize",
        load_if_exists=True
    )

    print(f"Estudio finalizado.")
    
    # 3. Lanzamos la optimización
    study.optimize(objective, n_trials=trials)
    

    # 1. Configuración del modelo con tus mejores parámetros de Optuna
    model_dir = f"models/{protein}_gat"
    metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)
    rmse_metric = dc.metrics.Metric(dc.metrics.rms_score)
    
    final_model = dc.models.GATModel(
        n_tasks=1,
        graph_attention_layers=[16] * study.best_params['n_layers'],
        n_attention_heads=study.best_params['n_heads'],
        dropout=study.best_params['dropout'],
        learning_rate=study.best_params['lr'],
        model_dir=model_dir,
    )

    final_model.restore()
    
    # 2. El Callback: Se encarga de guardar el MEJOR modelo en disco automáticamente
    # Evalúa cada 10 épocas y si mejora el R2, guarda un checkpoint.
    callback = dc.models.ValidationCallback(
        valid_ds, 
        interval=50, 
        metrics=[metric],
        save_dir=model_dir
    )
    
    
    # 3. Listas para la gráfica (Manual)
    train_errors = []
    epochs = 500

    
    #final_model.fit(train_ds, nb_epoch=epochs, callbacks=callback, all_losses=train_errors)
    
    # 4. EL PASO MÁGICO: Restaurar el mejor modelo encontrado por el callback
    print("\nRestaurando el mejor modelo guardado durante el entrenamiento...")
    final_model.restore() 
    
    # 5. Evaluación final (Ya con el modelo optimizado cargado)
    final_r2_test = final_model.evaluate(test_ds, [metric])['pearson_r2_score']
    print(f"--- RESULTADO FINAL EN TEST (Modelo Restaurado): {final_r2_test:.4f} ---")


def train_ensemble(protein, optimize):
    return