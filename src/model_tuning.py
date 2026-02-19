import optuna
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import optuna.visualization as vis

# 1. Gráfico de Historia de Optimización


def run_optimization(X_train, y_train, trials):
    def objective(trial):
        # 1. Sugerir el modelo a probar
        classifier_name = trial.suggest_categorical("classifier", ["XGBoost"])
        
        if classifier_name == "RandomForest":
            n_estimators = trial.suggest_int("rf_n_estimators", 100, 1000)
            max_depth = trial.suggest_int("rf_max_depth", 10, 50)
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1)
            
        else:
            param = {
                "n_estimators": trial.suggest_int("xgb_n_estimators", 100, 1000),
                "learning_rate": trial.suggest_float("xgb_lr", 0.01, 0.3),
                "max_depth": trial.suggest_int("xgb_max_depth", 3, 12),
                "subsample": trial.suggest_float("xgb_subsample", 0.5, 1.0)
            }
            model = xgb.XGBRegressor(**param, n_jobs=-1)
    
        # 2. Validación Cruzada (Cross-Validation)
        # No usamos solo un split, sino 5 rotaciones para asegurar que el modelo es estable
        score = cross_val_score(model, X_train, y_train, cv=5, scoring='r2').mean()
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials) # Prueba 30 combinaciones distintas
    
    print(f"Mejor R2 encontrado: {study.best_value}")
    print(f"Mejores parámetros: {study.best_params}")

    return study.best_value, study.best_params


def evaluate_final_model(model, X_train, y_train, X_test, y_test, best_value):
    # 1. Métricas en el set de entrenamiento (qué tan bien se aprendió los datos)
    train_r2 = model.score(X_train, y_train)
    
    # 2. Métricas en el set de validación (el promedio que dio Optuna)
    val_r2 = best_value
    
    # 3. Métricas en el set de TEST (el mundo real)
    test_pred = model.predict(X_test)
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
    
    return {"train_r2": train_r2, "test_r2": test_r2, "gap": gap}