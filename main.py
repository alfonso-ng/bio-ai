# main.py
import config
from src.data_loader import download_chembl_data, load_data
from src.features import featurize_smiles
from src.model_tuning import run_optimization, evaluate_final_model
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import joblib

def run_pipeline():
    # 1. Obtener datos
    df = load_data(config.TARGET_ID)
    
    # 2. Vectorizar
    X = featurize_smiles(df, config.RADIUS, config.N_BITS)
    y = df['pchembl_value']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.TEST_SIZE, random_state=42)
    
    # 3. Optimizar y Entrenar (El paso que tarda)
    best_value, best_params = run_optimization(X_train, y_train, trials=config.OPTUNA_TRIALS)
    
    # 4. Crear y Guardar modelo final
    if best_params["classifier"] == "RandomForest":
        final_model = RandomForestRegressor(
            n_estimators=best_params["rf_n_estimators"],
            max_depth=best_params["rf_max_depth"],
            n_jobs=-1
        )
    else:
        final_model = xgb.XGBRegressor(
            n_estimators=best_params["xgb_n_estimators"],
            learning_rate=best_params["xgb_lr"],
            max_depth=best_params["xgb_max_depth"],
            subsample=best_params["xgb_subsample"],
            n_jobs=-1
        )
        
    final_model.fit(X_train, y_train)
    
    report = evaluate_final_model(final_model, X_train, y_train, X_test, y_test, best_value)
    joblib.dump(final_model, f"models/best_{config.TARGET_ID}_model.pkl")


if __name__ == "__main__":
    run_pipeline()