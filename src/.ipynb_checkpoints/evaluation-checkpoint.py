import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import optuna
import deepchem as dc

def evaluate_xgb(protein, train_ds, valid_ds, test_ds):

    model = joblib.load(f"models/{protein}_xgb.pkl")
    
    X_train, y_train = train_ds.X, train_ds.y
    X_valid, y_valid = valid_ds.X, valid_ds.y
    X_test, y_test = test_ds.X, test_ds.y
    
    train_pred = model.predict(X_train)
    train_score = r2_score(y_train, train_pred)
        
    valid_pred = model.predict(X_valid)
    valid_score = r2_score(y_valid, valid_pred)
    
    test_pred = model.predict(X_test)
    test_score = r2_score(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))


    test_true = test_ds.y

    plt.figure(figsize=(7, 7))
    plt.scatter(test_true, test_pred, alpha=0.5, c='orange', edgecolors='k')
    plt.plot([test_true.min(), test_true.max()], [test_true.min(), test_true.max()], 'r--', lw=2)
    plt.xlabel('Real pChEMBL (Experimental)')
    plt.ylabel('Predicted pChEMBL')
    plt.title('XGBRegressor')
    plt.show()

    return train_score, valid_score, test_score, test_rmse


def evaluate_gat(protein, train_ds, valid_ds, test_ds):
    storage_name = f"sqlite:///studies/{protein}_study_GAT.db"
    study = optuna.load_study(study_name=f"{protein}_optimization", storage=storage_name)
    
    model = dc.models.GATModel(
        n_tasks=1,
        graph_attention_layers=[16] * study.best_params['n_layers'],
        n_attention_heads=study.best_params['n_heads'],
        dropout=study.best_params['dropout'],
        learning_rate=study.best_params['lr'],
        model_dir=f"models/{protein}_gat/" 
    )
    
    model.restore()

    metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)
    metric_rmse = dc.metrics.Metric(dc.metrics.rms_score)
    
    train_score = model.evaluate(train_ds, [metric])
    valid_score = model.evaluate(valid_ds, [metric])
    test_score = model.evaluate(test_ds, [metric, metric_rmse])

    test_pred = model.predict(test_ds)
    test_true = test_ds.y

    plt.scatter(test_true, test_pred, alpha=0.5, c='teal', edgecolors='k')
    plt.plot([test_true.min(), test_true.max()], [test_true.min(), test_true.max()], 'r--', lw=2)
    plt.xlabel('pChEMBL Real (Experimental)')
    plt.ylabel('pChEMBL Predicho')
    plt.title('Graph Attention Network')
    plt.show()

    return train_score['pearson_r2_score'], valid_score['pearson_r2_score'], test_score['pearson_r2_score'], test_score['rms_score']
    
    

def evaluate_ensemble():
    return