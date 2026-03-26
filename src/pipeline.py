import os
import deepchem as dc
import pandas as pd
import joblib
from src.training import train_gat, train_xgb, train_ensemble
from src.featurization import get_featurizer

def run_training(protein, model, trials):
    data_path = f"data/{protein}_nonredundant.csv"

    featurizer = get_featurizer(model)
    loader = dc.data.CSVLoader(
        tasks=["pchembl_value"],
        feature_field="canonical_smiles",
        featurizer=featurizer
    )
    dataset = loader.create_dataset(data_path)

    splitter = dc.splits.ScaffoldSplitter()
    train_ds, valid_ds, test_ds = splitter.train_valid_test_split(
        dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=42    
    )

    if model == "xgb":
        model = train_xgb(protein, train_ds, valid_ds, test_ds, trials)
    elif model == "gat":
        model= train_gat(protein, train_ds, valid_ds, test_ds, trials)
    else:
        train_ensemble(protein, train_ds, valid_ds, test_ds, trials)

def run_evaluation(protein, model):
    print(f"Evaluando modelo {model} para proteína {protein}")

def make_prediction(protein, model, smiles):
    print(f"Ejecutando predicción de {protein} con modelo {model} para archivo {smiles}")