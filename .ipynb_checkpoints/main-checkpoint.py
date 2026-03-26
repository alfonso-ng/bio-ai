import argparse
import sys
from src.pipeline import run_training, run_evaluation, make_prediction

def main():
    # 1. Creamos el manejador principal de argumentos
    parser = argparse.ArgumentParser(
        description="Ligand Based Drug Discovery Platform"
    )
    
    # 2. Definimos los "sub-comandos" (acciones que el software puede hacer)
    subparsers = parser.add_subparsers(dest="command", help="Comandos disponibles")

    # --- COMANDO: TRAIN ---
    # Ejemplo: python main.py train --protein CHEMBL4210421.csv --model gat
    train_parser = subparsers.add_parser("train", help="Train new model")
    train_parser.add_argument("--protein", type=str, required=True, help="CHEMBL file route")
    train_parser.add_argument("--model", choices=["gat", "xgb", "ensemble"], default="xgb", help="Model architecture type")
    train_parser.add_argument("--trials", type=int, help="Number of Optuna trials")

    # --- COMANDO: EVALUATE ---
    # Ejemplo: python main.py evaluate --protein CHEMBL4210421.csv --model gat
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate existing models and generate graphs")
    eval_parser.add_argument("--protein", type=str, required=True)
    eval_parser.add_argument("--model", choices=["gat", "xgb", "ensemble"], default="ensemble")

    # --- COMANDO: PREDICT ---
    # Ejemplo: python main.py predict --protein LRRK2 --smiles input.csv
    predict_parser = subparsers.add_parser("predict", help="Predict activities for a list of SMILES")
    predict_parser.add_argument("--protein", type=str, required=True)
    predict_parser.add_argument("--model", choices=["gat", "xgb", "ensemble"], default="xgb", help="Model architecture type")
    predict_parser.add_argument("--smiles", type=str, required=True, help="Input SMILES file route")

    # 3. Parsear los argumentos de la terminal
    args = parser.parse_args()

    # 4. Lógica de enrutamiento: ¿Qué función de 'src/pipeline.py' llamamos?
    if args.command == "train":
        print(f"[*] Initiating training phase for {args.protein} using {args.model}...")
        run_training(args.protein, args.model, args.trials)
        
    elif args.command == "evaluate":
        print(f"[*] Evaluation results of model {args.model} for {args.protein}...")
        run_evaluation(args.protein, args.model)
        
    elif args.command == "predict":
        # Esta función nos devolverá el valor de pChEMBL predicho
        prediction = make_prediction(args.protein, args.model, args.smiles)
        # print(f"\n[RESULT] Predicted affinity for {args.protein}: {prediction:.3f} pChEMBL")
        
    else:
        # Si el usuario no escribe ningún comando, mostramos la ayuda
        parser.print_help()

if __name__ == "__main__":
    main()