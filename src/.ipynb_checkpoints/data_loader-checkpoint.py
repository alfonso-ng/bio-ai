from chembl_webresource_client.new_client import new_client
import pandas as pd
import os.path
import numpy as np

def download_chembl_data(target):

    file_path = f"data/{target}_raw_data.csv"
    print(file_path)
    if os.path.exists(file_path):
        print("Data found locally, no need for fetching")
        df = pd.read_csv(file_path)
    else:
        # 1. Buscar actividades para target
        print("Fetching data...")
        activities = new_client.activity
        res = activities.filter(target_chembl_id=target).filter(standard_type='IC50')
        print(f" Read {len(res)} activities")
        print("Creating DataFrame...")
        
        # 2. Convertir a DataFrame (esto puede tardar un poco por el volumen de datos)
        df = pd.DataFrame.from_dict(res)
        print("DataFrame created")
        print("Cleaning data...")
    
    # 3. Limpiar valores nulos y duplicados
    df = df.dropna(subset=['standard_value', 'canonical_smiles'])
    # Aplicamos la conversión
    df['value_nm'] = df.apply(convert_to_nm, axis=1)
    
    # Eliminamos filas que no pudimos convertir (None) o con valor 0 (error en datos)
    df = df.dropna(subset=['value_nm'])
    df = df[df['value_nm'] > 0]

    # Calculamos pchembl_value a partir de standard_value
    df["pchembl_value"] = 9 - np.log10(df["value_nm"])

    # 4. Guardar los datos para usar localmente en el futuro
    print("Saving data locally...")
    df.to_csv(file_path)
    print("Setup succesful!")
    return df

def convert_to_nm(row):
    value = float(row['standard_value'])
    unit = row['standard_units']
    
    if unit == 'nM':
        return value
    elif unit == 'uM' or unit == 'µM':
        return value * 1000
    elif unit == 'mM':
        return value * 1_000_000
    elif unit == 'M':
        return value * 1_000_000_000
    else:
        return None # Unidades raras que mejor descartar

def load_data(target):

    filepath = f"data/{target}_nonredundant.csv"
    df = pd.read_csv(filepath)
    return df