import pandas as pd
from pathlib import Path

def load_data_set():
    data_path = Path(__file__).parent.parent / "data" / "bank-additional-full.csv"
    df = pd.read_csv(data_path, sep=";")
    return df


