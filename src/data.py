import pandas as pd
from pathlib import Path

data_path = Path(__file__).parent.parent / "data" / "bank-additional-full.csv"
df = pd.read_csv(data_path)

print(df.head())