from project_paths import RAW_DATA_DIR
from pathlib import Path
import pandas as pd

DATA_DIR = Path(
    RAW_DATA_DIR
)

files = sorted(DATA_DIR.glob("*.csv"))

if not files:
    raise FileNotFoundError("No CSV files found.")

# Inspect only the first daily file
file = files[0]

print(f"Inspecting: {file.name}")
print()

df = pd.read_csv(file, nrows=5)

print(f"Total columns: {len(df.columns)}")
print()

print("All columns")
print("----------------------------------------")

for column in df.columns:
    print(column)

print()
print("SMART columns")
print("----------------------------------------")

smart_columns = [
    column
    for column in df.columns
    if column.startswith("smart_")
]

print(f"Total SMART columns: {len(smart_columns)}")

for column in smart_columns:
    print(column)