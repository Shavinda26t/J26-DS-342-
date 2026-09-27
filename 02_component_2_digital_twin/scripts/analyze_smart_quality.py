from pathlib import Path
from collections import defaultdict
import pandas as pd

DATA_DIR = Path(
    r"D:\db module 4th yera 1sem\data_Q1_2023\data_Q1_2023"
)

files = sorted(DATA_DIR.glob("*.csv"))

if not files:
    raise FileNotFoundError("No CSV files found.")

print(f"Files found: {len(files)}")

# Get SMART column names from the first file
sample = pd.read_csv(files[0], nrows=5)

smart_columns = [
    col for col in sample.columns
    if col.startswith("smart_")
]

print(f"SMART columns: {len(smart_columns)}")
print()

# Statistics
total_rows = 0

non_null_count = defaultdict(int)
unique_values = defaultdict(set)
non_zero_count = defaultdict(int)

for i, file in enumerate(files, start=1):

    print(f"Processing {i}/{len(files)}: {file.name}")

    df = pd.read_csv(
        file,
        usecols=smart_columns
    )

    total_rows += len(df)

    for column in smart_columns:

        series = df[column]

        non_null_count[column] += series.notna().sum()

        non_zero_count[column] += (
            (series.fillna(0) != 0).sum()
        )

        # Limit unique-value memory
        values = series.dropna().unique()

        if len(unique_values[column]) < 1000:
            unique_values[column].update(
                values[:1000]
            )

print()
print("=" * 70)
print("Q1 2023 SMART DATA QUALITY ANALYSIS")
print("=" * 70)

results = []

for column in smart_columns:

    missing_percentage = (
        1 - non_null_count[column] / total_rows
    ) * 100

    non_zero_percentage = (
        non_zero_count[column] / total_rows
    ) * 100

    results.append({
        "column": column,
        "missing_percentage": missing_percentage,
        "non_zero_percentage": non_zero_percentage,
        "unique_values_sample": len(unique_values[column])
    })

result_df = pd.DataFrame(results)

result_df = result_df.sort_values(
    "missing_percentage"
)

print()
print("Most complete SMART attributes")
print("-" * 70)

print(
    result_df.head(30).to_string(index=False)
)

print()
print("Attributes with highest non-zero percentage")
print("-" * 70)

print(
    result_df.sort_values(
        "non_zero_percentage",
        ascending=False
    ).head(30).to_string(index=False)
)

# Save results
output_dir = Path(
    "02_component_2_digital_twin/results"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

output_file = output_dir / "smart_data_quality.csv"

result_df.to_csv(
    output_file,
    index=False
)

print()
print(f"Results saved to: {output_file}")