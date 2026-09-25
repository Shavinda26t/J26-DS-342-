from pathlib import Path
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Q4_2025"
    / "data_Q4_2025"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "smart_attribute_audit"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


SELECTED_MODEL = "ST12000NM0008"

csv_files = sorted(DATA_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        f"No CSV files found in {DATA_DIR}"
    )


print("Selected HDD model:", SELECTED_MODEL)
print("CSV files:", len(csv_files))


# -------------------------------------------------
# Get SMART column names from first CSV
# -------------------------------------------------

sample_columns = pd.read_csv(
    csv_files[0],
    nrows=0
).columns.tolist()

smart_columns = [
    col for col in sample_columns
    if col.startswith("smart_")
]

print("SMART columns found:", len(smart_columns))


# -------------------------------------------------
# Statistics containers
# -------------------------------------------------

stats = {
    col: {
        "total_rows": 0,
        "missing": 0,
        "non_missing": 0,
        "zero": 0,
        "non_zero": 0,
        "unique_values": set()
    }
    for col in smart_columns
}


selected_rows = 0


# -------------------------------------------------
# Scan files one by one
# -------------------------------------------------

for i, file in enumerate(csv_files, start=1):

    print(
        f"[{i}/{len(csv_files)}] "
        f"Processing {file.name}"
    )

    usecols = [
        "model",
        "serial_number",
        "date",
        "failure"
    ] + smart_columns

    df = pd.read_csv(
        file,
        usecols=usecols,
        low_memory=False
    )

    df = df[
        df["model"] == SELECTED_MODEL
    ]

    if df.empty:
        continue

    selected_rows += len(df)

    for col in smart_columns:

        series = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        total = len(series)

        missing = series.isna().sum()

        valid = series.dropna()

        zero = (valid == 0).sum()
        non_zero = (valid != 0).sum()

        stats[col]["total_rows"] += total
        stats[col]["missing"] += missing
        stats[col]["non_missing"] += len(valid)
        stats[col]["zero"] += zero
        stats[col]["non_zero"] += non_zero

        # Limit unique-value storage to avoid excessive RAM usage
        if len(stats[col]["unique_values"]) < 10000:

            unique_vals = valid.unique()

            remaining = (
                10000
                - len(stats[col]["unique_values"])
            )

            stats[col]["unique_values"].update(
                unique_vals[:remaining]
            )


# -------------------------------------------------
# Build summary table
# -------------------------------------------------

rows = []

for col, values in stats.items():

    total = values["total_rows"]

    if total == 0:
        continue

    missing_pct = (
        values["missing"] / total
    ) * 100

    non_zero_pct = (
        values["non_zero"] / total
    ) * 100

    rows.append(
        {
            "smart_attribute": col,
            "total_rows": total,
            "missing_rows":
                values["missing"],
            "missing_pct":
                round(missing_pct, 4),
            "non_zero_rows":
                values["non_zero"],
            "non_zero_pct":
                round(non_zero_pct, 4),
            "unique_values":
                len(values["unique_values"])
        }
    )


summary = pd.DataFrame(rows)

summary = summary.sort_values(
    by=[
        "missing_pct",
        "non_zero_pct"
    ],
    ascending=[
        True,
        False
    ]
)


output_file = (
    OUTPUT_DIR
    / "ST12000NM0008_smart_attribute_audit.csv"
)

summary.to_csv(
    output_file,
    index=False
)


print("\n========== SMART AUDIT ==========")

print("Selected model:", SELECTED_MODEL)
print("Selected rows:", f"{selected_rows:,}")

print("\nTop SMART attributes:\n")

print(
    summary.head(50)
    .to_string(index=False)
)

print("\nSaved to:")
print(output_file)