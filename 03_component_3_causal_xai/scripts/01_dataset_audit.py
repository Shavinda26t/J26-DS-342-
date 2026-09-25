from pathlib import Path
from collections import defaultdict

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

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
    / "dataset_audit"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


csv_files = sorted(DATA_DIR.glob("*.csv"))

print(f"CSV files found: {len(csv_files)}")

if not csv_files:
    raise FileNotFoundError(
        f"No CSV files found in {DATA_DIR}"
    )


model_drives = defaultdict(set)
model_failed_drives = defaultdict(set)
model_rows = defaultdict(int)

total_rows = 0
all_dates = []


for number, file in enumerate(csv_files, start=1):

    print(
        f"[{number}/{len(csv_files)}] "
        f"Processing {file.name}"
    )

    df = pd.read_csv(
        file,
        usecols=[
            "date",
            "serial_number",
            "model",
            "failure"
        ],
        low_memory=False
    )

    total_rows += len(df)

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    all_dates.append(df["date"].min())
    all_dates.append(df["date"].max())

    for model, group in df.groupby("model"):

        model_rows[model] += len(group)

        model_drives[model].update(
            group["serial_number"]
            .dropna()
            .astype(str)
        )

        failed_serials = (
            group.loc[
                group["failure"] == 1,
                "serial_number"
            ]
            .dropna()
            .astype(str)
        )

        model_failed_drives[model].update(
            failed_serials
        )


summary_rows = []

for model in model_drives:

    unique_drives = len(
        model_drives[model]
    )

    failed_drives = len(
        model_failed_drives[model]
    )

    failure_rate = (
        failed_drives / unique_drives
        if unique_drives > 0
        else 0
    )

    summary_rows.append(
        {
            "model": model,
            "daily_records":
                model_rows[model],
            "unique_drives":
                unique_drives,
            "failed_drives":
                failed_drives,
            "failure_rate":
                failure_rate
        }
    )


summary = pd.DataFrame(summary_rows)

summary = summary.sort_values(
    by=[
        "failed_drives",
        "unique_drives"
    ],
    ascending=False
)

summary.to_csv(
    OUTPUT_DIR
    / "Q4_2025_model_summary.csv",
    index=False
)


print("\n========== DATASET SUMMARY ==========")

print(
    "Total rows:",
    f"{total_rows:,}"
)

print(
    "Date range:",
    min(all_dates),
    "to",
    max(all_dates)
)

print(
    "Number of HDD models:",
    len(summary)
)

print("\nTop 25 HDD models:\n")

print(
    summary.head(25)
    .to_string(index=False)
)

print(
    "\nSaved to:",
    OUTPUT_DIR
    / "Q4_2025_model_summary.csv"
)