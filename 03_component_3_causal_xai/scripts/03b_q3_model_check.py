from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

Q3_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Q4_2025"
    / "data_Q3_2025"
)

SELECTED_MODEL = "ST12000NM0008"

files = sorted(Q3_DIR.glob("*.csv"))

if not files:
    raise FileNotFoundError(
        f"No Q3 CSV files found in {Q3_DIR}"
    )

total_rows = 0
selected_rows = 0
serials = set()
failed_serials = set()
dates = []

for i, file in enumerate(files, start=1):

    print(
        f"[{i}/{len(files)}] "
        f"{file.name}"
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

    df = df[
        df["model"] == SELECTED_MODEL
    ].copy()

    if df.empty:
        continue

    selected_rows += len(df)

    serials.update(
        df["serial_number"]
        .dropna()
        .astype(str)
    )

    failed_serials.update(
        df.loc[
            df["failure"] == 1,
            "serial_number"
        ]
        .dropna()
        .astype(str)
    )

    dates.extend(
        pd.to_datetime(
            df["date"],
            errors="coerce"
        ).dropna()
    )


print("\n" + "=" * 60)
print("Q3 MODEL CHECK")
print("=" * 60)

print(
    "CSV files:",
    len(files)
)

print(
    "All Q3 rows:",
    f"{total_rows:,}"
)

print(
    f"{SELECTED_MODEL} rows:",
    f"{selected_rows:,}"
)

print(
    "Unique drives:",
    f"{len(serials):,}"
)

print(
    "Failed drives:",
    f"{len(failed_serials):,}"
)

if dates:
    print(
        "Date range:",
        min(dates),
        "to",
        max(dates)
    )