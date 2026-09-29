from pathlib import Path
import pandas as pd

DATA_DIR = Path(
    r"D:\db module 4th yera 1sem\data_Q1_2023\data_Q1_2023"
)

OUTPUT_DIR = Path("data/interim")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "temporal_features.csv"

SMART_COLUMNS = [
    "smart_1_normalized",
    "smart_5_normalized",
    "smart_9_normalized",
    "smart_12_normalized",
    "smart_187_normalized",
    "smart_192_normalized",
    "smart_194_normalized",
    "smart_197_normalized",
    "smart_198_normalized",
    "smart_199_normalized",
]

USE_COLUMNS = ["date", "serial_number", "failure"] + SMART_COLUMNS

files = sorted(DATA_DIR.glob("*.csv"))

print(f"Files found: {len(files)}")

all_data = []

for i, file in enumerate(files, start=1):
    print(f"Reading {i}/{len(files)}: {file.name}")

    df = pd.read_csv(file, usecols=USE_COLUMNS)

    df["date"] = pd.to_datetime(df["date"])

    all_data.append(df)

data = pd.concat(all_data, ignore_index=True)

print(f"Total observations: {len(data):,}")

# Sort each HDD chronologically
data = data.sort_values(
    ["serial_number", "date"]
).reset_index(drop=True)

# Create temporal features for each SMART attribute
for column in SMART_COLUMNS:

    # Previous observation
    data[f"{column}_previous"] = (
        data.groupby("serial_number")[column].shift(1)
    )

    # Change from previous observation
    data[f"{column}_delta"] = (
        data[column] - data[f"{column}_previous"]
    )

    # 7-day change
    data[f"{column}_change_7d"] = (
        data.groupby("serial_number")[column].diff(7)
    )

    # 14-day change
    data[f"{column}_change_14d"] = (
        data.groupby("serial_number")[column].diff(14)
    )

    # Rolling mean over previous 7 observations
    data[f"{column}_rolling_mean_7"] = (
        data.groupby("serial_number")[column]
        .transform(lambda x: x.rolling(7, min_periods=2).mean())
    )

    # Rolling standard deviation
    data[f"{column}_rolling_std_7"] = (
        data.groupby("serial_number")[column]
        .transform(lambda x: x.rolling(7, min_periods=2).std())
    )

# Save result
data.to_csv(OUTPUT_FILE, index=False)

print()
print("Temporal feature engineering completed.")
print(f"Output: {OUTPUT_FILE}")
print(f"Rows: {len(data):,}")
print(f"Columns: {len(data.columns)}")