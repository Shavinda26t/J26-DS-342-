from pathlib import Path
import pandas as pd

# ============================================================
# Component 2 - Separate Raw SMART Experiment
# This script does NOT modify the existing V5 dataset/model.
# ============================================================

SOURCE_DIR = Path(
    r"D:\db module 4th yera 1sem\data_Q1_2023\data_Q1_2023"
)

OUTPUT_DIR = Path(r".\results\raw_smart_experiment")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "raw_smart_temporal_features.csv"

# Start with SMART attributes commonly associated with degradation.
# SMART 1 is deliberately excluded because its raw meaning is vendor-dependent.
RAW_SMART_COLUMNS = [
    "smart_5_raw",
    "smart_187_raw",
    "smart_197_raw",
    "smart_198_raw",
]

BASE_COLUMNS = ["date", "serial_number", "failure"]
USE_COLUMNS = BASE_COLUMNS + RAW_SMART_COLUMNS


def main():
    print("=" * 72)
    print("COMPONENT 2 - BUILD RAW SMART EXPERIMENT FEATURES")
    print("=" * 72)
    print(f"Source: {SOURCE_DIR}")
    print(f"Output: {OUTPUT_FILE}")

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"Source folder not found: {SOURCE_DIR}\n"
            "Update SOURCE_DIR at the top of this script."
        )

    files = sorted(SOURCE_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {SOURCE_DIR}")

    # Check the first file so a missing SMART column fails clearly.
    header = pd.read_csv(files[0], nrows=0)
    missing = [c for c in USE_COLUMNS if c not in header.columns]
    if missing:
        raise ValueError(
            "Required columns missing from source CSV: "
            + ", ".join(missing)
        )

    frames = []
    for i, file in enumerate(files, start=1):
        print(f"Reading {i}/{len(files)}: {file.name}")
        frame = pd.read_csv(file, usecols=USE_COLUMNS)

        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        for column in RAW_SMART_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    del frames

    # Remove unusable identifiers/dates and order observations per drive.
    data = data.dropna(subset=["date", "serial_number"])
    data = data.sort_values(
        ["serial_number", "date"]
    ).reset_index(drop=True)

    # Avoid duplicated HDD/date rows if source files overlap.
    data = data.drop_duplicates(
        subset=["serial_number", "date"],
        keep="last"
    ).reset_index(drop=True)

    print(f"Observations after cleaning: {len(data):,}")
    print(f"Unique HDDs: {data['serial_number'].nunique():,}")

    grouped = data.groupby("serial_number", sort=False)

    # Raw feature availability flags and per-HDD temporal features.
    for column in RAW_SMART_COLUMNS:
        print(f"Engineering features for {column}...")

        data[f"{column}_available"] = (
            data[column].notna().astype("int8")
        )

        data[f"{column}_previous"] = grouped[column].shift(1)
        data[f"{column}_delta"] = data[column] - data[f"{column}_previous"]
        data[f"{column}_change_7obs"] = grouped[column].diff(7)
        data[f"{column}_change_14obs"] = grouped[column].diff(14)

        data[f"{column}_rolling_mean_7obs"] = (
            grouped[column]
            .transform(lambda x: x.rolling(7, min_periods=2).mean())
        )
        data[f"{column}_rolling_std_7obs"] = (
            grouped[column]
            .transform(lambda x: x.rolling(7, min_periods=2).std())
        )

    data.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 72)
    print("RAW SMART EXPERIMENT FEATURE DATASET CREATED")
    print("=" * 72)
    print(f"Output: {OUTPUT_FILE}")
    print(f"Rows: {len(data):,}")
    print(f"Columns: {len(data.columns)}")
    print("\nImportant:")
    print("- Existing V5 files and baseline model were not changed.")
    print("- '7obs' and '14obs' mean observations, not guaranteed calendar days.")
    print("- Raw SMART values can have different scales; fit scaling on training data only.")
    print("- This is the feature-building stage only; sequence generation and model training come next.")


if __name__ == "__main__":
    main()
