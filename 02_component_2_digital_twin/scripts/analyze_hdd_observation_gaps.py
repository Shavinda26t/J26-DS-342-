from pathlib import Path
import pandas as pd

DATA_DIR = Path(
    r"D:\db module 4th yera 1sem\data_Q1_2023\data_Q1_2023"
)

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "hdd_observation_gap_analysis.csv"

USE_COLUMNS = [
    "date",
    "serial_number",
]

files = sorted(DATA_DIR.glob("*.csv"))

print("=" * 70)
print("HDD OBSERVATION GAP ANALYSIS")
print("=" * 70)

print(f"Files found: {len(files)}")

# ---------------------------------------------------------
# 1. Read only date and serial number
# ---------------------------------------------------------

all_data = []

for i, file in enumerate(files, start=1):
    print(f"Reading {i}/{len(files)}: {file.name}")

    df = pd.read_csv(
        file,
        usecols=USE_COLUMNS
    )

    df["date"] = pd.to_datetime(df["date"])

    all_data.append(df)

data = pd.concat(
    all_data,
    ignore_index=True
)

print()
print(f"Total observations: {len(data):,}")

# ---------------------------------------------------------
# 2. Remove duplicate HDD/date observations if any
# ---------------------------------------------------------

data = data.drop_duplicates(
    subset=["serial_number", "date"]
)

# ---------------------------------------------------------
# 3. Sort chronologically
# ---------------------------------------------------------

data = data.sort_values(
    ["serial_number", "date"]
).reset_index(drop=True)

# ---------------------------------------------------------
# 4. Calculate gap between consecutive observations
# ---------------------------------------------------------

data["previous_date"] = (
    data.groupby("serial_number")["date"].shift(1)
)

data["gap_days"] = (
    data["date"] - data["previous_date"]
).dt.days

# ---------------------------------------------------------
# 5. Analyse gaps
# ---------------------------------------------------------

valid_gaps = data["gap_days"].dropna()

print()
print("=" * 70)
print("OBSERVATION GAP SUMMARY")
print("=" * 70)

print(f"Unique HDDs: {data['serial_number'].nunique():,}")

print(
    f"Consecutive observations: "
    f"{len(valid_gaps):,}"
)

print(
    f"Mean gap: "
    f"{valid_gaps.mean():.2f} days"
)

print(
    f"Median gap: "
    f"{valid_gaps.median():.2f} days"
)

print(
    f"Maximum gap: "
    f"{valid_gaps.max():.0f} days"
)

print()

print("Gap distribution:")

gap_distribution = (
    valid_gaps
    .value_counts()
    .sort_index()
)

print(gap_distribution.head(20))

# ---------------------------------------------------------
# 6. Important gap categories
# ---------------------------------------------------------

print()
print("=" * 70)
print("IMPORTANT GAP CATEGORIES")
print("=" * 70)

for threshold in [1, 2, 3, 7, 14, 30]:

    count = (valid_gaps >= threshold).sum()

    percentage = (
        count / len(valid_gaps) * 100
    )

    print(
        f"Gaps >= {threshold:2d} days: "
        f"{count:,} "
        f"({percentage:.2f}%)"
    )

# ---------------------------------------------------------
# 7. HDD-level gap summary
# ---------------------------------------------------------

hdd_gap_summary = (
    data.groupby("serial_number")["gap_days"]
    .agg(
        observations="count",
        mean_gap="mean",
        max_gap="max"
    )
    .reset_index()
)

# ---------------------------------------------------------
# 8. Save HDD-level summary
# ---------------------------------------------------------

hdd_gap_summary.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print(f"Saved to:")
print(OUTPUT_FILE)