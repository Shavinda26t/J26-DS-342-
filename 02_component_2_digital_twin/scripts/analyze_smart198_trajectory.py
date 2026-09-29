from pathlib import Path
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path(
    r"D:\db module 4th yera 1sem\data_Q1_2023\data_Q1_2023"
)

OUTPUT_FILE = Path(
    r"D:\Project\J26-DS-342-\02_component_2_digital_twin"
    r"\results\smart198_individual_trajectory.csv"
)

SMART_COLUMN = "smart_198_normalized"
WINDOWS = [30, 14, 7, 3, 1, 0]


# ============================================================
# FIND INPUT FILES
# ============================================================

files = sorted(DATA_DIR.glob("*.csv"))

print(f"Files found: {len(files)}")

if not files:
    raise FileNotFoundError(
        f"No CSV files found in {DATA_DIR}"
    )


# ============================================================
# STEP 1 — FIND FAILURE DATES
# ============================================================

print()
print("Finding HDD failure dates...")
print("----------------------------------------")

failure_dates = {}

for i, file in enumerate(files, start=1):

    print(
        f"Processing failure data "
        f"{i}/{len(files)}: {file.name}"
    )

    df = pd.read_csv(
        file,
        usecols=["date", "serial_number", "failure"]
    )

    df["date"] = pd.to_datetime(df["date"])

    failed = df[df["failure"] == 1]

    for serial, date in zip(
        failed["serial_number"],
        failed["date"]
    ):
        failure_dates[serial] = date


print()
print(f"Unique failed HDDs: {len(failure_dates)}")


# ============================================================
# STEP 2 — COLLECT SMART 187 OBSERVATIONS
# ============================================================

print()
print("Collecting SMART 187 trajectory...")
print("----------------------------------------")

records = []

required_columns = [
    "date",
    "serial_number",
    SMART_COLUMN
]

for i, file in enumerate(files, start=1):

    print(
        f"Processing SMART 187 "
        f"{i}/{len(files)}: {file.name}"
    )

    available_columns = pd.read_csv(
        file,
        nrows=0
    ).columns.tolist()

    columns_to_read = [
        column
        for column in required_columns
        if column in available_columns
    ]

    if SMART_COLUMN not in columns_to_read:
        continue

    df = pd.read_csv(
        file,
        usecols=columns_to_read
    )

    df["date"] = pd.to_datetime(df["date"])

    # Only failed HDDs.
    df = df[
        df["serial_number"].isin(failure_dates)
    ].copy()

    if df.empty:
        continue

    # Add failure date.
    df["failure_date"] = df["serial_number"].map(
        failure_dates
    )

    # Days before failure.
    df["days_before_failure"] = (
        df["failure_date"] - df["date"]
    ).dt.days

    # Keep requested windows.
    df = df[
        df["days_before_failure"].isin(WINDOWS)
    ].copy()

    if df.empty:
        continue

    records.append(
        df[
            [
                "serial_number",
                "days_before_failure",
                SMART_COLUMN
            ]
        ]
    )


if not records:
    raise RuntimeError(
        "No SMART 187 observations found."
    )


observations = pd.concat(
    records,
    ignore_index=True
)


# ============================================================
# STEP 3 — CREATE HDD × TIME-WINDOW TABLE
# ============================================================

print()
print("Creating individual HDD trajectories...")
print("----------------------------------------")

trajectory = observations.pivot_table(
    index="serial_number",
    columns="days_before_failure",
    values=SMART_COLUMN,
    aggfunc="mean"
)

# Keep only HDDs with at least two observations.
trajectory = trajectory.dropna(
    how="all"
)

trajectory.columns = [
    f"day_minus_{column}"
    if column != 0
    else "failure_day"
    for column in trajectory.columns
]


# ============================================================
# STEP 4 — SAVE INDIVIDUAL TRAJECTORIES
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

trajectory.to_csv(
    OUTPUT_FILE
)


# ============================================================
# STEP 5 — SUMMARY BY TIME WINDOW
# ============================================================

summary = []

for column in trajectory.columns:

    values = trajectory[column].dropna()

    if values.empty:
        continue

    summary.append(
        {
            "time_window": column,
            "sample_count": len(values),
            "mean": values.mean(),
            "median": values.median(),
            "std": values.std(),
            "min": values.min(),
            "max": values.max()
        }
    )


summary_df = pd.DataFrame(summary)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("========================================")
print("SMART 187 INDIVIDUAL TRAJECTORY")
print("========================================")

print(
    f"HDDs represented: "
    f"{trajectory.index.nunique()}"
)

print()
print(summary_df.to_string(index=False))

print()
print("Saved individual trajectories to:")
print(OUTPUT_FILE)

print()
print("Analysis complete.")