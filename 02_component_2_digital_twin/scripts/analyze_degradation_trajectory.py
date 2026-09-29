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
    r"\results\degradation_trajectory.csv"
)

# Candidate SMART attributes selected from the quality analysis.
# We start with a small set rather than all 174 SMART columns.
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

WINDOWS = [30, 14, 7, 3, 1]


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
# STEP 1 — FIND HDD FAILURE DATES
# ============================================================

print()
print("Finding HDD failure dates...")
print("----------------------------------------")

failure_dates = {}

for i, file in enumerate(files, start=1):

    print(f"Processing failure data {i}/{len(files)}: {file.name}")

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
# STEP 2 — COLLECT PRE-FAILURE SMART OBSERVATIONS
# ============================================================

print()
print("Collecting pre-failure SMART observations...")
print("----------------------------------------")

records = []

required_columns = [
    "date",
    "serial_number",
    "failure"
] + SMART_COLUMNS

for i, file in enumerate(files, start=1):

    print(
        f"Processing SMART data {i}/{len(files)}: "
        f"{file.name}"
    )

    # Read only columns needed for this analysis.
    available_columns = pd.read_csv(
        file,
        nrows=0
    ).columns.tolist()

    columns_to_read = [
        column
        for column in required_columns
        if column in available_columns
    ]

    df = pd.read_csv(
        file,
        usecols=columns_to_read
    )

    df["date"] = pd.to_datetime(df["date"])

    # Keep only HDDs that eventually failed in Q1.
    df = df[
        df["serial_number"].isin(failure_dates)
    ].copy()

    if df.empty:
        continue

    # Add failure date for each HDD.
    df["failure_date"] = df["serial_number"].map(
        failure_dates
    )

    # Calculate days until failure.
    df["days_before_failure"] = (
        df["failure_date"] - df["date"]
    ).dt.days

    # Keep observations before failure and on failure day.
    df = df[
        (df["days_before_failure"] >= 0)
        & (df["days_before_failure"] <= max(WINDOWS))
    ]

    if df.empty:
        continue

    keep_columns = [
        "date",
        "serial_number",
        "failure_date",
        "days_before_failure"
    ] + [
        column
        for column in SMART_COLUMNS
        if column in df.columns
    ]

    records.append(
        df[keep_columns]
    )


if not records:
    raise RuntimeError(
        "No pre-failure SMART observations were found."
    )


trajectory = pd.concat(
    records,
    ignore_index=True
)


# ============================================================
# STEP 3 — ALIGN TO FAILURE WINDOWS
# ============================================================

print()
print("Creating failure-relative trajectory...")
print("----------------------------------------")

trajectory = trajectory[
    trajectory["days_before_failure"].isin(
        WINDOWS + [0]
    )
].copy()


# ============================================================
# STEP 4 — CALCULATE SUMMARY STATISTICS
# ============================================================

summary_records = []

for smart_column in SMART_COLUMNS:

    if smart_column not in trajectory.columns:
        continue

    for window in WINDOWS + [0]:

        values = trajectory.loc[
            trajectory["days_before_failure"] == window,
            smart_column
        ].dropna()

        if values.empty:
            continue

        summary_records.append(
            {
                "smart_column": smart_column,
                "days_before_failure": window,
                "sample_count": len(values),
                "mean": values.mean(),
                "median": values.median(),
                "std": values.std(),
                "min": values.min(),
                "max": values.max(),
            }
        )


summary = pd.DataFrame(summary_records)


# ============================================================
# STEP 5 — SAVE RESULTS
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

summary.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("========================================")
print("SMART DEGRADATION TRAJECTORY ANALYSIS")
print("========================================")

print(f"Failed HDDs analysed: {len(failure_dates)}")
print(
    f"Pre-failure observations: "
    f"{len(trajectory):,}"
)
print(
    f"Trajectory summary rows: "
    f"{len(summary):,}"
)

print()
print(f"Saved results to:")
print(OUTPUT_FILE)

print()
print("Analysis complete.")