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
    r"\results\individual_degradation_analysis.csv"
)

SMART_COLUMNS = [
    "smart_1_normalized",
    "smart_5_normalized",
    "smart_187_normalized",
    "smart_197_normalized",
    "smart_198_normalized",
]

# We compare an early pre-failure point with a late point.
EARLY_WINDOW = 30
LATE_WINDOW = 1


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
# STEP 2 — COLLECT 30-DAY AND 1-DAY OBSERVATIONS
# ============================================================

print()
print("Collecting individual HDD observations...")
print("----------------------------------------")

records = []

required_columns = [
    "date",
    "serial_number"
] + SMART_COLUMNS

for i, file in enumerate(files, start=1):

    print(
        f"Processing SMART data "
        f"{i}/{len(files)}: {file.name}"
    )

    # Check which columns actually exist.
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

    # Keep only HDDs that eventually failed.
    df = df[
        df["serial_number"].isin(failure_dates)
    ].copy()

    if df.empty:
        continue

    # Add each HDD's failure date.
    df["failure_date"] = df["serial_number"].map(
        failure_dates
    )

    # Calculate days before failure.
    df["days_before_failure"] = (
        df["failure_date"] - df["date"]
    ).dt.days

    # Keep only the two comparison windows.
    df = df[
        df["days_before_failure"].isin(
            [EARLY_WINDOW, LATE_WINDOW]
        )
    ].copy()

    if df.empty:
        continue

    keep_columns = [
        "serial_number",
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
        "No individual pre-failure observations found."
    )


observations = pd.concat(
    records,
    ignore_index=True
)


# ============================================================
# STEP 3 — CALCULATE INDIVIDUAL CHANGES
# ============================================================

print()
print("Calculating individual HDD changes...")
print("----------------------------------------")

results = []

for smart_column in SMART_COLUMNS:

    if smart_column not in observations.columns:
        continue

    values = observations[
        [
            "serial_number",
            "days_before_failure",
            smart_column
        ]
    ].copy()

    # Convert the two windows into separate columns.
    pivot = values.pivot_table(
        index="serial_number",
        columns="days_before_failure",
        values=smart_column,
        aggfunc="mean"
    )

    # Only HDDs with BOTH observations can be compared.
    if (
        EARLY_WINDOW not in pivot.columns
        or LATE_WINDOW not in pivot.columns
    ):
        continue

    pivot = pivot.dropna(
        subset=[EARLY_WINDOW, LATE_WINDOW]
    )

    if pivot.empty:
        continue

    pivot["change"] = (
        pivot[LATE_WINDOW]
        - pivot[EARLY_WINDOW]
    )

    pivot["direction"] = pivot["change"].apply(
        lambda value:
            "decreased"
            if value < 0
            else (
                "increased"
                if value > 0
                else "unchanged"
            )
    )

    total = len(pivot)

    decreased = int(
        (pivot["change"] < 0).sum()
    )

    increased = int(
        (pivot["change"] > 0).sum()
    )

    unchanged = int(
        (pivot["change"] == 0).sum()
    )

    results.append(
        {
            "smart_column": smart_column,
            "hdds_with_both_windows": total,
            "decreased_count": decreased,
            "decreased_percentage":
                decreased / total * 100,
            "increased_count": increased,
            "increased_percentage":
                increased / total * 100,
            "unchanged_count": unchanged,
            "unchanged_percentage":
                unchanged / total * 100,
            "mean_change":
                pivot["change"].mean(),
            "median_change":
                pivot["change"].median(),
            "std_change":
                pivot["change"].std(),
            "minimum_change":
                pivot["change"].min(),
            "maximum_change":
                pivot["change"].max(),
        }
    )


# ============================================================
# STEP 4 — SAVE RESULTS
# ============================================================

summary = pd.DataFrame(results)

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
print("INDIVIDUAL HDD DEGRADATION ANALYSIS")
print("========================================")

print(
    f"HDDs with comparable observations: "
    f"{len(observations['serial_number'].unique())}"
)

print()
print(
    f"Comparison: "
    f"{EARLY_WINDOW} days before failure "
    f"→ {LATE_WINDOW} day before failure"
)

print()
print(summary.to_string(index=False))

print()
print("Saved results to:")
print(OUTPUT_FILE)

print()
print("Analysis complete.")