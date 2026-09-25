from pathlib import Path
import pandas as pd


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned"
    / "ST12000NM0008_Q3_Q4"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "failure_event_validation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# LOAD METADATA
# ==========================================================

files = sorted(
    INPUT_DIR.glob("*.parquet")
)

if not files:
    raise FileNotFoundError(
        f"No Parquet files found in {INPUT_DIR}"
    )


parts = []

for i, file in enumerate(
    files,
    start=1
):

    print(
        f"[{i}/{len(files)}] "
        f"{file.name}"
    )

    temp = pd.read_parquet(
        file,
        columns=[
            "date",
            "serial_number",
            "model",
            "failure"
        ]
    )

    parts.append(temp)


df = pd.concat(
    parts,
    ignore_index=True
)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

df["failure"] = pd.to_numeric(
    df["failure"],
    errors="coerce"
).fillna(0).astype("int8")


df = df.sort_values(
    [
        "serial_number",
        "date"
    ]
)


# ==========================================================
# FAILURE COUNTS BY DRIVE
# ==========================================================

failure_counts = (
    df
    .groupby(
        "serial_number"
    )["failure"]
    .sum()
)


multiple_failure_serials = (
    failure_counts[
        failure_counts > 1
    ]
    .index
    .tolist()
)


print(
    "\nDrives with >1 failure row:",
    len(multiple_failure_serials)
)


# ==========================================================
# BUILD SUSPICIOUS DRIVE SUMMARY
# ==========================================================

summary_rows = []

detail_rows = []


for serial in multiple_failure_serials:

    drive = df[
        df["serial_number"] == serial
    ].copy()

    failure_rows = drive[
        drive["failure"] == 1
    ].copy()

    first_failure_date = (
        failure_rows["date"].min()
    )

    rows_after_first_failure = drive[
        drive["date"]
        > first_failure_date
    ]

    summary_rows.append(
        {
            "serial_number":
                serial,

            "model":
                drive["model"].iloc[0],

            "first_observed_date":
                drive["date"].min(),

            "last_observed_date":
                drive["date"].max(),

            "failure_row_count":
                len(failure_rows),

            "first_failure_date":
                first_failure_date,

            "last_failure_date":
                failure_rows["date"].max(),

            "rows_after_first_failure":
                len(rows_after_first_failure)
        }
    )


    # Keep context around each suspicious drive
    drive["is_after_first_failure"] = (
        drive["date"]
        > first_failure_date
    )

    detail_rows.append(
        drive
    )


summary_df = pd.DataFrame(
    summary_rows
)


if detail_rows:

    detail_df = pd.concat(
        detail_rows,
        ignore_index=True
    )

else:

    detail_df = pd.DataFrame()


# ==========================================================
# SAVE
# ==========================================================

summary_file = (
    OUTPUT_DIR
    / "multiple_failure_drive_summary.csv"
)

detail_file = (
    OUTPUT_DIR
    / "multiple_failure_drive_details.csv"
)


summary_df.to_csv(
    summary_file,
    index=False
)

detail_df.to_csv(
    detail_file,
    index=False
)


# ==========================================================
# PRINT
# ==========================================================

print("\n")
print("=" * 70)
print("FAILURE EVENT VALIDATION")
print("=" * 70)

print(
    "Total rows:",
    f"{len(df):,}"
)

print(
    "Unique HDDs:",
    f"{df['serial_number'].nunique():,}"
)

print(
    "Total failure rows:",
    f"{int(df['failure'].sum()):,}"
)

print(
    "Unique HDDs with failure:",
    f"{(failure_counts > 0).sum():,}"
)

print(
    "HDDs with multiple failure rows:",
    len(multiple_failure_serials)
)


if not summary_df.empty:

    print(
        "\nSuspicious HDDs:\n"
    )

    print(
        summary_df.to_string(
            index=False
        )
    )


print(
    "\nSummary saved to:"
)

print(
    summary_file
)

print(
    "\nDetails saved to:"
)

print(
    detail_file
)