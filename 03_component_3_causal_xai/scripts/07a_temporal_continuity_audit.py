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
    / "labeled_clean"
    / "ST12000NM0008_Q3_Q4"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "temporal_continuity"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# FIND FILES
# ==========================================================

files = sorted(
    INPUT_DIR.glob("*.parquet")
)

if not files:
    raise FileNotFoundError(
        f"No Parquet files found in {INPUT_DIR}"
    )

print(
    "Parquet files found:",
    len(files)
)


# ==========================================================
# LOAD ONLY SERIAL + DATE
# ==========================================================

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
            "serial_number",
            "date"
        ]
    )

    parts.append(temp)


df = pd.concat(
    parts,
    ignore_index=True
)

del parts


# ==========================================================
# PREPARE
# ==========================================================

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "serial_number",
        "date"
    ]
)

df = df.sort_values(
    [
        "serial_number",
        "date"
    ]
).reset_index(drop=True)


# ==========================================================
# DAYS BETWEEN CONSECUTIVE RECORDS
# ==========================================================

df[
    "days_since_previous_record"
] = (
    df
    .groupby(
        "serial_number"
    )["date"]
    .diff()
    .dt.days
)


transitions = df[
    df[
        "days_since_previous_record"
    ].notna()
].copy()


# ==========================================================
# BASIC COUNTS
# ==========================================================

total_rows = len(df)

unique_drives = (
    df["serial_number"]
    .nunique()
)

total_transitions = len(
    transitions
)

daily_transitions = int(
    (
        transitions[
            "days_since_previous_record"
        ] == 1
    ).sum()
)

gap_transitions = int(
    (
        transitions[
            "days_since_previous_record"
        ] > 1
    ).sum()
)

same_day_transitions = int(
    (
        transitions[
            "days_since_previous_record"
        ] == 0
    ).sum()
)

negative_transitions = int(
    (
        transitions[
            "days_since_previous_record"
        ] < 0
    ).sum()
)


daily_pct = (
    daily_transitions
    / total_transitions
    * 100
    if total_transitions > 0
    else 0
)

gap_pct = (
    gap_transitions
    / total_transitions
    * 100
    if total_transitions > 0
    else 0
)


# ==========================================================
# GAP DISTRIBUTION
# ==========================================================

gap_distribution = (
    transitions[
        "days_since_previous_record"
    ]
    .value_counts()
    .sort_index()
    .rename_axis(
        "gap_days"
    )
    .reset_index(
        name="count"
    )
)

gap_distribution[
    "percentage"
] = (
    gap_distribution["count"]
    / total_transitions
    * 100
)


# ==========================================================
# DRIVE-LEVEL GAP SUMMARY
# ==========================================================

drive_gap_summary = (
    transitions
    .groupby(
        "serial_number"
    )
    .agg(
        transitions=(
            "days_since_previous_record",
            "count"
        ),

        max_gap_days=(
            "days_since_previous_record",
            "max"
        ),

        mean_gap_days=(
            "days_since_previous_record",
            "mean"
        )
    )
    .reset_index()
)


gap_counts = (
    transitions.loc[
        transitions[
            "days_since_previous_record"
        ] > 1
    ]
    .groupby(
        "serial_number"
    )
    .size()
    .rename(
        "gap_count"
    )
)


drive_gap_summary = (
    drive_gap_summary
    .merge(
        gap_counts,
        on="serial_number",
        how="left"
    )
)

drive_gap_summary[
    "gap_count"
] = (
    drive_gap_summary[
        "gap_count"
    ]
    .fillna(0)
    .astype(int)
)


drives_with_gaps = (
    drive_gap_summary[
        drive_gap_summary[
            "gap_count"
        ] > 0
    ]
    .copy()
)


# ==========================================================
# SUMMARY TABLE
# ==========================================================

max_gap = (
    transitions[
        "days_since_previous_record"
    ].max()
)

summary_df = pd.DataFrame(
    [
        {
            "total_rows":
                total_rows,

            "unique_drives":
                unique_drives,

            "total_transitions":
                total_transitions,

            "one_day_transitions":
                daily_transitions,

            "one_day_transition_pct":
                round(
                    daily_pct,
                    6
                ),

            "gap_gt_1_day_transitions":
                gap_transitions,

            "gap_gt_1_day_pct":
                round(
                    gap_pct,
                    6
                ),

            "same_day_transitions":
                same_day_transitions,

            "negative_transitions":
                negative_transitions,

            "drives_with_gaps":
                len(
                    drives_with_gaps
                ),

            "max_gap_days":
                max_gap
        }
    ]
)


# ==========================================================
# SAVE RESULTS
# ==========================================================

summary_file = (
    OUTPUT_DIR
    / "temporal_continuity_summary.csv"
)

distribution_file = (
    OUTPUT_DIR
    / "gap_distribution.csv"
)

drive_gap_file = (
    OUTPUT_DIR
    / "drives_with_gaps.csv"
)


summary_df.to_csv(
    summary_file,
    index=False
)

gap_distribution.to_csv(
    distribution_file,
    index=False
)

drives_with_gaps.to_csv(
    drive_gap_file,
    index=False
)


# ==========================================================
# PRINT
# ==========================================================

print("\n")
print("=" * 70)
print("TEMPORAL CONTINUITY AUDIT")
print("=" * 70)

print(
    "Total rows:",
    f"{total_rows:,}"
)

print(
    "Unique HDDs:",
    f"{unique_drives:,}"
)

print(
    "Total consecutive-record transitions:",
    f"{total_transitions:,}"
)

print(
    "Exactly 1-day transitions:",
    f"{daily_transitions:,}"
)

print(
    "1-day transition percentage:",
    f"{daily_pct:.6f}%"
)

print(
    "Transitions with gap > 1 day:",
    f"{gap_transitions:,}"
)

print(
    "Gap > 1 day percentage:",
    f"{gap_pct:.6f}%"
)

print(
    "Drives containing at least one gap:",
    f"{len(drives_with_gaps):,}"
)

print(
    "Maximum observed gap:",
    max_gap,
    "days"
)

print(
    "Same-day transitions:",
    f"{same_day_transitions:,}"
)

print(
    "Negative transitions:",
    f"{negative_transitions:,}"
)

print(
    "\nMost common gap lengths:\n"
)

print(
    gap_distribution
    .head(15)
    .to_string(
        index=False
    )
)

print(
    "\nSummary saved to:"
)

print(
    summary_file
)