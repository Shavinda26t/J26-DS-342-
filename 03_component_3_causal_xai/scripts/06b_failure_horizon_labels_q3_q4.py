from pathlib import Path
import pandas as pd


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLEANED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned"
    / "ST12000NM0008_Q3_Q4"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "labeled"
    / "ST12000NM0008_Q3_Q4"
)

AUDIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "failure_horizon_labels_q3_q4"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

AUDIT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# SETTINGS
# ==========================================================

HORIZONS = [7, 14, 30]


# ==========================================================
# FIND CLEANED FILES
# ==========================================================

parquet_files = sorted(
    CLEANED_DIR.glob("*.parquet")
)

if not parquet_files:
    raise FileNotFoundError(
        f"No Parquet files found in {CLEANED_DIR}"
    )

print(
    "Cleaned Parquet files found:",
    len(parquet_files)
)


# ==========================================================
# LOAD ONLY DRIVE / DATE / FAILURE METADATA
# ==========================================================

metadata_parts = []

for i, file in enumerate(
    parquet_files,
    start=1
):

    print(
        f"[Metadata {i}/{len(parquet_files)}] "
        f"{file.name}"
    )

    temp = pd.read_parquet(
        file,
        columns=[
            "serial_number",
            "date",
            "failure"
        ]
    )

    metadata_parts.append(temp)


metadata = pd.concat(
    metadata_parts,
    ignore_index=True
)

metadata["date"] = pd.to_datetime(
    metadata["date"],
    errors="coerce"
)

metadata["failure"] = pd.to_numeric(
    metadata["failure"],
    errors="coerce"
).fillna(0).astype("int8")


print(
    "\nMetadata rows:",
    f"{len(metadata):,}"
)

print(
    "Unique HDDs:",
    f"{metadata['serial_number'].nunique():,}"
)


# ==========================================================
# BUILD DRIVE SUMMARY
# ==========================================================

drive_summary = (
    metadata
    .groupby(
        "serial_number",
        as_index=True
    )
    .agg(
        first_observed_date=(
            "date",
            "min"
        ),

        last_observed_date=(
            "date",
            "max"
        ),

        observed_days=(
            "date",
            "nunique"
        ),

        failure_rows=(
            "failure",
            "sum"
        )
    )
)


# ==========================================================
# IDENTIFY FIRST FAILURE DATE
# ==========================================================

failure_dates = (
    metadata.loc[
        metadata["failure"] == 1
    ]
    .groupby(
        "serial_number"
    )["date"]
    .min()
    .rename(
        "failure_date"
    )
)

drive_summary = (
    drive_summary
    .join(
        failure_dates
    )
)

drive_summary[
    "has_observed_failure"
] = (
    drive_summary[
        "failure_date"
    ].notna()
)


# ==========================================================
# FAILURE MONTH
# ==========================================================

drive_summary[
    "failure_month"
] = (
    drive_summary[
        "failure_date"
    ]
    .dt
    .to_period("M")
    .astype("string")
)


# ==========================================================
# SAVE DRIVE FOLLOW-UP SUMMARY
# ==========================================================

drive_summary_file = (
    AUDIT_DIR
    / "drive_followup_summary_q3_q4.csv"
)

drive_summary.reset_index().to_csv(
    drive_summary_file,
    index=False
)


# ==========================================================
# PREPARE SUMMARY FOR MERGE
# ==========================================================

summary_for_merge = (
    drive_summary[
        [
            "last_observed_date",
            "failure_date"
        ]
    ]
    .reset_index()
)


# ==========================================================
# LABEL COUNTERS
# ==========================================================

label_summary = {
    horizon: {
        "positive": 0,
        "negative": 0,
        "censored": 0
    }
    for horizon in HORIZONS
}

failure_day_rows = 0
post_failure_rows = 0
total_output_rows = 0


# ==========================================================
# PROCESS EACH DAILY FILE
# ==========================================================

for i, file in enumerate(
    parquet_files,
    start=1
):

    print(
        f"[Labeling {i}/{len(parquet_files)}] "
        f"{file.name}"
    )

    df = pd.read_parquet(
        file
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )


    # ------------------------------------------------------
    # MERGE DRIVE FOLLOW-UP INFORMATION
    # ------------------------------------------------------

    df = df.merge(
        summary_for_merge,
        on="serial_number",
        how="left",
        validate="many_to_one"
    )


    # ------------------------------------------------------
    # CALENDAR DAYS UNTIL OBSERVED FAILURE
    # ------------------------------------------------------

    df[
        "days_to_failure"
    ] = (
        df["failure_date"]
        - df["date"]
    ).dt.days


    # ------------------------------------------------------
    # AVAILABLE FUTURE FOLLOW-UP
    # ------------------------------------------------------

    df[
        "future_followup_days"
    ] = (
        df["last_observed_date"]
        - df["date"]
    ).dt.days


    # ------------------------------------------------------
    # FAILURE DAY
    # ------------------------------------------------------

    is_failure_day = (
        df["failure"] == 1
    )


    # ------------------------------------------------------
    # POST-FAILURE ROW
    # ------------------------------------------------------

    is_post_failure = (
        df["failure_date"].notna()
        & (
            df["date"]
            > df["failure_date"]
        )
    )


    failure_day_rows += int(
        is_failure_day.sum()
    )

    post_failure_rows += int(
        is_post_failure.sum()
    )


    # ------------------------------------------------------
    # PREDICTION-ELIGIBLE ROW
    # ------------------------------------------------------

    df[
        "eligible_for_prediction"
    ] = (
        ~is_failure_day
        & ~is_post_failure
    )


    # ======================================================
    # CREATE HORIZON LABELS
    # ======================================================

    for horizon in HORIZONS:

        label_name = (
            f"failure_within_{horizon}d"
        )

        label = pd.Series(
            pd.NA,
            index=df.index,
            dtype="Int8"
        )


        # --------------------------------------------------
        # POSITIVE
        # Failure occurs in next 1..H calendar days
        # --------------------------------------------------

        positive = (
            df[
                "eligible_for_prediction"
            ]
            & df[
                "failure_date"
            ].notna()
            & df[
                "days_to_failure"
            ].between(
                1,
                horizon
            )
        )

        label.loc[
            positive
        ] = 1


        # --------------------------------------------------
        # NEGATIVE TYPE 1
        # Observed failure exists, but farther than horizon
        # --------------------------------------------------

        negative_future_failure = (
            df[
                "eligible_for_prediction"
            ]
            & df[
                "failure_date"
            ].notna()
            & (
                df[
                    "days_to_failure"
                ]
                > horizon
            )
        )

        label.loc[
            negative_future_failure
        ] = 0


        # --------------------------------------------------
        # NEGATIVE TYPE 2
        # No observed failure and enough future follow-up
        # --------------------------------------------------

        negative_no_failure = (
            df[
                "eligible_for_prediction"
            ]
            & df[
                "failure_date"
            ].isna()
            & (
                df[
                    "future_followup_days"
                ]
                >= horizon
            )
        )

        label.loc[
            negative_no_failure
        ] = 0


        # --------------------------------------------------
        # FAILURE DAY / POST-FAILURE ARE NOT PREDICTION ROWS
        # --------------------------------------------------

        label.loc[
            is_failure_day
        ] = pd.NA

        label.loc[
            is_post_failure
        ] = pd.NA


        df[
            label_name
        ] = label


        # --------------------------------------------------
        # COUNTS
        # --------------------------------------------------

        label_summary[
            horizon
        ]["positive"] += int(
            (label == 1).sum()
        )

        label_summary[
            horizon
        ]["negative"] += int(
            (label == 0).sum()
        )

        label_summary[
            horizon
        ]["censored"] += int(
            label.isna().sum()
        )


    # ------------------------------------------------------
    # REMOVE TEMPORARY COLUMN
    # ------------------------------------------------------

    df = df.drop(
        columns=[
            "last_observed_date"
        ]
    )


    # ------------------------------------------------------
    # SORT
    # ------------------------------------------------------

    df = df.sort_values(
        [
            "serial_number",
            "date"
        ]
    )


    # ------------------------------------------------------
    # SAVE
    # ------------------------------------------------------

    output_file = (
        OUTPUT_DIR
        / file.name
    )

    df.to_parquet(
        output_file,
        index=False
    )

    total_output_rows += len(df)


# ==========================================================
# CREATE HORIZON SUMMARY
# ==========================================================

summary_rows = []

for horizon in HORIZONS:

    counts = (
        label_summary[
            horizon
        ]
    )

    usable = (
        counts["positive"]
        + counts["negative"]
    )

    positive_rate = (
        counts["positive"]
        / usable
        * 100
        if usable > 0
        else 0
    )

    summary_rows.append(
        {
            "horizon_days":
                horizon,

            "positive_rows":
                counts["positive"],

            "negative_rows":
                counts["negative"],

            "censored_rows":
                counts["censored"],

            "usable_rows":
                usable,

            "positive_rate_pct":
                round(
                    positive_rate,
                    6
                )
        }
    )


label_summary_df = pd.DataFrame(
    summary_rows
)

label_summary_file = (
    AUDIT_DIR
    / "failure_horizon_summary_q3_q4.csv"
)

label_summary_df.to_csv(
    label_summary_file,
    index=False
)


# ==========================================================
# FAILURE DISTRIBUTION BY MONTH
# ==========================================================

failure_month_summary = (
    drive_summary[
        drive_summary[
            "has_observed_failure"
        ]
    ]
    .groupby(
        "failure_month",
        dropna=True
    )
    .size()
    .reset_index(
        name="failed_drives"
    )
)

failure_month_file = (
    AUDIT_DIR
    / "failure_month_summary_q3_q4.csv"
)

failure_month_summary.to_csv(
    failure_month_file,
    index=False
)


# ==========================================================
# FINAL OUTPUT
# ==========================================================

print("\n")
print("=" * 70)
print("Q3 + Q4 FAILURE HORIZON LABELING COMPLETE")
print("=" * 70)

print(
    "Total dataset rows:",
    f"{total_output_rows:,}"
)

print(
    "Unique HDDs:",
    f"{len(drive_summary):,}"
)

print(
    "Observed failed HDDs:",
    f"{int(drive_summary['has_observed_failure'].sum()):,}"
)

print(
    "Failure-day rows excluded:",
    f"{failure_day_rows:,}"
)

print(
    "Post-failure rows found:",
    f"{post_failure_rows:,}"
)

print(
    "\nLabel Summary:\n"
)

print(
    label_summary_df.to_string(
        index=False
    )
)

print(
    "\nFailures by month:\n"
)

print(
    failure_month_summary.to_string(
        index=False
    )
)

print(
    "\nLabeled data saved to:"
)

print(
    OUTPUT_DIR
)

print(
    "\nHorizon summary saved to:"
)

print(
    label_summary_file
)