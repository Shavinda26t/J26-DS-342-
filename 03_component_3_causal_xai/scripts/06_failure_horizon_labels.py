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
    / "ST12000NM0008"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "labeled"
    / "ST12000NM0008"
)

AUDIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "failure_horizon_labels"
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
    "Parquet files found:",
    len(parquet_files)
)


# ==========================================================
# STEP 1:
# LOAD ONLY DRIVE / DATE / FAILURE INFORMATION
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
# STEP 2:
# BUILD PER-DRIVE OBSERVATION SUMMARY
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
# IDENTIFY FAILURE DATE
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
# SAVE DRIVE SUMMARY
# ==========================================================

drive_summary_file = (
    AUDIT_DIR
    / "drive_followup_summary.csv"
)

drive_summary.reset_index().to_csv(
    drive_summary_file,
    index=False
)


print(
    "\nObserved failed HDDs:",
    int(
        drive_summary[
            "has_observed_failure"
        ].sum()
    )
)


# ==========================================================
# STEP 3:
# GENERATE LABELS FILE BY FILE
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


summary_for_merge = (
    drive_summary[
        [
            "last_observed_date",
            "failure_date"
        ]
    ]
    .reset_index()
)


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

    df = df.merge(
        summary_for_merge,
        on="serial_number",
        how="left",
        validate="many_to_one"
    )


    # ======================================================
    # DAYS UNTIL OBSERVED FAILURE
    # ======================================================

    df[
        "days_to_failure"
    ] = (
        df["failure_date"]
        - df["date"]
    ).dt.days


    # ======================================================
    # AVAILABLE FUTURE FOLLOW-UP
    # ======================================================

    df[
        "future_followup_days"
    ] = (
        df["last_observed_date"]
        - df["date"]
    ).dt.days


    # ======================================================
    # IDENTIFY FAILURE-DAY / POST-FAILURE ROWS
    # ======================================================

    is_failure_day = (
        df["failure"] == 1
    )

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


    # A row can be used for prediction only before failure.
    df[
        "eligible_for_prediction"
    ] = (
        ~is_failure_day
        & ~is_post_failure
    )


    # ======================================================
    # CREATE EACH HORIZON LABEL
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
        # POSITIVE:
        # failure occurs in next 1..H days
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
        # NEGATIVE CASE 1:
        # an observed failure exists,
        # but it is farther than H days away
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
        # NEGATIVE CASE 2:
        # no observed failure,
        # but enough future follow-up exists
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
        # FAILURE DAY ITSELF:
        # deliberately not used for prediction
        # --------------------------------------------------

        label.loc[
            is_failure_day
        ] = pd.NA


        # --------------------------------------------------
        # POST-FAILURE:
        # also not valid
        # --------------------------------------------------

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
            (
                label == 1
            ).sum()
        )

        label_summary[
            horizon
        ]["negative"] += int(
            (
                label == 0
            ).sum()
        )

        label_summary[
            horizon
        ]["censored"] += int(
            label.isna().sum()
        )


    # ======================================================
    # REMOVE TEMPORARY MERGE FIELDS THAT ARE NOT NEEDED
    # ======================================================

    df = df.drop(
        columns=[
            "last_observed_date"
        ]
    )


    # ======================================================
    # SORT
    # ======================================================

    df = df.sort_values(
        [
            "serial_number",
            "date"
        ]
    )


    # ======================================================
    # SAVE DAILY LABELED FILE
    # ======================================================

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
# STEP 4:
# CREATE LABEL SUMMARY
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
    / "failure_horizon_summary.csv"
)

label_summary_df.to_csv(
    label_summary_file,
    index=False
)


# ==========================================================
# PRINT FINAL RESULTS
# ==========================================================

print("\n")
print("=" * 70)
print("FAILURE HORIZON LABELING COMPLETE")
print("=" * 70)

print(
    "Total labeled dataset rows:",
    f"{total_output_rows:,}"
)

print(
    "Failure-day rows excluded:",
    f"{failure_day_rows:,}"
)

print(
    "Post-failure rows found:",
    f"{post_failure_rows:,}"
)

print("\nLabel Summary:\n")

print(
    label_summary_df.to_string(
        index=False
    )
)

print("\nDrive summary saved to:")

print(
    drive_summary_file
)

print("\nLabel summary saved to:")

print(
    label_summary_file
)

print("\nLabeled Parquet files saved to:")

print(
    OUTPUT_DIR
)