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
    / "labeled"
    / "ST12000NM0008_Q3_Q4"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "labeled_clean"
    / "ST12000NM0008_Q3_Q4"
)

AUDIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "anomaly_removal"
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

EXCLUDED_SERIALS = {
    "ZL005DRH"
}

HORIZONS = [7, 14, 30]


# ==========================================================
# FIND INPUT FILES
# ==========================================================

files = sorted(
    INPUT_DIR.glob("*.parquet")
)

if not files:
    raise FileNotFoundError(
        f"No labeled Parquet files found in {INPUT_DIR}"
    )


# ==========================================================
# COUNTERS
# ==========================================================

total_input_rows = 0
total_removed_rows = 0
total_output_rows = 0

failure_rows_before = 0
failure_rows_after = 0

unique_serials_before = set()
unique_serials_after = set()

failed_serials_before = set()
failed_serials_after = set()

label_summary = {
    horizon: {
        "positive": 0,
        "negative": 0,
        "censored": 0
    }
    for horizon in HORIZONS
}


# ==========================================================
# PROCESS FILES
# ==========================================================

for i, file in enumerate(
    files,
    start=1
):

    print(
        f"[{i}/{len(files)}] "
        f"{file.name}"
    )

    df = pd.read_parquet(
        file
    )

    total_input_rows += len(df)

    # ------------------------------------------------------
    # BEFORE FILTER COUNTS
    # ------------------------------------------------------

    unique_serials_before.update(
        df["serial_number"]
        .dropna()
        .astype(str)
    )

    failed_before = df.loc[
        df["failure"] == 1,
        "serial_number"
    ].dropna().astype(str)

    failed_serials_before.update(
        failed_before
    )

    failure_rows_before += int(
        (df["failure"] == 1).sum()
    )


    # ------------------------------------------------------
    # REMOVE ANOMALOUS SERIALS
    # ------------------------------------------------------

    remove_mask = (
        df["serial_number"]
        .isin(EXCLUDED_SERIALS)
    )

    removed = int(
        remove_mask.sum()
    )

    total_removed_rows += removed

    df = df.loc[
        ~remove_mask
    ].copy()


    # ------------------------------------------------------
    # AFTER FILTER COUNTS
    # ------------------------------------------------------

    unique_serials_after.update(
        df["serial_number"]
        .dropna()
        .astype(str)
    )

    failed_after = df.loc[
        df["failure"] == 1,
        "serial_number"
    ].dropna().astype(str)

    failed_serials_after.update(
        failed_after
    )

    failure_rows_after += int(
        (df["failure"] == 1).sum()
    )


    # ------------------------------------------------------
    # LABEL COUNTS
    # ------------------------------------------------------

    for horizon in HORIZONS:

        col = (
            f"failure_within_{horizon}d"
        )

        label_summary[
            horizon
        ]["positive"] += int(
            (df[col] == 1).sum()
        )

        label_summary[
            horizon
        ]["negative"] += int(
            (df[col] == 0).sum()
        )

        label_summary[
            horizon
        ]["censored"] += int(
            df[col].isna().sum()
        )


    # ------------------------------------------------------
    # SAVE FILTERED FILE
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
# LABEL SUMMARY
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


summary_df = pd.DataFrame(
    summary_rows
)


# ==========================================================
# EXCLUSION AUDIT
# ==========================================================

exclusion_df = pd.DataFrame(
    [
        {
            "serial_number":
                serial,

            "reason":
                (
                    "Multiple recorded failure events "
                    "and post-first-failure observation"
                )
        }
        for serial in EXCLUDED_SERIALS
    ]
)


exclusion_file = (
    AUDIT_DIR
    / "excluded_drives.csv"
)

summary_file = (
    AUDIT_DIR
    / "filtered_failure_horizon_summary.csv"
)

exclusion_df.to_csv(
    exclusion_file,
    index=False
)

summary_df.to_csv(
    summary_file,
    index=False
)


# ==========================================================
# FINAL OUTPUT
# ==========================================================

print("\n")
print("=" * 70)
print("ANOMALOUS DRIVE REMOVAL COMPLETE")
print("=" * 70)

print(
    "Input rows:",
    f"{total_input_rows:,}"
)

print(
    "Rows removed:",
    f"{total_removed_rows:,}"
)

print(
    "Final rows:",
    f"{total_output_rows:,}"
)

print(
    "\nUnique HDDs before:",
    f"{len(unique_serials_before):,}"
)

print(
    "Unique HDDs after:",
    f"{len(unique_serials_after):,}"
)

print(
    "\nFailure rows before:",
    f"{failure_rows_before:,}"
)

print(
    "Failure rows after:",
    f"{failure_rows_after:,}"
)

print(
    "\nUnique failed HDDs before:",
    f"{len(failed_serials_before):,}"
)

print(
    "Unique failed HDDs after:",
    f"{len(failed_serials_after):,}"
)

print(
    "\nFiltered label summary:\n"
)

print(
    summary_df.to_string(
        index=False
    )
)

print(
    "\nFiltered data saved to:"
)

print(
    OUTPUT_DIR
)

print(
    "\nExclusion record saved to:"
)

print(
    exclusion_file
)