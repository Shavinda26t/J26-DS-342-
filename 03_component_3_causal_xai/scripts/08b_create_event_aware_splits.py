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
    / "temporal"
    / "ST12000NM0008_Q3_Q4"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
    / "ST12000NM0008_Q3_Q4"
)

AUDIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "event_aware_splits"
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

TRAIN_MONTHS = [
    "2025-08",
    "2025-09",
]

VALIDATION_MONTHS = [
    "2025-10",
]

TEST_MONTHS = [
    "2025-11",
]


# ==========================================================
# LOAD MONTH FUNCTION
# ==========================================================

def load_month(month):

    file = (
        INPUT_DIR
        / f"{month}.parquet"
    )

    if not file.exists():

        raise FileNotFoundError(
            f"Missing file: {file}"
        )

    print(
        "Loading:",
        file.name
    )

    df = pd.read_parquet(
        file
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    return df


# ==========================================================
# LOAD AUGUST - NOVEMBER
# ==========================================================

month_data = {}

required_months = (
    TRAIN_MONTHS
    + VALIDATION_MONTHS
    + TEST_MONTHS
)

for month in required_months:

    month_data[month] = (
        load_month(month)
    )


# ==========================================================
# HELPER TO COMBINE MONTHS
# ==========================================================

def combine_months(months):

    return pd.concat(
        [
            month_data[m]
            for m in months
        ],
        ignore_index=True
    )


raw_train = combine_months(
    TRAIN_MONTHS
)

raw_validation = combine_months(
    VALIDATION_MONTHS
)

raw_test = combine_months(
    TEST_MONTHS
)


print("\nBase temporal periods:")

print(
    "Train:",
    raw_train["date"].min(),
    "to",
    raw_train["date"].max()
)

print(
    "Validation:",
    raw_validation["date"].min(),
    "to",
    raw_validation["date"].max()
)

print(
    "Test:",
    raw_test["date"].min(),
    "to",
    raw_test["date"].max()
)


# ==========================================================
# AUDIT STORAGE
# ==========================================================

summary_rows = []

exclusion_rows = []

overlap_rows = []


# ==========================================================
# PROCESS EACH HORIZON INDEPENDENTLY
# ==========================================================

for horizon in HORIZONS:

    print("\n")
    print("=" * 70)

    print(
        f"CREATING {horizon}-DAY SPLIT"
    )

    print("=" * 70)


    target = (
        f"failure_within_{horizon}d"
    )

    history_flag = (
        f"has_sufficient_{horizon}d_history"
    )


    # ======================================================
    # FILTER TO RESEARCH-ELIGIBLE ROWS
    # ======================================================

    train = raw_train[
        (raw_train["eligible_for_prediction"] == True)
        & (raw_train[history_flag] == True)
        & (raw_train[target].notna())
    ].copy()


    validation = raw_validation[
        (raw_validation["eligible_for_prediction"] == True)
        & (raw_validation[history_flag] == True)
        & (raw_validation[target].notna())
    ].copy()


    test = raw_test[
        (raw_test["eligible_for_prediction"] == True)
        & (raw_test[history_flag] == True)
        & (raw_test[target].notna())
    ].copy()


    # ======================================================
    # FIND FUTURE POSITIVE HDDs
    # ======================================================

    validation_positive_hdds = set(
        validation.loc[
            validation[target] == 1,
            "serial_number"
        ]
        .dropna()
        .astype(str)
    )


    test_positive_hdds = set(
        test.loc[
            test[target] == 1,
            "serial_number"
        ]
        .dropna()
        .astype(str)
    )


    # ======================================================
    # EVENT-AWARE EXCLUSION
    # ======================================================

    # Future validation failures must not appear in training.
    #
    # Future test failures must not appear in either
    # training or validation.

    train_excluded_hdds = (
        validation_positive_hdds
        | test_positive_hdds
    )


    validation_excluded_hdds = (
        test_positive_hdds
    )


    train_before = len(train)

    validation_before = len(
        validation
    )

    test_before = len(test)


    train = train[
        ~train[
            "serial_number"
        ].astype(str).isin(
            train_excluded_hdds
        )
    ].copy()


    validation = validation[
        ~validation[
            "serial_number"
        ].astype(str).isin(
            validation_excluded_hdds
        )
    ].copy()


    # Test remains untouched.
    #
    # It represents the final future evaluation period.


    # ======================================================
    # POSITIVE SETS AFTER FILTERING
    # ======================================================

    train_positive_hdds = set(
        train.loc[
            train[target] == 1,
            "serial_number"
        ]
        .dropna()
        .astype(str)
    )


    validation_positive_hdds_after = set(
        validation.loc[
            validation[target] == 1,
            "serial_number"
        ]
        .dropna()
        .astype(str)
    )


    test_positive_hdds_after = set(
        test.loc[
            test[target] == 1,
            "serial_number"
        ]
        .dropna()
        .astype(str)
    )


    # ======================================================
    # VERIFY POSITIVE EVENT OVERLAP
    # ======================================================

    train_val_overlap = (
        train_positive_hdds
        & validation_positive_hdds_after
    )


    train_test_overlap = (
        train_positive_hdds
        & test_positive_hdds_after
    )


    val_test_overlap = (
        validation_positive_hdds_after
        & test_positive_hdds_after
    )


    # ======================================================
    # SAVE HORIZON DATA
    # ======================================================

    horizon_dir = (
        OUTPUT_DIR
        / f"{horizon}d"
    )

    horizon_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    train_file = (
        horizon_dir
        / "train.parquet"
    )

    validation_file = (
        horizon_dir
        / "validation.parquet"
    )

    test_file = (
        horizon_dir
        / "test.parquet"
    )


    train.to_parquet(
        train_file,
        index=False,
        compression="zstd"
    )

    validation.to_parquet(
        validation_file,
        index=False,
        compression="zstd"
    )

    test.to_parquet(
        test_file,
        index=False,
        compression="zstd"
    )


    # ======================================================
    # SAVE EXCLUDED HDD LIST
    # ======================================================

    for serial in sorted(
        train_excluded_hdds
    ):

        if (
            serial
            in test_positive_hdds
        ):

            reason = (
                "Future positive HDD in test period"
            )

        else:

            reason = (
                "Future positive HDD in validation period"
            )


        exclusion_rows.append(
            {
                "horizon_days":
                    horizon,

                "excluded_from":
                    "train",

                "serial_number":
                    serial,

                "reason":
                    reason
            }
        )


    for serial in sorted(
        validation_excluded_hdds
    ):

        exclusion_rows.append(
            {
                "horizon_days":
                    horizon,

                "excluded_from":
                    "validation",

                "serial_number":
                    serial,

                "reason":
                    "Future positive HDD in test period"
            }
        )


    # ======================================================
    # SPLIT STATISTICS
    # ======================================================

    split_objects = {
        "train": train,
        "validation": validation,
        "test": test,
    }


    rows_before = {
        "train": train_before,
        "validation": validation_before,
        "test": test_before,
    }


    for split_name, split_df in (
        split_objects.items()
    ):

        positive_rows = int(
            (
                split_df[target]
                == 1
            ).sum()
        )

        negative_rows = int(
            (
                split_df[target]
                == 0
            ).sum()
        )

        usable_rows = (
            positive_rows
            + negative_rows
        )


        summary_rows.append(
            {
                "horizon_days":
                    horizon,

                "split":
                    split_name,

                "rows_before_event_filter":
                    rows_before[
                        split_name
                    ],

                "rows_after_event_filter":
                    len(
                        split_df
                    ),

                "rows_removed":
                    (
                        rows_before[
                            split_name
                        ]
                        - len(
                            split_df
                        )
                    ),

                "positive_rows":
                    positive_rows,

                "negative_rows":
                    negative_rows,

                "positive_rate_pct":
                    round(
                        (
                            positive_rows
                            / usable_rows
                            * 100
                        )
                        if usable_rows > 0
                        else 0,
                        6
                    ),

                "unique_hdds":
                    split_df[
                        "serial_number"
                    ].nunique(),

                "unique_positive_hdds":
                    split_df.loc[
                        split_df[
                            target
                        ] == 1,
                        "serial_number"
                    ].nunique(),

                "first_date":
                    split_df[
                        "date"
                    ].min(),

                "last_date":
                    split_df[
                        "date"
                    ].max(),
            }
        )


    # ======================================================
    # OVERLAP AUDIT
    # ======================================================

    overlap_rows.append(
        {
            "horizon_days":
                horizon,

            "train_validation_shared_positive_hdds":
                len(
                    train_val_overlap
                ),

            "train_test_shared_positive_hdds":
                len(
                    train_test_overlap
                ),

            "validation_test_shared_positive_hdds":
                len(
                    val_test_overlap
                ),
        }
    )


    print(
        f"\n{horizon}d:"
    )

    print(
        "Train rows:",
        f"{len(train):,}",
        "| positive HDDs:",
        len(
            train_positive_hdds
        )
    )

    print(
        "Validation rows:",
        f"{len(validation):,}",
        "| positive HDDs:",
        len(
            validation_positive_hdds_after
        )
    )

    print(
        "Test rows:",
        f"{len(test):,}",
        "| positive HDDs:",
        len(
            test_positive_hdds_after
        )
    )


    print(
        "Positive-event overlap "
        "(train-val / train-test / val-test):",
        len(train_val_overlap),
        "/",
        len(train_test_overlap),
        "/",
        len(val_test_overlap)
    )


# ==========================================================
# SAVE AUDITS
# ==========================================================

summary_df = pd.DataFrame(
    summary_rows
)

overlap_df = pd.DataFrame(
    overlap_rows
)

exclusion_df = pd.DataFrame(
    exclusion_rows
)


summary_file = (
    AUDIT_DIR
    / "event_aware_split_summary.csv"
)

overlap_file = (
    AUDIT_DIR
    / "event_aware_overlap_check.csv"
)

exclusion_file = (
    AUDIT_DIR
    / "event_aware_exclusions.csv"
)


summary_df.to_csv(
    summary_file,
    index=False
)

overlap_df.to_csv(
    overlap_file,
    index=False
)

exclusion_df.to_csv(
    exclusion_file,
    index=False
)


# ==========================================================
# FINAL OUTPUT
# ==========================================================

print("\n")
print("=" * 70)

print(
    "EVENT-AWARE TEMPORAL SPLIT COMPLETE"
)

print("=" * 70)


print(
    "\nSPLIT SUMMARY:\n"
)

print(
    summary_df.to_string(
        index=False
    )
)


print(
    "\nPOSITIVE-EVENT OVERLAP CHECK:\n"
)

print(
    overlap_df.to_string(
        index=False
    )
)


print(
    "\nSplit datasets saved to:"
)

print(
    OUTPUT_DIR
)


print(
    "\nAudit files saved to:"
)

print(
    AUDIT_DIR
)