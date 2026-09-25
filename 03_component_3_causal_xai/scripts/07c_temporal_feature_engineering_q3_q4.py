from pathlib import Path

import numpy as np
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
    / "processed"
    / "temporal"
    / "ST12000NM0008_Q3_Q4"
)

AUDIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "temporal_feature_audit_q3_q4"
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

SMART_FEATURES = [
    "smart_1_raw",
    "smart_4_raw",
    "smart_5_raw",
    "smart_7_raw",
    "smart_9_raw",
    "smart_12_raw",
    "smart_187_raw",
    "smart_188_raw",
    "smart_192_raw",
    "smart_193_raw",
    "smart_194_raw",
    "smart_197_raw",
    "smart_199_raw",
    "smart_240_raw",
    "smart_241_raw",
    "smart_242_raw",
]

WINDOWS = [
    7,
    14,
    30
]

TARGET_COLUMNS = [
    "failure_within_7d",
    "failure_within_14d",
    "failure_within_30d",
]


# ==========================================================
# FIND DAILY FILES
# ==========================================================

files = sorted(
    INPUT_DIR.glob("*.parquet")
)

if not files:
    raise FileNotFoundError(
        f"No Parquet files found in {INPUT_DIR}"
    )


# ==========================================================
# MAP DATE -> FILE
# ==========================================================

file_map = {}

for file in files:

    try:

        file_date = pd.Timestamp(
            file.stem
        )

        file_map[file_date] = file

    except ValueError:

        print(
            "Skipping file with unexpected name:",
            file.name
        )


all_dates = sorted(
    file_map.keys()
)

print(
    "Daily files found:",
    len(all_dates)
)

print(
    "Date range:",
    min(all_dates),
    "to",
    max(all_dates)
)


# ==========================================================
# CHECK AVAILABLE COLUMNS
# ==========================================================

first_file = file_map[
    min(all_dates)
]

available_columns = (
    pd.read_parquet(
        first_file
    )
    .columns
    .tolist()
)


metadata_candidates = [
    "date",
    "serial_number",
    "model",
    "capacity_bytes",
    "failure",

    "datacenter",
    "cluster_id",
    "vault_id",
    "pod_id",
    "pod_slot_num",
    "is_legacy_format",

    "failure_date",
    "days_to_failure",
    "future_followup_days",
    "eligible_for_prediction",
]


metadata_columns = [
    col
    for col in metadata_candidates
    if col in available_columns
]


required_columns = (
    metadata_columns
    + TARGET_COLUMNS
    + SMART_FEATURES
)


# ==========================================================
# MONTHS TO PROCESS
# ==========================================================

months = pd.period_range(
    start=min(all_dates).to_period("M"),
    end=max(all_dates).to_period("M"),
    freq="M"
)


# ==========================================================
# AUDIT STORAGE
# ==========================================================

month_summaries = []

feature_missing = {}
feature_total = {}


# ==========================================================
# PROCESS ONE MONTH AT A TIME
# ==========================================================

for month in months:

    print("\n")
    print("=" * 70)

    print(
        f"PROCESSING MONTH: {month}"
    )

    print("=" * 70)


    # ------------------------------------------------------
    # MONTH START / END
    # ------------------------------------------------------

    month_start = (
        month.start_time.normalize()
    )

    month_end = (
        month.end_time.normalize()
    )


    # ------------------------------------------------------
    # LOAD 30 DAYS OF HISTORY BEFORE MONTH START
    # ------------------------------------------------------

    context_start = (
        month_start
        - pd.Timedelta(days=30)
    )


    selected_dates = [
        date
        for date in all_dates
        if (
            context_start
            <= date
            <= month_end
        )
    ]


    if not selected_dates:
        continue


    print(
        "Context:",
        min(selected_dates),
        "to",
        max(selected_dates)
    )


    # ------------------------------------------------------
    # LOAD CONTEXT DATA
    # ------------------------------------------------------

    parts = []

    for date in selected_dates:

        file = file_map[
            date
        ]

        temp = pd.read_parquet(
            file,
            columns=required_columns
        )

        parts.append(
            temp
        )


    work = pd.concat(
        parts,
        ignore_index=True
    )

    del parts


    # ------------------------------------------------------
    # PREPARE DATE
    # ------------------------------------------------------

    work["date"] = pd.to_datetime(
        work["date"],
        errors="coerce"
    )


    work = work.sort_values(
        [
            "serial_number",
            "date"
        ]
    ).reset_index(
        drop=True
    )


    # ------------------------------------------------------
    # NUMERIC SMART VALUES
    # ------------------------------------------------------

    for feature in SMART_FEATURES:

        work[feature] = pd.to_numeric(
            work[feature],
            errors="coerce"
        ).astype(
            "float32"
        )


    # ------------------------------------------------------
    # CREATE UNIQUE SERIAL-DATE INDEX
    # ------------------------------------------------------

    work_keys = pd.MultiIndex.from_frame(
        work[
            [
                "serial_number",
                "date"
            ]
        ]
    )


    # ------------------------------------------------------
    # IDENTIFY CURRENT MONTH ROWS
    # ------------------------------------------------------

    current_mask = (
        (work["date"] >= month_start)
        &
        (work["date"] <= month_end)
    )


    out = (
        work.loc[
            current_mask,
            required_columns
        ]
        .copy()
    )


    out = out.reset_index(
        drop=True
    )


    target_keys = pd.MultiIndex.from_frame(
        out[
            [
                "serial_number",
                "date"
            ]
        ]
    )


    print(
        "Output rows:",
        f"{len(out):,}"
    )


    # ======================================================
    # DAYS SINCE PREVIOUS RECORD
    # ======================================================

    previous_gap = (
        work
        .groupby(
            "serial_number",
            sort=False
        )["date"]
        .diff()
        .dt.days
    )


    previous_gap.index = (
        work_keys
    )


    out[
        "days_since_previous_record"
    ] = (
        previous_gap
        .reindex(
            target_keys
        )
        .to_numpy()
    )


    # ======================================================
    # 1-DAY SMART DIFFERENCE
    # ======================================================

    print(
        "Creating 1-day SMART changes..."
    )


    differences = (
        work
        .groupby(
            "serial_number",
            sort=False
        )[SMART_FEATURES]
        .diff()
    )


    differences.index = (
        work_keys
    )


    target_differences = (
        differences
        .reindex(
            target_keys
        )
    )


    # A difference is called "1d" only when the previous
    # observation is actually one calendar day earlier.

    valid_1d = (
        out[
            "days_since_previous_record"
        ]
        == 1
    )


    for feature in SMART_FEATURES:

        new_col = (
            f"{feature}_diff_1d"
        )

        values = (
            target_differences[
                feature
            ]
            .to_numpy(
                dtype="float32"
            )
        )


        values[
            ~valid_1d.to_numpy()
        ] = np.nan


        out[
            new_col
        ] = values


    del differences
    del target_differences


    # ======================================================
    # CALENDAR ROLLING WINDOWS
    # ======================================================

    grouped = (
        work.groupby(
            "serial_number",
            sort=False
        )
    )


    for window in WINDOWS:

        print(
            f"Creating {window}-calendar-day features..."
        )


        # --------------------------------------------------
        # OBSERVATION COUNT
        # --------------------------------------------------

        observation_count = (
            grouped
            .rolling(
                window=f"{window}D",
                on="date",
                min_periods=1
            )["failure"]
            .count()
        )


        out[
            f"observations_{window}d"
        ] = (
            observation_count
            .reindex(
                target_keys
            )
            .to_numpy()
            .astype(
                "int16"
            )
        )


        del observation_count


        # --------------------------------------------------
        # ROLLING MEAN
        # --------------------------------------------------

        rolling_mean = (
            grouped
            .rolling(
                window=f"{window}D",
                on="date",
                min_periods=1
            )[SMART_FEATURES]
            .mean()
        )


        rolling_mean = (
            rolling_mean
            .reindex(
                target_keys
            )
        )


        for feature in SMART_FEATURES:

            out[
                f"{feature}_mean_{window}d"
            ] = (
                rolling_mean[
                    feature
                ]
                .to_numpy(
                    dtype="float32"
                )
            )


        del rolling_mean


        # --------------------------------------------------
        # ROLLING STANDARD DEVIATION
        # --------------------------------------------------

        rolling_std = (
            grouped
            .rolling(
                window=f"{window}D",
                on="date",
                min_periods=2
            )[SMART_FEATURES]
            .std()
        )


        rolling_std = (
            rolling_std
            .reindex(
                target_keys
            )
        )


        for feature in SMART_FEATURES:

            out[
                f"{feature}_std_{window}d"
            ] = (
                rolling_std[
                    feature
                ]
                .to_numpy(
                    dtype="float32"
                )
            )


        del rolling_std


        # --------------------------------------------------
        # ROLLING MAXIMUM
        # --------------------------------------------------

        rolling_max = (
            grouped
            .rolling(
                window=f"{window}D",
                on="date",
                min_periods=1
            )[SMART_FEATURES]
            .max()
        )


        rolling_max = (
            rolling_max
            .reindex(
                target_keys
            )
        )


        for feature in SMART_FEATURES:

            out[
                f"{feature}_max_{window}d"
            ] = (
                rolling_max[
                    feature
                ]
                .to_numpy(
                    dtype="float32"
                )
            )


        del rolling_max


    # ======================================================
    # HISTORY COVERAGE FLAGS
    # ======================================================

    out[
        "has_sufficient_7d_history"
    ] = (
        out[
            "observations_7d"
        ]
        >= 6
    )


    out[
        "has_sufficient_14d_history"
    ] = (
        out[
            "observations_14d"
        ]
        >= 12
    )


    out[
        "has_sufficient_30d_history"
    ] = (
        out[
            "observations_30d"
        ]
        >= 24
    )


    # ======================================================
    # MONTH SUMMARY
    # ======================================================

    month_summary = {
        "month":
            str(month),

        "rows":
            len(out),

        "unique_drives":
            out[
                "serial_number"
            ].nunique(),

        "rows_with_7d_history":
            int(
                out[
                    "has_sufficient_7d_history"
                ].sum()
            ),

        "rows_with_14d_history":
            int(
                out[
                    "has_sufficient_14d_history"
                ].sum()
            ),

        "rows_with_30d_history":
            int(
                out[
                    "has_sufficient_30d_history"
                ].sum()
            ),
    }


    for horizon in WINDOWS:

        target = (
            f"failure_within_{horizon}d"
        )

        month_summary[
            f"positive_{horizon}d"
        ] = int(
            (
                out[target]
                == 1
            ).sum()
        )

        month_summary[
            f"negative_{horizon}d"
        ] = int(
            (
                out[target]
                == 0
            ).sum()
        )

        month_summary[
            f"censored_{horizon}d"
        ] = int(
            out[target]
            .isna()
            .sum()
        )


    month_summaries.append(
        month_summary
    )


    # ======================================================
    # FEATURE MISSINGNESS AUDIT
    # ======================================================

    model_feature_columns = [
        col
        for col in out.columns
        if (
            col in SMART_FEATURES
            or "_diff_1d" in col
            or "_mean_" in col
            or "_std_" in col
            or "_max_" in col
        )
    ]


    for col in model_feature_columns:

        feature_total[col] = (
            feature_total.get(
                col,
                0
            )
            + len(out)
        )

        feature_missing[col] = (
            feature_missing.get(
                col,
                0
            )
            + int(
                out[col]
                .isna()
                .sum()
            )
        )


    # ======================================================
    # SAVE MONTH
    # ======================================================

    output_file = (
        OUTPUT_DIR
        / f"{month}.parquet"
    )


    out.to_parquet(
        output_file,
        index=False,
        compression="zstd"
    )


    print(
        "Saved:",
        output_file
    )


    del work
    del out


# ==========================================================
# SAVE MONTH SUMMARY
# ==========================================================

month_summary_df = pd.DataFrame(
    month_summaries
)

month_summary_file = (
    AUDIT_DIR
    / "temporal_month_summary.csv"
)

month_summary_df.to_csv(
    month_summary_file,
    index=False
)


# ==========================================================
# SAVE FEATURE AUDIT
# ==========================================================

feature_rows = []

for feature in sorted(
    feature_total
):

    total = (
        feature_total[
            feature
        ]
    )

    missing = (
        feature_missing[
            feature
        ]
    )

    feature_rows.append(
        {
            "feature":
                feature,

            "total_rows":
                total,

            "missing_rows":
                missing,

            "missing_pct":
                round(
                    missing
                    / total
                    * 100,
                    6
                )
        }
    )


feature_summary_df = pd.DataFrame(
    feature_rows
)

feature_summary_file = (
    AUDIT_DIR
    / "temporal_feature_summary.csv"
)

feature_summary_df.to_csv(
    feature_summary_file,
    index=False
)


# ==========================================================
# FINAL SUMMARY
# ==========================================================

print("\n")
print("=" * 70)
print("Q3 + Q4 TEMPORAL FEATURE ENGINEERING COMPLETE")
print("=" * 70)

print(
    "Months processed:",
    len(month_summary_df)
)

print(
    "SMART attributes:",
    len(SMART_FEATURES)
)

print(
    "Model feature columns:",
    len(feature_summary_df)
)

print(
    "\nMonthly summary:\n"
)

print(
    month_summary_df.to_string(
        index=False
    )
)

print(
    "\nTemporal datasets saved to:"
)

print(
    OUTPUT_DIR
)

print(
    "\nFeature summary saved to:"
)

print(
    feature_summary_file
)