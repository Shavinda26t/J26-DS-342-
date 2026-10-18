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
    / "causal"
    / "ST12000NM0008_Q3_Q4"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "causal_final"
    / "ST12000NM0008_Q3_Q4"
)

AUDIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "causal_final_audit"
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

TARGET = "failure_within_30d"


# These are cumulative counters.
# Negative daily changes are treated as invalid reset/wrap
# transitions rather than genuine negative activity.

MONOTONIC_DELTA_SOURCES = [
    "smart_4_raw",
    "smart_5_raw",
    "smart_9_raw",
    "smart_12_raw",
    "smart_187_raw",
    "smart_192_raw",
    "smart_193_raw",
    "smart_199_raw",
    "smart_241_raw",
    "smart_242_raw",
]


# ==========================================================
# HELPERS
# ==========================================================

def positive_log_delta(series):

    """
    For cumulative counters:
    - negative delta -> NaN
    - zero remains zero
    - positive changes -> log1p
    """

    series = pd.to_numeric(
        series,
        errors="coerce"
    ).astype("float64")

    series = series.mask(
        series < 0,
        np.nan
    )

    return np.log1p(
        series
    )


def signed_log1p(series):

    """
    Preserve the direction of variables that may
    legitimately increase or decrease.
    """

    series = pd.to_numeric(
        series,
        errors="coerce"
    ).astype("float64")

    return (
        np.sign(series)
        * np.log1p(
            np.abs(series)
        )
    )


# ==========================================================
# FIND MONTHLY FILES
# ==========================================================

files = sorted(
    INPUT_DIR.glob("*.parquet")
)

if not files:

    raise FileNotFoundError(
        f"No causal files found in {INPUT_DIR}"
    )


print(
    "Causal files found:",
    len(files)
)


# ==========================================================
# FINAL VARIABLE DEFINITIONS
# ==========================================================

FINAL_FEATURES = [

    # ------------------------------------------------------
    # AGE / EXPOSURE CONTEXT
    # ------------------------------------------------------

    "drive_age_hours",


    # ------------------------------------------------------
    # DEGRADATION STATE
    # ------------------------------------------------------

    "reallocated_sectors",
    "reallocated_increase_1d",

    "reported_uncorrectable",
    "reported_uncorrectable_increase_1d",

    "pending_sectors",
    "pending_sector_change_1d",

    "crc_errors",
    "crc_error_increase_1d",


    # ------------------------------------------------------
    # TEMPERATURE
    # ------------------------------------------------------

    "temperature",
    "temperature_change_1d",


    # ------------------------------------------------------
    # DAILY OPERATING ACTIVITY
    # ------------------------------------------------------

    "start_stop_activity_1d",
    "power_cycle_activity_1d",
    "poweroff_retract_activity_1d",
    "load_cycle_activity_1d",


    # ------------------------------------------------------
    # DAILY WORKLOAD
    # ------------------------------------------------------

    "write_workload_1d",
    "read_workload_1d",
]


# ==========================================================
# AUDIT STORAGE
# ==========================================================

negative_delta_rows = []

feature_missing_counts = {
    feature: 0
    for feature in FINAL_FEATURES
}

feature_total_counts = {
    feature: 0
    for feature in FINAL_FEATURES
}

month_summary_rows = []


# ==========================================================
# PROCESS MONTH BY MONTH
# ==========================================================

for i, file in enumerate(
    files,
    start=1
):

    print(
        f"[{i}/{len(files)}] "
        f"Processing {file.name}"
    )

    df = pd.read_parquet(
        file
    )


    # ======================================================
    # IDENTIFIERS / OUTCOME
    # ======================================================

    out = df[
        [
            "date",
            "serial_number",
            "days_since_previous_record",
            TARGET,
        ]
    ].copy()


    # ======================================================
    # 1. DRIVE AGE
    # ======================================================

    out[
        "drive_age_hours"
    ] = pd.to_numeric(
        df["smart_9_raw"],
        errors="coerce"
    ).astype(
        "float64"
    )


    # ======================================================
    # 2. REALLOCATED SECTORS - SMART 5
    # ======================================================

    out[
        "reallocated_sectors"
    ] = np.log1p(
        pd.to_numeric(
            df["smart_5_raw"],
            errors="coerce"
        ).clip(
            lower=0
        )
    )


    out[
        "reallocated_increase_1d"
    ] = positive_log_delta(
        df[
            "smart_5_raw_delta_1d"
        ]
    )


    # ======================================================
    # 3. REPORTED UNCORRECTABLE - SMART 187
    # ======================================================

    out[
        "reported_uncorrectable"
    ] = np.log1p(
        pd.to_numeric(
            df["smart_187_raw"],
            errors="coerce"
        ).clip(
            lower=0
        )
    )


    out[
        "reported_uncorrectable_increase_1d"
    ] = positive_log_delta(
        df[
            "smart_187_raw_delta_1d"
        ]
    )


    # ======================================================
    # 4. CURRENT PENDING SECTOR - SMART 197
    #
    # This is a state/gauge and may increase OR decrease.
    # ======================================================

    out[
        "pending_sectors"
    ] = np.log1p(
        pd.to_numeric(
            df["smart_197_raw"],
            errors="coerce"
        ).clip(
            lower=0
        )
    )


    out[
        "pending_sector_change_1d"
    ] = signed_log1p(
        df[
            "smart_197_raw_delta_1d"
        ]
    )


    # ======================================================
    # 5. CRC ERRORS - SMART 199
    # ======================================================

    out[
        "crc_errors"
    ] = np.log1p(
        pd.to_numeric(
            df["smart_199_raw"],
            errors="coerce"
        ).clip(
            lower=0
        )
    )


    out[
        "crc_error_increase_1d"
    ] = positive_log_delta(
        df[
            "smart_199_raw_delta_1d"
        ]
    )


    # ======================================================
    # 6. TEMPERATURE - SMART 194
    #
    # Temperature legitimately increases and decreases.
    # ======================================================

    out[
        "temperature"
    ] = pd.to_numeric(
        df[
            "smart_194_raw"
        ],
        errors="coerce"
    ).astype(
        "float64"
    )


    out[
        "temperature_change_1d"
    ] = pd.to_numeric(
        df[
            "smart_194_raw_delta_1d"
        ],
        errors="coerce"
    ).astype(
        "float64"
    )


    # ======================================================
    # 7. DAILY OPERATING ACTIVITY
    # ======================================================

    out[
        "start_stop_activity_1d"
    ] = positive_log_delta(
        df[
            "smart_4_raw_delta_1d"
        ]
    )


    out[
        "power_cycle_activity_1d"
    ] = positive_log_delta(
        df[
            "smart_12_raw_delta_1d"
        ]
    )


    out[
        "poweroff_retract_activity_1d"
    ] = positive_log_delta(
        df[
            "smart_192_raw_delta_1d"
        ]
    )


    out[
        "load_cycle_activity_1d"
    ] = positive_log_delta(
        df[
            "smart_193_raw_delta_1d"
        ]
    )


    # ======================================================
    # 8. DAILY WORKLOAD
    # ======================================================

    out[
        "write_workload_1d"
    ] = positive_log_delta(
        df[
            "smart_241_raw_delta_1d"
        ]
    )


    out[
        "read_workload_1d"
    ] = positive_log_delta(
        df[
            "smart_242_raw_delta_1d"
        ]
    )


    # ======================================================
    # AUDIT NEGATIVE DELTAS IN MONOTONIC COUNTERS
    # ======================================================

    for source in MONOTONIC_DELTA_SOURCES:

        delta_col = (
            f"{source}_delta_1d"
        )

        values = pd.to_numeric(
            df[
                delta_col
            ],
            errors="coerce"
        )

        negative_count = int(
            (
                values < 0
            ).sum()
        )


        negative_delta_rows.append(
            {
                "month":
                    file.stem,

                "source_feature":
                    source,

                "negative_delta_rows":
                    negative_count,
            }
        )


    # ======================================================
    # MISSINGNESS AUDIT
    # ======================================================

    for feature in FINAL_FEATURES:

        feature_total_counts[
            feature
        ] += len(
            out
        )

        feature_missing_counts[
            feature
        ] += int(
            out[
                feature
            ]
            .isna()
            .sum()
        )


    # ======================================================
    # MONTH SUMMARY
    # ======================================================

    month_summary_rows.append(
        {
            "month":
                file.stem,

            "rows":
                len(
                    out
                ),

            "unique_hdds":
                out[
                    "serial_number"
                ].nunique(),

            "positive_rows":
                int(
                    (
                        out[
                            TARGET
                        ]
                        == 1
                    ).sum()
                ),

            "negative_rows":
                int(
                    (
                        out[
                            TARGET
                        ]
                        == 0
                    ).sum()
                ),
        }
    )


    # ======================================================
    # SAVE MONTH
    # ======================================================

    output_file = (
        OUTPUT_DIR
        / file.name
    )


    out.to_parquet(
        output_file,
        index=False,
        compression="zstd"
    )


# ==========================================================
# FINAL FEATURE AUDIT
# ==========================================================

feature_audit_rows = []


for feature in FINAL_FEATURES:

    total = (
        feature_total_counts[
            feature
        ]
    )

    missing = (
        feature_missing_counts[
            feature
        ]
    )


    feature_audit_rows.append(
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


feature_audit_df = pd.DataFrame(
    feature_audit_rows
)

negative_delta_df = pd.DataFrame(
    negative_delta_rows
)

month_summary_df = pd.DataFrame(
    month_summary_rows
)


# ==========================================================
# FEATURE MANIFEST
# ==========================================================

feature_manifest_df = pd.DataFrame(
    [
        {
            "feature":
                "drive_age_hours",

            "role":
                "context_confounder"
        },

        {
            "feature":
                "reallocated_sectors",

            "role":
                "degradation_state"
        },

        {
            "feature":
                "reallocated_increase_1d",

            "role":
                "degradation_change"
        },

        {
            "feature":
                "reported_uncorrectable",

            "role":
                "error_state"
        },

        {
            "feature":
                "reported_uncorrectable_increase_1d",

            "role":
                "error_change"
        },

        {
            "feature":
                "pending_sectors",

            "role":
                "degradation_state"
        },

        {
            "feature":
                "pending_sector_change_1d",

            "role":
                "degradation_change"
        },

        {
            "feature":
                "crc_errors",

            "role":
                "interface_error_state"
        },

        {
            "feature":
                "crc_error_increase_1d",

            "role":
                "interface_error_change"
        },

        {
            "feature":
                "temperature",

            "role":
                "environment_context"
        },

        {
            "feature":
                "temperature_change_1d",

            "role":
                "environment_change"
        },

        {
            "feature":
                "start_stop_activity_1d",

            "role":
                "usage_context"
        },

        {
            "feature":
                "power_cycle_activity_1d",

            "role":
                "usage_context"
        },

        {
            "feature":
                "poweroff_retract_activity_1d",

            "role":
                "usage_context"
        },

        {
            "feature":
                "load_cycle_activity_1d",

            "role":
                "mechanical_usage"
        },

        {
            "feature":
                "write_workload_1d",

            "role":
                "workload_context"
        },

        {
            "feature":
                "read_workload_1d",

            "role":
                "workload_context"
        },
    ]
)


# ==========================================================
# SAVE AUDITS
# ==========================================================

feature_audit_file = (
    AUDIT_DIR
    / "final_causal_feature_audit.csv"
)

negative_delta_file = (
    AUDIT_DIR
    / "negative_counter_delta_audit.csv"
)

month_summary_file = (
    AUDIT_DIR
    / "final_causal_month_summary.csv"
)

manifest_file = (
    AUDIT_DIR
    / "final_causal_feature_manifest.csv"
)


feature_audit_df.to_csv(
    feature_audit_file,
    index=False
)

negative_delta_df.to_csv(
    negative_delta_file,
    index=False
)

month_summary_df.to_csv(
    month_summary_file,
    index=False
)

feature_manifest_df.to_csv(
    manifest_file,
    index=False
)


# ==========================================================
# FINAL OUTPUT
# ==========================================================

print("\n")
print("=" * 70)
print("FINAL CAUSAL VARIABLE PREPARATION COMPLETE")
print("=" * 70)


print(
    "Final causal predictors:",
    len(
        FINAL_FEATURES
    )
)


print(
    "\nFeature missingness:\n"
)

print(
    feature_audit_df.to_string(
        index=False
    )
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
    "\nNegative cumulative-counter changes:"
)

negative_summary = (
    negative_delta_df
    .groupby(
        "source_feature"
    )[
        "negative_delta_rows"
    ]
    .sum()
    .reset_index()
)


print(
    negative_summary.to_string(
        index=False
    )
)


print(
    "\nFinal causal data saved to:"
)

print(
    OUTPUT_DIR
)


print(
    "\nFeature manifest saved to:"
)

print(
    manifest_file
)