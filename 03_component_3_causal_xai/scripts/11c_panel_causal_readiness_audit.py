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
    / "causal_final"
    / "ST12000NM0008_Q3_Q4"
)

MANIFEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "causal_final_audit"
    / "final_causal_feature_manifest.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "panel_causal_readiness"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# SETTINGS
# ==========================================================

TARGET = "failure_within_30d"

CANDIDATE_LAGS = [
    1,
    3,
    7,
    14
]


# ==========================================================
# LOAD FEATURE MANIFEST
# ==========================================================

manifest = pd.read_csv(
    MANIFEST_FILE
)

CAUSAL_FEATURES = (
    manifest[
        "feature"
    ]
    .tolist()
)

print(
    "Causal predictors:",
    len(CAUSAL_FEATURES)
)


# ==========================================================
# LOAD CAUSAL DATA
# ==========================================================

files = sorted(
    INPUT_DIR.glob("*.parquet")
)

if not files:

    raise FileNotFoundError(
        f"No causal files found in {INPUT_DIR}"
    )


parts = []


for i, file in enumerate(
    files,
    start=1
):

    print(
        f"[{i}/{len(files)}] "
        f"Loading {file.name}"
    )

    columns = (
        [
            "date",
            "serial_number",
            "days_since_previous_record",
            TARGET,
        ]
        + CAUSAL_FEATURES
    )


    temp = pd.read_parquet(
        file,
        columns=columns
    )

    parts.append(
        temp
    )


df = pd.concat(
    parts,
    ignore_index=True
)

del parts


df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)


df = df.sort_values(
    [
        "serial_number",
        "date"
    ]
).reset_index(
    drop=True
)


print(
    "\nRows loaded:",
    f"{len(df):,}"
)

print(
    "Unique HDDs:",
    f"{df['serial_number'].nunique():,}"
)


# ==========================================================
# COMPLETE-CASE FLAG
# ==========================================================

df[
    "complete_causal_row"
] = (
    df[
        CAUSAL_FEATURES
    ]
    .notna()
    .all(
        axis=1
    )
)


# ==========================================================
# DAILY CONTINUITY FLAG
# ==========================================================

df[
    "is_daily_transition"
] = (
    df[
        "days_since_previous_record"
    ]
    == 1
)


# ==========================================================
# HDD-LEVEL SUMMARY
# ==========================================================

drive_summary_rows = []


for serial, drive in df.groupby(
    "serial_number",
    sort=False
):

    drive = drive.sort_values(
        "date"
    )

    rows = len(
        drive
    )

    complete_rows = int(
        drive[
            "complete_causal_row"
        ].sum()
    )

    transitions = (
        drive[
            "days_since_previous_record"
        ]
        .dropna()
    )

    daily_transitions = int(
        (
            transitions == 1
        ).sum()
    )

    gap_transitions = int(
        (
            transitions > 1
        ).sum()
    )

    max_gap = (
        float(
            transitions.max()
        )
        if len(
            transitions
        ) > 0
        else np.nan
    )

    continuity_pct = (
        daily_transitions
        / len(
            transitions
        )
        * 100
        if len(
            transitions
        ) > 0
        else np.nan
    )

    positive_rows = int(
        (
            drive[
                TARGET
            ]
            == 1
        ).sum()
    )

    drive_summary_rows.append(
        {
            "serial_number":
                serial,

            "first_date":
                drive[
                    "date"
                ].min(),

            "last_date":
                drive[
                    "date"
                ].max(),

            "rows":
                rows,

            "complete_rows":
                complete_rows,

            "complete_row_pct":
                round(
                    complete_rows
                    / rows
                    * 100,
                    6
                )
                if rows > 0
                else np.nan,

            "transitions":
                len(
                    transitions
                ),

            "daily_transitions":
                daily_transitions,

            "gap_transitions":
                gap_transitions,

            "continuity_pct":
                round(
                    continuity_pct,
                    6
                )
                if not np.isnan(
                    continuity_pct
                )
                else np.nan,

            "max_gap_days":
                max_gap,

            "positive_30d_rows":
                positive_rows,

            "has_positive_30d":
                positive_rows > 0,
        }
    )


drive_summary = pd.DataFrame(
    drive_summary_rows
)


# ==========================================================
# LAG ELIGIBILITY
# ==========================================================

lag_summary_rows = []


for lag in CANDIDATE_LAGS:

    # Conservative requirement:
    # at least 4x the maximum lag in available rows.
    minimum_rows = (
        lag * 4
    )

    eligible = drive_summary[
        (
            drive_summary[
                "complete_rows"
            ]
            >= minimum_rows
        )
        &
        (
            drive_summary[
                "continuity_pct"
            ]
            >= 95.0
        )
    ]


    lag_summary_rows.append(
        {
            "max_lag_days":
                lag,

            "minimum_complete_rows":
                minimum_rows,

            "eligible_hdds":
                len(
                    eligible
                ),

            "eligible_positive_hdds":
                int(
                    eligible[
                        "has_positive_30d"
                    ].sum()
                ),

            "median_complete_rows":
                (
                    eligible[
                        "complete_rows"
                    ]
                    .median()
                    if len(
                        eligible
                    ) > 0
                    else np.nan
                ),

            "median_continuity_pct":
                (
                    eligible[
                        "continuity_pct"
                    ]
                    .median()
                    if len(
                        eligible
                    ) > 0
                    else np.nan
                ),
        }
    )


lag_summary = pd.DataFrame(
    lag_summary_rows
)


# ==========================================================
# DATASET-LEVEL SUMMARY
# ==========================================================

complete_rows_total = int(
    df[
        "complete_causal_row"
    ].sum()
)


dataset_summary = pd.DataFrame(
    [
        {
            "rows":
                len(df),

            "unique_hdds":
                df[
                    "serial_number"
                ].nunique(),

            "complete_causal_rows":
                complete_rows_total,

            "complete_causal_row_pct":
                round(
                    complete_rows_total
                    / len(df)
                    * 100,
                    6
                ),

            "hdds_with_100pct_continuity":
                int(
                    (
                        drive_summary[
                            "continuity_pct"
                        ]
                        == 100
                    ).sum()
                ),

            "hdds_with_at_least_99pct_continuity":
                int(
                    (
                        drive_summary[
                            "continuity_pct"
                        ]
                        >= 99
                    ).sum()
                ),

            "hdds_with_at_least_95pct_continuity":
                int(
                    (
                        drive_summary[
                            "continuity_pct"
                        ]
                        >= 95
                    ).sum()
                ),

            "hdds_with_positive_30d":
                int(
                    drive_summary[
                        "has_positive_30d"
                    ].sum()
                ),

            "median_rows_per_hdd":
                drive_summary[
                    "rows"
                ].median(),

            "median_complete_rows_per_hdd":
                drive_summary[
                    "complete_rows"
                ].median(),

            "median_continuity_pct":
                drive_summary[
                    "continuity_pct"
                ].median(),

            "maximum_gap_days":
                drive_summary[
                    "max_gap_days"
                ].max(),
        }
    ]
)


# ==========================================================
# SAVE RESULTS
# ==========================================================

dataset_summary_file = (
    OUTPUT_DIR
    / "panel_dataset_summary.csv"
)

drive_summary_file = (
    OUTPUT_DIR
    / "panel_drive_summary.csv"
)

lag_summary_file = (
    OUTPUT_DIR
    / "lag_eligibility_summary.csv"
)


dataset_summary.to_csv(
    dataset_summary_file,
    index=False
)

drive_summary.to_csv(
    drive_summary_file,
    index=False
)

lag_summary.to_csv(
    lag_summary_file,
    index=False
)


# ==========================================================
# PRINT RESULTS
# ==========================================================

print("\n")
print("=" * 70)
print("PANEL CAUSAL READINESS AUDIT")
print("=" * 70)


print(
    "\nDataset summary:\n"
)

print(
    dataset_summary.to_string(
        index=False
    )
)


print(
    "\nLag eligibility:\n"
)

print(
    lag_summary.to_string(
        index=False
    )
)


print(
    "\nSequence length distribution:\n"
)

print(
    drive_summary[
        "complete_rows"
    ]
    .describe(
        percentiles=[
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ]
    )
    .to_string()
)


print(
    "\nSaved to:"
)

print(
    OUTPUT_DIR
)