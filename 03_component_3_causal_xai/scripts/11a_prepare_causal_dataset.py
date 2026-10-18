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

CAUSAL_MANIFEST = (
    PROJECT_ROOT
    / "config"
    / "causal_features.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "causal"
    / "ST12000NM0008_Q3_Q4"
)

AUDIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "causal_dataset_audit"
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

# July is kept out of the final causal dataset.
# We only need July 31 as possible context for Aug 1 deltas.
CONTEXT_START = pd.Timestamp(
    "2025-07-31"
)

ANALYSIS_START = pd.Timestamp(
    "2025-08-01"
)

ANALYSIS_END = pd.Timestamp(
    "2025-11-30"
)


# ==========================================================
# LOAD CAUSAL FEATURE MANIFEST
# ==========================================================

manifest = pd.read_csv(
    CAUSAL_MANIFEST
)

manifest[
    "use_in_causal_analysis"
] = (
    manifest[
        "use_in_causal_analysis"
    ]
    .astype(str)
    .str.strip()
    .str.lower()
)


selected_manifest = (
    manifest[
        manifest[
            "use_in_causal_analysis"
        ] == "yes"
    ]
    .copy()
)


RAW_FEATURES = (
    selected_manifest[
        "raw_feature"
    ]
    .tolist()
)


review_features = (
    manifest[
        manifest[
            "use_in_causal_analysis"
        ] == "review"
    ][
        "raw_feature"
    ]
    .tolist()
)


print(
    "Selected causal raw features:",
    len(RAW_FEATURES)
)

for feature in RAW_FEATURES:

    print(
        "  KEEP:",
        feature
    )


print(
    "\nExcluded pending semantic review:"
)

for feature in review_features:

    print(
        "  REVIEW:",
        feature
    )


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


selected_files = []


for file in files:

    try:

        date = pd.Timestamp(
            file.stem
        )

    except ValueError:

        continue


    if (
        CONTEXT_START
        <= date
        <= ANALYSIS_END
    ):

        selected_files.append(
            file
        )


print(
    "\nDaily files used:",
    len(selected_files)
)


# ==========================================================
# LOAD ONLY NECESSARY COLUMNS
# ==========================================================

columns = (
    [
        "date",
        "serial_number",
        "eligible_for_prediction",
        TARGET,
    ]
    + RAW_FEATURES
)


parts = []


for i, file in enumerate(
    selected_files,
    start=1
):

    print(
        f"[{i}/{len(selected_files)}] "
        f"{file.name}"
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


print(
    "\nRows loaded including context:",
    f"{len(df):,}"
)


# ==========================================================
# PREPARE DATE / SORT
# ==========================================================

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)


df = df.dropna(
    subset=[
        "date",
        "serial_number"
    ]
)


df = df.sort_values(
    [
        "serial_number",
        "date"
    ]
).reset_index(
    drop=True
)


# ==========================================================
# KEEP SMART VALUES AS FLOAT64
#
# Important:
# SMART 241/242 can contain very large cumulative values.
# Float64 avoids losing precision before calculating deltas.
# ==========================================================

for feature in RAW_FEATURES:

    df[feature] = pd.to_numeric(
        df[feature],
        errors="coerce"
    ).astype(
        "float64"
    )


# ==========================================================
# TEMPORAL CONTINUITY
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


# ==========================================================
# TRUE 1-DAY DELTAS
# ==========================================================

print(
    "\nCreating per-HDD 1-day deltas..."
)


for feature in RAW_FEATURES:

    delta_name = (
        f"{feature}_delta_1d"
    )


    delta = (
        df
        .groupby(
            "serial_number",
            sort=False
        )[feature]
        .diff()
    )


    # Only call it a one-day change when the immediately
    # previous record is exactly one calendar day earlier.
    delta.loc[
        df[
            "days_since_previous_record"
        ] != 1
    ] = np.nan


    df[
        delta_name
    ] = delta


# ==========================================================
# FILTER TO ANALYSIS PERIOD
# ==========================================================

df = df[
    (df["date"] >= ANALYSIS_START)
    & (df["date"] <= ANALYSIS_END)
].copy()


# ==========================================================
# ONLY PRE-FAILURE / PREDICTION-ELIGIBLE ROWS
# ==========================================================

df = df[
    df[
        "eligible_for_prediction"
    ] == True
].copy()


# ==========================================================
# REMOVE CENSORED 30-DAY OUTCOMES
# ==========================================================

before_target_filter = len(
    df
)


df = df[
    df[
        TARGET
    ].notna()
].copy()


censored_removed = (
    before_target_filter
    - len(df)
)


df[
    TARGET
] = (
    df[
        TARGET
    ]
    .astype(
        "int8"
    )
)


print(
    "\nCensored rows removed:",
    f"{censored_removed:,}"
)


# ==========================================================
# FINAL CAUSAL FEATURE LIST
# ==========================================================

DELTA_FEATURES = [
    f"{feature}_delta_1d"
    for feature in RAW_FEATURES
]


CAUSAL_FEATURES = (
    RAW_FEATURES
    + DELTA_FEATURES
)


# ==========================================================
# CAUSAL VARIABLE AUDIT
# ==========================================================

audit_rows = []


for feature in CAUSAL_FEATURES:

    values = pd.to_numeric(
        df[feature],
        errors="coerce"
    )


    valid = (
        values.dropna()
    )


    missing_rows = int(
        values.isna().sum()
    )


    if len(valid) > 0:

        minimum = float(
            valid.min()
        )

        maximum = float(
            valid.max()
        )

        mean = float(
            valid.mean()
        )

        median = float(
            valid.median()
        )

        nonzero_pct = float(
            (
                valid != 0
            ).mean()
            * 100
        )

        negative_pct = float(
            (
                valid < 0
            ).mean()
            * 100
        )

    else:

        minimum = np.nan
        maximum = np.nan
        mean = np.nan
        median = np.nan
        nonzero_pct = np.nan
        negative_pct = np.nan


    audit_rows.append(
        {
            "feature":
                feature,

            "total_rows":
                len(df),

            "missing_rows":
                missing_rows,

            "missing_pct":
                round(
                    missing_rows
                    / len(df)
                    * 100,
                    6
                ),

            "nonzero_pct":
                round(
                    nonzero_pct,
                    6
                )
                if not np.isnan(
                    nonzero_pct
                )
                else np.nan,

            "negative_pct":
                round(
                    negative_pct,
                    6
                )
                if not np.isnan(
                    negative_pct
                )
                else np.nan,

            "min":
                minimum,

            "median":
                median,

            "mean":
                mean,

            "max":
                maximum,
        }
    )


audit_df = pd.DataFrame(
    audit_rows
)


audit_file = (
    AUDIT_DIR
    / "causal_variable_audit.csv"
)

audit_df.to_csv(
    audit_file,
    index=False
)


# ==========================================================
# MONTHLY DATASET SUMMARY
# ==========================================================

df[
    "month"
] = (
    df[
        "date"
    ]
    .dt
    .to_period(
        "M"
    )
    .astype(str)
)


month_summary_rows = []


for month, month_df in (
    df.groupby(
        "month"
    )
):

    positive = (
        month_df[
            month_df[
                TARGET
            ] == 1
        ]
    )


    month_summary_rows.append(
        {
            "month":
                month,

            "rows":
                len(
                    month_df
                ),

            "unique_hdds":
                month_df[
                    "serial_number"
                ].nunique(),

            "positive_rows":
                int(
                    (
                        month_df[
                            TARGET
                        ] == 1
                    ).sum()
                ),

            "negative_rows":
                int(
                    (
                        month_df[
                            TARGET
                        ] == 0
                    ).sum()
                ),

            "unique_positive_hdds":
                positive[
                    "serial_number"
                ].nunique(),
        }
    )


month_summary_df = pd.DataFrame(
    month_summary_rows
)


month_summary_file = (
    AUDIT_DIR
    / "causal_dataset_month_summary.csv"
)


month_summary_df.to_csv(
    month_summary_file,
    index=False
)


# ==========================================================
# SAVE CAUSAL FEATURE MANIFEST
# ==========================================================

manifest_rows = []


role_map = dict(
    zip(
        selected_manifest[
            "raw_feature"
        ],
        selected_manifest[
            "role"
        ]
    )
)


for feature in RAW_FEATURES:

    manifest_rows.append(
        {
            "feature":
                feature,

            "source_feature":
                feature,

            "transformation":
                "level",

            "role":
                role_map[
                    feature
                ],
        }
    )


    manifest_rows.append(
        {
            "feature":
                f"{feature}_delta_1d",

            "source_feature":
                feature,

            "transformation":
                "1_day_delta",

            "role":
                role_map[
                    feature
                ],
        }
    )


causal_feature_manifest_df = (
    pd.DataFrame(
        manifest_rows
    )
)


causal_feature_manifest_file = (
    AUDIT_DIR
    / "causal_analysis_feature_manifest.csv"
)


causal_feature_manifest_df.to_csv(
    causal_feature_manifest_file,
    index=False
)


# ==========================================================
# SAVE DATA MONTH BY MONTH
# ==========================================================

OUTPUT_COLUMNS = (
    [
        "date",
        "serial_number",
        "days_since_previous_record",
        TARGET,
    ]
    + CAUSAL_FEATURES
)


for month, month_df in (
    df.groupby(
        "month"
    )
):

    output_file = (
        OUTPUT_DIR
        / f"{month}.parquet"
    )


    month_df[
        OUTPUT_COLUMNS
    ].to_parquet(
        output_file,
        index=False,
        compression="zstd"
    )


    print(
        "Saved:",
        output_file
    )


# ==========================================================
# FINAL OUTPUT
# ==========================================================

print("\n")
print("=" * 70)

print(
    "CAUSAL DATASET PREPARATION COMPLETE"
)

print("=" * 70)


print(
    "Analysis period:",
    ANALYSIS_START.date(),
    "to",
    ANALYSIS_END.date()
)


print(
    "Rows:",
    f"{len(df):,}"
)


print(
    "Unique HDDs:",
    f"{df['serial_number'].nunique():,}"
)


print(
    "Raw causal variables:",
    len(
        RAW_FEATURES
    )
)


print(
    "Delta causal variables:",
    len(
        DELTA_FEATURES
    )
)


print(
    "Total causal candidate variables:",
    len(
        CAUSAL_FEATURES
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
    "\nVariable audit saved to:"
)

print(
    audit_file
)


print(
    "\nCausal feature manifest saved to:"
)

print(
    causal_feature_manifest_file
)


print(
    "\nCausal datasets saved to:"
)

print(
    OUTPUT_DIR
)