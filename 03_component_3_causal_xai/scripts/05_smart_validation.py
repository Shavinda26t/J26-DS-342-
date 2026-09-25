from pathlib import Path
import numpy as np
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
    / "interim"
    / "smart_validation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# SETTINGS
# ==========================================================

SAMPLE_PER_FILE = 3000

HIGH_CORRELATION_THRESHOLD = 0.95


# ==========================================================
# FIND CLEANED PARQUET FILES
# ==========================================================

parquet_files = sorted(
    CLEANED_DIR.glob("*.parquet")
)

if not parquet_files:
    raise FileNotFoundError(
        f"No Parquet files found in {CLEANED_DIR}"
    )

print("Parquet files found:", len(parquet_files))


# ==========================================================
# DETECT SMART COLUMNS
# ==========================================================

sample_header = pd.read_parquet(
    parquet_files[0]
)

smart_columns = [
    col
    for col in sample_header.columns
    if col.startswith("smart_")
]

print("SMART columns found:", len(smart_columns))


# ==========================================================
# FULL-DATA STATISTICS
# ==========================================================

stats = {}

for col in smart_columns:
    stats[col] = {
        "total": 0,
        "missing": 0,
        "non_missing": 0,
        "zero": 0,
        "non_zero": 0,
        "sum": 0.0,
        "sum_sq": 0.0,
        "min": np.inf,
        "max": -np.inf
    }


# ==========================================================
# SPECIAL PAIRS WE WANT TO VERIFY
# ==========================================================

pairs_to_check = [
    ("smart_190_raw", "smart_194_raw"),
    ("smart_197_raw", "smart_198_raw"),
    ("smart_1_raw", "smart_195_raw")
]

pair_stats = {}

for a, b in pairs_to_check:
    pair_stats[(a, b)] = {
        "both_available": 0,
        "equal": 0,
        "different": 0,
        "only_one_missing": 0
    }


# ==========================================================
# SAMPLE DATA FOR CORRELATION / QUANTILES
# ==========================================================

samples = []


# ==========================================================
# PROCESS FILE BY FILE
# ==========================================================

for i, file in enumerate(
    parquet_files,
    start=1
):

    print(
        f"[{i}/{len(parquet_files)}] "
        f"Processing {file.name}"
    )

    df = pd.read_parquet(file)

    # ------------------------------------------------------
    # FULL DATA STATISTICS
    # ------------------------------------------------------

    for col in smart_columns:

        values = pd.to_numeric(
            df[col],
            errors="coerce"
        ).astype("float64")

        total = len(values)
        missing = values.isna().sum()

        valid = values.dropna()

        stats[col]["total"] += total
        stats[col]["missing"] += missing
        stats[col]["non_missing"] += len(valid)

        if len(valid) == 0:
            continue

        stats[col]["zero"] += (
            valid == 0
        ).sum()

        stats[col]["non_zero"] += (
            valid != 0
        ).sum()

        stats[col]["sum"] += (
            valid.sum()
        )

        stats[col]["sum_sq"] += (
            np.square(valid).sum()
        )

        stats[col]["min"] = min(
            stats[col]["min"],
            valid.min()
        )

        stats[col]["max"] = max(
            stats[col]["max"],
            valid.max()
        )


    # ------------------------------------------------------
    # CHECK SUSPICIOUS PAIRS
    # ------------------------------------------------------

    for a, b in pairs_to_check:

        if (
            a not in df.columns
            or b not in df.columns
        ):
            continue

        a_values = pd.to_numeric(
            df[a],
            errors="coerce"
        )

        b_values = pd.to_numeric(
            df[b],
            errors="coerce"
        )

        both_valid = (
            a_values.notna()
            & b_values.notna()
        )

        one_missing = (
            a_values.isna()
            ^ b_values.isna()
        )

        equal_values = (
            a_values[both_valid]
            == b_values[both_valid]
        )

        pair_stats[(a, b)][
            "both_available"
        ] += both_valid.sum()

        pair_stats[(a, b)][
            "equal"
        ] += equal_values.sum()

        pair_stats[(a, b)][
            "different"
        ] += (
            (~equal_values).sum()
        )

        pair_stats[(a, b)][
            "only_one_missing"
        ] += one_missing.sum()


    # ------------------------------------------------------
    # TAKE SMALL RANDOM SAMPLE
    # ------------------------------------------------------

    n_sample = min(
        SAMPLE_PER_FILE,
        len(df)
    )

    sampled = df[
        smart_columns
    ].sample(
        n=n_sample,
        random_state=42 + i
    )

    samples.append(sampled)


# ==========================================================
# COMBINE RANDOM SAMPLE
# ==========================================================

sample_df = pd.concat(
    samples,
    ignore_index=True
)

print(
    "\nRows used for correlation analysis:",
    f"{len(sample_df):,}"
)


# ==========================================================
# SAMPLE QUANTILES
# ==========================================================

quantiles = sample_df[
    smart_columns
].quantile(
    [
        0.01,
        0.25,
        0.50,
        0.75,
        0.99
    ]
)


# ==========================================================
# CREATE FULL SUMMARY
# ==========================================================

summary_rows = []

for col in smart_columns:

    s = stats[col]

    n = s["non_missing"]

    if n == 0:
        continue

    mean = (
        s["sum"] / n
    )

    variance = (
        s["sum_sq"] / n
        - mean ** 2
    )

    variance = max(
        variance,
        0
    )

    std = np.sqrt(variance)

    missing_pct = (
        s["missing"]
        / s["total"]
        * 100
    )

    non_zero_pct = (
        s["non_zero"]
        / s["total"]
        * 100
    )

    summary_rows.append(
        {
            "smart_attribute": col,

            "total_rows":
                s["total"],

            "missing_pct":
                round(
                    missing_pct,
                    6
                ),

            "non_zero_pct":
                round(
                    non_zero_pct,
                    6
                ),

            "min":
                s["min"],

            "q01":
                quantiles.loc[
                    0.01,
                    col
                ],

            "q25":
                quantiles.loc[
                    0.25,
                    col
                ],

            "median":
                quantiles.loc[
                    0.50,
                    col
                ],

            "q75":
                quantiles.loc[
                    0.75,
                    col
                ],

            "q99":
                quantiles.loc[
                    0.99,
                    col
                ],

            "max":
                s["max"],

            "mean":
                mean,

            "std":
                std
        }
    )


summary_df = pd.DataFrame(
    summary_rows
)

summary_file = (
    OUTPUT_DIR
    / "smart_cleaned_summary.csv"
)

summary_df.to_csv(
    summary_file,
    index=False
)


# ==========================================================
# CORRELATION ANALYSIS
# ==========================================================

correlation_matrix = sample_df[
    smart_columns
].corr(
    method="pearson"
)

correlation_file = (
    OUTPUT_DIR
    / "smart_pearson_correlation.csv"
)

correlation_matrix.to_csv(
    correlation_file
)


# ==========================================================
# FIND HIGHLY CORRELATED PAIRS
# ==========================================================

high_corr_rows = []

for i in range(
    len(smart_columns)
):

    for j in range(
        i + 1,
        len(smart_columns)
    ):

        feature_a = (
            smart_columns[i]
        )

        feature_b = (
            smart_columns[j]
        )

        corr = (
            correlation_matrix.loc[
                feature_a,
                feature_b
            ]
        )

        if pd.isna(corr):
            continue

        if (
            abs(corr)
            >= HIGH_CORRELATION_THRESHOLD
        ):

            high_corr_rows.append(
                {
                    "feature_a":
                        feature_a,

                    "feature_b":
                        feature_b,

                    "pearson_correlation":
                        corr,

                    "absolute_correlation":
                        abs(corr)
                }
            )


high_corr_df = pd.DataFrame(
    high_corr_rows
)

if not high_corr_df.empty:

    high_corr_df = (
        high_corr_df
        .sort_values(
            "absolute_correlation",
            ascending=False
        )
    )


high_corr_file = (
    OUTPUT_DIR
    / "high_correlation_pairs.csv"
)

high_corr_df.to_csv(
    high_corr_file,
    index=False
)


# ==========================================================
# SPECIAL PAIR COMPARISON
# ==========================================================

pair_rows = []

for (a, b), values in pair_stats.items():

    both = values[
        "both_available"
    ]

    equal = values[
        "equal"
    ]

    equality_pct = (
        equal / both * 100
        if both > 0
        else np.nan
    )

    pair_rows.append(
        {
            "feature_a": a,

            "feature_b": b,

            "both_available":
                both,

            "equal_rows":
                equal,

            "different_rows":
                values["different"],

            "equality_pct":
                equality_pct,

            "only_one_missing":
                values[
                    "only_one_missing"
                ]
        }
    )


pair_df = pd.DataFrame(
    pair_rows
)

pair_file = (
    OUTPUT_DIR
    / "suspicious_pair_comparison.csv"
)

pair_df.to_csv(
    pair_file,
    index=False
)


# ==========================================================
# PRINT RESULTS
# ==========================================================

print("\n")
print("=" * 70)
print("SMART VALIDATION COMPLETE")
print("=" * 70)

print(
    "\nSMART attributes analysed:",
    len(smart_columns)
)

print(
    "Correlation sample size:",
    f"{len(sample_df):,}"
)

print(
    "\nHighly correlated pairs (|r| >= 0.95):",
    len(high_corr_df)
)


print("\nSuspicious pair comparison:\n")

print(
    pair_df.to_string(
        index=False
    )
)


print("\nFiles created:")

print(summary_file)
print(correlation_file)
print(high_corr_file)
print(pair_file)