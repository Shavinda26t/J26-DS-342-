from pathlib import Path
import pandas as pd


# ============================================================
# Component 2 - Digital Twin
# Health-State Feature Distribution Analysis
# ============================================================

INPUT_FILE = Path(
    "results/candidate_health_state_features.csv"
)

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = (
    OUTPUT_DIR /
    "health_state_feature_distributions.csv"
)


FEATURES = [
    "smart_1_normalized",
    "smart_1_normalized_change_7d",
    "smart_1_normalized_rolling_std_7",

    "smart_5_normalized",
    "smart_5_normalized_change_7d",

    "smart_187_normalized",
    "smart_187_normalized_change_7d",
    "smart_187_normalized_rolling_std_7",

    "smart_197_normalized",
    "smart_197_normalized_change_7d",

    "smart_198_normalized",
    "smart_198_normalized_change_7d",
]


def main():

    print("=" * 70)
    print("COMPONENT 2 - HEALTH-STATE FEATURE DISTRIBUTION ANALYSIS")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    print(f"\nInput: {INPUT_FILE}")

    # --------------------------------------------------------
    # Check columns
    # --------------------------------------------------------

    header = pd.read_csv(
        INPUT_FILE,
        nrows=0
    )

    available = [
        feature
        for feature in FEATURES
        if feature in header.columns
    ]

    missing = [
        feature
        for feature in FEATURES
        if feature not in header.columns
    ]

    print(
        f"\nRequested features: {len(FEATURES)}"
    )

    print(
        f"Available features: {len(available)}"
    )

    if missing:

        print("\nMissing columns:")

        for feature in missing:
            print(f"  - {feature}")

    # --------------------------------------------------------
    # Read required features
    # --------------------------------------------------------

    print("\nLoading feature data...")

    data = pd.read_csv(
        INPUT_FILE,
        usecols=available
    )

    print(
        f"Rows: {len(data):,}"
    )

    # --------------------------------------------------------
    # Calculate distribution statistics
    # --------------------------------------------------------

    rows = []

    for feature in available:

        series = data[feature]

        valid = series.dropna()

        if len(valid) == 0:

            continue

        rows.append(
            {
                "feature": feature,

                "count": len(valid),

                "missing_percentage":
                    series.isna().mean() * 100,

                "min":
                    valid.min(),

                "p01":
                    valid.quantile(0.01),

                "p05":
                    valid.quantile(0.05),

                "p25":
                    valid.quantile(0.25),

                "median":
                    valid.median(),

                "p75":
                    valid.quantile(0.75),

                "p95":
                    valid.quantile(0.95),

                "p99":
                    valid.quantile(0.99),

                "max":
                    valid.max(),

                "mean":
                    valid.mean(),

                "std":
                    valid.std(),

                "mean_absolute":
                    valid.abs().mean(),

                "negative_percentage":
                    (valid < 0).mean() * 100,

                "zero_percentage":
                    (valid == 0).mean() * 100,

                "positive_percentage":
                    (valid > 0).mean() * 100,
            }
        )

    summary = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    summary.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DISTRIBUTION ANALYSIS COMPLETED")
    print("=" * 70)

    print(
        f"\nOutput: {OUTPUT_FILE}"
    )

    print("\nFeature distributions:")

    display_columns = [
        "feature",
        "missing_percentage",
        "min",
        "p01",
        "median",
        "p99",
        "max",
        "mean",
        "std",
        "negative_percentage",
        "zero_percentage",
        "positive_percentage",
    ]

    print(
        summary[
            display_columns
        ].to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("INTERPRETATION CHECK")
    print("=" * 70)

    print(
        """
The results will be used to determine:

1. Whether current SMART values require scaling.
2. Whether temporal-change features contain both
   positive and negative values.
3. Whether extreme values require robust scaling.
4. Whether different SMART attributes require
   different transformation directions.
5. Whether missing values need to be handled
   before health-state construction.

No final health score is calculated at this stage.
"""
    )


if __name__ == "__main__":
    main()