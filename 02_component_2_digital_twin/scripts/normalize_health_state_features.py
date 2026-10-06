from pathlib import Path
import pandas as pd


# ============================================================
# Component 2 - Digital Twin
# Robust Normalisation of Health-State Features
# ============================================================

INPUT_FILE = Path(
    "results/candidate_health_state_features.csv"
)

DISTRIBUTION_FILE = Path(
    "results/health_state_feature_distributions.csv"
)

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = (
    OUTPUT_DIR /
    "normalized_health_state_features.csv"
)


BASE_COLUMNS = [
    "date",
    "serial_number",
    "failure",
]


HEALTH_FEATURES = [
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


AVAILABILITY_COLUMNS = [
    "smart_187_available",
    "smart_1_available",
    "smart_5_available",
    "smart_197_available",
    "smart_198_available",
]


def robust_scale(series, lower, upper):
    """
    Percentile-based scaling.

    Values below the lower percentile become 0.
    Values above the upper percentile become 1.

    Missing values remain NaN.
    """

    denominator = upper - lower

    if denominator <= 0:
        return pd.Series(
            0.5,
            index=series.index,
            dtype="float32"
        )

    scaled = (
        (series - lower) /
        denominator
    )

    return scaled.clip(
        lower=0.0,
        upper=1.0
    )


def main():

    print("=" * 70)
    print("COMPONENT 2 - ROBUST HEALTH-STATE NORMALISATION")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    if not DISTRIBUTION_FILE.exists():
        raise FileNotFoundError(
            f"Distribution file not found: "
            f"{DISTRIBUTION_FILE}"
        )

    # --------------------------------------------------------
    # Load distribution statistics
    # --------------------------------------------------------

    distributions = pd.read_csv(
        DISTRIBUTION_FILE
    )

    if "feature" not in distributions.columns:
        raise ValueError(
            "Distribution file does not contain "
            "'feature' column."
        )

    distribution_lookup = (
        distributions
        .set_index("feature")
    )

    print(
        f"\nDistribution records: "
        f"{len(distributions)}"
    )

    # --------------------------------------------------------
    # Check required features
    # --------------------------------------------------------

    header = pd.read_csv(
        INPUT_FILE,
        nrows=0
    )

    available_columns = set(
        header.columns
    )

    required_columns = (
        BASE_COLUMNS
        + HEALTH_FEATURES
        + AVAILABILITY_COLUMNS
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in available_columns
    ]

    if missing_columns:

        print("\nMissing columns:")

        for column in missing_columns:
            print(f"  - {column}")

        raise ValueError(
            "Required columns are missing."
        )

    # --------------------------------------------------------
    # Determine percentile bounds
    # --------------------------------------------------------

    bounds = {}

    for feature in HEALTH_FEATURES:

        if feature not in distribution_lookup.index:
            raise ValueError(
                f"No distribution statistics found "
                f"for {feature}"
            )

        lower = distribution_lookup.loc[
            feature,
            "p01"
        ]

        upper = distribution_lookup.loc[
            feature,
            "p99"
        ]

        bounds[feature] = (
            float(lower),
            float(upper)
        )

    print("\nUsing percentile-based bounds:")

    for feature, (lower, upper) in bounds.items():

        print(
            f"{feature}: "
            f"P01={lower:.4f}, "
            f"P99={upper:.4f}"
        )

    # --------------------------------------------------------
    # Process dataset in chunks
    # --------------------------------------------------------

    chunksize = 500_000

    first_chunk = True
    total_rows = 0

    use_columns = (
        BASE_COLUMNS
        + HEALTH_FEATURES
        + AVAILABILITY_COLUMNS
    )

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            INPUT_FILE,
            usecols=use_columns,
            chunksize=chunksize
        ),
        start=1
    ):

        total_rows += len(chunk)

        print(
            f"Processing chunk {chunk_number}: "
            f"{len(chunk):,} rows"
        )

        # ----------------------------------------------------
        # Create normalized versions
        # ----------------------------------------------------

        for feature in HEALTH_FEATURES:

            lower, upper = bounds[feature]

            output_column = (
                f"{feature}_scaled"
            )

            chunk[output_column] = robust_scale(
                chunk[feature],
                lower,
                upper
            ).astype("float32")

        # ----------------------------------------------------
        # Preserve availability information
        # ----------------------------------------------------

        # Availability is already represented as 0/1.
        # We do not scale these indicators.
        #
        # They explicitly tell the Digital Twin whether
        # an observation was available.
        #
        # Missing SMART values remain NaN in the scaled
        # feature itself.
        # ----------------------------------------------------

        chunk.to_csv(
            OUTPUT_FILE,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False
        )

        first_chunk = False

    print("\n" + "=" * 70)
    print("ROBUST NORMALISATION COMPLETED")
    print("=" * 70)

    print(
        f"\nOutput: {OUTPUT_FILE}"
    )

    print(
        f"Total rows: {total_rows:,}"
    )

    print(
        "\nScaling method:"
    )

    print(
        "P01 -> 0"
    )

    print(
        "P99 -> 1"
    )

    print(
        "Values outside the percentile range "
        "are clipped to [0, 1]."
    )

    print(
        "\nMissing values remain missing."
    )

    print(
        "Availability indicators are preserved."
    )

    print(
        "\nNo final health score was calculated."
    )


if __name__ == "__main__":
    main()