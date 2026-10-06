import os
import numpy as np
import pandas as pd


# ================================================================
# CONFIGURATION
# ================================================================

INPUT_FILE = r".\results\candidate_health_state_features.csv"

OUTPUT_FILE = r".\results\normalized_health_state_features_v2.csv"

CHUNK_SIZE = 500_000


# ================================================================
# FEATURES
# ================================================================

HEALTH_FEATURES = [
    "smart_1_normalized",
    "smart_5_normalized",
    "smart_187_normalized",
    "smart_197_normalized",
    "smart_198_normalized",
    "smart_1_normalized_change_7d",
    "smart_5_normalized_change_7d",
    "smart_187_normalized_change_7d",
    "smart_197_normalized_change_7d",
    "smart_198_normalized_change_7d",
]


# ================================================================
# AVAILABILITY FEATURES
# ================================================================

AVAILABILITY_FEATURES = [
    "smart_1_available",
    "smart_5_available",
    "smart_187_available",
    "smart_197_available",
    "smart_198_available",
]


# ================================================================
# OTHER COLUMNS
# ================================================================

IDENTIFIER_COLUMNS = [
    "date",
    "serial_number",
    "failure",
]


# ================================================================
# STEP 1
# CALCULATE GLOBAL PERCENTILES
#
# This is exploratory normalization only.
#
# Final predictive preprocessing must calculate these values
# using training data only.
# ================================================================

def calculate_percentiles():

    print("=" * 70)
    print("STEP 1: CALCULATING EXPLORATORY P01/P99 VALUES")
    print("=" * 70)

    # ------------------------------------------------------------
    # Collect sampled values rather than loading the entire
    # 21-million-row dataset into memory.
    # ------------------------------------------------------------

    samples = {
        feature: []
        for feature in HEALTH_FEATURES
    }

    total_rows = 0

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            INPUT_FILE,
            usecols=HEALTH_FEATURES,
            chunksize=CHUNK_SIZE
        ),
        start=1
    ):

        print(
            f"Reading chunk {chunk_number}..."
        )

        total_rows += len(chunk)

        for feature in HEALTH_FEATURES:

            values = pd.to_numeric(
                chunk[feature],
                errors="coerce"
            ).dropna().to_numpy()

            if len(values) > 0:

                # Sample at most 20,000 values per chunk.
                if len(values) > 20_000:

                    rng = np.random.default_rng(
                        seed=42 + chunk_number
                    )

                    values = rng.choice(
                        values,
                        size=20_000,
                        replace=False
                    )

                samples[feature].append(
                    values
                )

    print(
        f"\nRows inspected: {total_rows:,}"
    )

    percentiles = {}

    print("\nCalculated ranges:\n")

    for feature in HEALTH_FEATURES:

        if not samples[feature]:

            percentiles[feature] = {
                "p01": np.nan,
                "p99": np.nan,
            }

            print(
                f"{feature}: NO VALID DATA"
            )

            continue

        values = np.concatenate(
            samples[feature]
        )

        p01 = np.percentile(
            values,
            1
        )

        p99 = np.percentile(
            values,
            99
        )

        percentiles[feature] = {
            "p01": p01,
            "p99": p99,
        }

        print(
            f"{feature}: "
            f"P01={p01:.6f}, "
            f"P99={p99:.6f}"
        )

    return percentiles


# ================================================================
# STEP 2
# NORMALIZE
# ================================================================

def normalize_data(percentiles):

    print("\n")
    print("=" * 70)
    print("STEP 2: BUILDING NORMALIZED HEALTH-STATE DATASET")
    print("=" * 70)

    if os.path.exists(OUTPUT_FILE):

        os.remove(
            OUTPUT_FILE
        )

    first_chunk = True

    total_rows = 0

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            INPUT_FILE,
            chunksize=CHUNK_SIZE
        ),
        start=1
    ):

        print(
            f"Processing chunk {chunk_number}..."
        )

        total_rows += len(chunk)

        # --------------------------------------------------------
        # Normalize health features
        # --------------------------------------------------------

        for feature in HEALTH_FEATURES:

            if feature not in chunk.columns:
                continue

            p01 = percentiles[feature]["p01"]
            p99 = percentiles[feature]["p99"]

            values = pd.to_numeric(
                chunk[feature],
                errors="coerce"
            )

            # ----------------------------------------------------
            # If P01 and P99 are identical, the feature has no
            # useful variation for this exploratory transformation.
            # ----------------------------------------------------

            if (
                pd.isna(p01)
                or pd.isna(p99)
                or p99 <= p01
            ):

                chunk[
                    feature + "_scaled"
                ] = np.nan

                continue

            # ----------------------------------------------------
            # Robust percentile scaling:
            #
            # P01 -> 0
            # P99 -> 1
            #
            # Values outside range are clipped.
            # Missing values remain missing.
            # ----------------------------------------------------

            scaled = (
                (values - p01)
                /
                (p99 - p01)
            )

            scaled = scaled.clip(
                lower=0.0,
                upper=1.0
            )

            chunk[
                feature + "_scaled"
            ] = scaled

        # --------------------------------------------------------
        # Keep original identifiers and availability indicators.
        #
        # Remove original health feature columns so the output
        # clearly contains the scaled representations.
        # --------------------------------------------------------

        output_columns = (
            IDENTIFIER_COLUMNS
            +
            [
                feature + "_scaled"
                for feature in HEALTH_FEATURES
            ]
            +
            [
                feature
                for feature in AVAILABILITY_FEATURES
                if feature in chunk.columns
            ]
        )

        output_chunk = chunk[
            [
                column
                for column in output_columns
                if column in chunk.columns
            ]
        ]

        # --------------------------------------------------------
        # Write
        # --------------------------------------------------------

        if first_chunk:

            output_chunk.to_csv(
                OUTPUT_FILE,
                index=False
            )

            first_chunk = False

        else:

            output_chunk.to_csv(
                OUTPUT_FILE,
                index=False,
                mode="a",
                header=False
            )

    print("\n")
    print("=" * 70)
    print("NORMALIZATION COMPLETE")
    print("=" * 70)

    print(
        f"\nRows processed: {total_rows:,}"
    )

    print(
        f"\nOutput:\n{OUTPUT_FILE}"
    )


# ================================================================
# MAIN
# ================================================================

def main():

    if not os.path.exists(INPUT_FILE):

        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    percentiles = calculate_percentiles()

    normalize_data(
        percentiles
    )


if __name__ == "__main__":
    main()