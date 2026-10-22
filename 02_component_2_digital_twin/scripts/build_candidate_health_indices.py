import os
import numpy as np
import pandas as pd


# ================================================================
# CONFIGURATION
# ================================================================

INPUT_FILE = r".\results\normalized_health_state_features.csv"

OUTPUT_FILE = r".\results\candidate_health_indices.csv"
SUMMARY_FILE = r".\results\candidate_health_index_summary.csv"

CHUNK_SIZE = 500_000


# ================================================================
# FEATURES
# ================================================================

CURRENT_FEATURES = [
    "smart_1_normalized",
    "smart_5_normalized",
    "smart_187_normalized",
    "smart_197_normalized",
    "smart_198_normalized",
]

CHANGE_FEATURES = [
    "smart_1_normalized_change_7d",
    "smart_5_normalized_change_7d",
    "smart_187_normalized_change_7d",
    "smart_197_normalized_change_7d",
    "smart_198_normalized_change_7d",
]


# ================================================================
# EVIDENCE-BASED WEIGHTS
# ================================================================
#
# These weights were derived from the exploratory failed-vs-healthy
# comparison at approximately 7 days before failure.
#
# They are NOT considered final model weights yet.
# They are being used to construct candidate health indices.
#

CURRENT_WEIGHTS = {
    "smart_1_normalized": 0.149699,
    "smart_5_normalized": 0.103773,
    "smart_187_normalized": 1.110968,
    "smart_197_normalized": 0.015804,
    "smart_198_normalized": 0.014146,
}


CHANGE_WEIGHTS = {
    "smart_1_normalized_change_7d": 0.196857,
    "smart_5_normalized_change_7d": 0.160298,
    "smart_187_normalized_change_7d": 0.514949,
    "smart_197_normalized_change_7d": 0.130358,
    "smart_198_normalized_change_7d": 0.130447,
}


# ================================================================
# NORMALISE WEIGHTS
# ================================================================

def normalize_weights(weights):
    """
    Convert raw evidence weights into weights that sum to 1.
    """
    total = sum(weights.values())

    if total <= 0:
        raise ValueError("Weight total must be greater than zero.")

    return {
        key: value / total
        for key, value in weights.items()
    }


CURRENT_WEIGHTS = normalize_weights(CURRENT_WEIGHTS)
CHANGE_WEIGHTS = normalize_weights(CHANGE_WEIGHTS)


# ================================================================
# WEIGHTED ROW MEAN
# ================================================================

def weighted_row_mean(df, features, weights):
    """
    Calculate a weighted mean for each row while handling missing
    observations.

    Only available features contribute to the weighted average.
    The weights are renormalised over the available features.

    Returns:
        health index
        number of available features
    """

    values = df[features].to_numpy(dtype=float)

    weight_array = np.array(
        [weights[feature] for feature in features],
        dtype=float
    )

    valid = ~np.isnan(values)

    weighted_values = np.where(
        valid,
        values * weight_array,
        0.0
    )

    weight_sum = np.sum(
        np.where(valid, weight_array, 0.0),
        axis=1
    )

    value_sum = np.sum(
        weighted_values,
        axis=1
    )

    result = np.full(
        len(df),
        np.nan,
        dtype=float
    )

    valid_rows = weight_sum > 0

    result[valid_rows] = (
        value_sum[valid_rows] /
        weight_sum[valid_rows]
    )

    available_count = np.sum(
        valid,
        axis=1
    )

    return result, available_count


# ================================================================
# MAIN
# ================================================================

def main():

    print("=" * 70)
    print("CANDIDATE DIGITAL TWIN HEALTH INDEX CONSTRUCTION")
    print("=" * 70)

    print("\nEvidence-based current-state weights:")

    for feature, weight in CURRENT_WEIGHTS.items():
        print(
            f"  {feature}: {weight:.4f}"
        )

    print("\nEvidence-based degradation weights:")

    for feature, weight in CHANGE_WEIGHTS.items():
        print(
            f"  {feature}: {weight:.4f}"
        )

    print("\n")

    # ------------------------------------------------------------
    # Check input
    # ------------------------------------------------------------

    if not os.path.exists(INPUT_FILE):

        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    # ------------------------------------------------------------
    # Remove previous output files
    # ------------------------------------------------------------

    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    if os.path.exists(SUMMARY_FILE):
        os.remove(SUMMARY_FILE)

    # ------------------------------------------------------------
    # Read required columns
    # ------------------------------------------------------------

    required_columns = [
        "date",
        "serial_number",
        "failure",
    ]

    required_columns.extend(CURRENT_FEATURES)
    required_columns.extend(CHANGE_FEATURES)

    # Availability columns may already exist in the input.
    availability_features = [
        "smart_1_available",
        "smart_5_available",
        "smart_187_available",
        "smart_197_available",
        "smart_198_available",
    ]

    # ------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------

    summary_records = []

    total_rows = 0

    # Candidate statistics
    candidate_names = [
        "health_index_A_equal",
        "health_index_B_evidence",
        "health_index_C_dynamic",
    ]

    candidate_values = {
        name: []
        for name in candidate_names
    }

    available_feature_counts = []

    # ------------------------------------------------------------
    # Process input in chunks
    # ------------------------------------------------------------

    first_chunk = True

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            INPUT_FILE,
            usecols=lambda column: column in required_columns
            or column in availability_features,
            chunksize=CHUNK_SIZE
        ),
        start=1
    ):

        print(
            f"Processing chunk {chunk_number}..."
        )

        total_rows += len(chunk)

        # --------------------------------------------------------
        # Convert feature columns to numeric
        # --------------------------------------------------------

        for feature in CURRENT_FEATURES + CHANGE_FEATURES:

            if feature in chunk.columns:

                chunk[feature] = pd.to_numeric(
                    chunk[feature],
                    errors="coerce"
                )

        # ========================================================
        # CANDIDATE A
        # ========================================================
        #
        # Equal-weight current health state.
        #
        # Every available SMART feature contributes equally.
        #

        current_values = chunk[
            CURRENT_FEATURES
        ].to_numpy(dtype=float)

        current_valid = ~np.isnan(
            current_values
        )

        current_sum = np.nansum(
            current_values,
            axis=1
        )

        current_count = np.sum(
            current_valid,
            axis=1
        )

        health_index_A = np.full(
            len(chunk),
            np.nan,
            dtype=float
        )

        valid_A = current_count > 0

        health_index_A[valid_A] = (
            current_sum[valid_A] /
            current_count[valid_A]
        )

        # ========================================================
        # CANDIDATE B
        # ========================================================
        #
        # Evidence-weighted current health state.
        #

        health_index_B, available_count = weighted_row_mean(
            chunk,
            CURRENT_FEATURES,
            CURRENT_WEIGHTS
        )

        # ========================================================
        # DYNAMIC DEGRADATION COMPONENT
        # ========================================================

        weighted_change, _ = weighted_row_mean(
            chunk,
            CHANGE_FEATURES,
            CHANGE_WEIGHTS
        )

        # --------------------------------------------------------
        # Convert negative change into degradation magnitude.
        #
        # Negative SMART change = deterioration signal.
        #
        # Positive change is not treated as degradation.
        # --------------------------------------------------------

        degradation_penalty = np.clip(
            -weighted_change,
            0.0,
            20.0
        ) / 20.0

        # --------------------------------------------------------
        # Candidate C
        #
        # Evidence-weighted current health
        # minus dynamic degradation penalty.
        #
        # 0.30 is intentionally an experimental coefficient.
        # --------------------------------------------------------

        health_index_C = (
            health_index_B -
            (0.30 * degradation_penalty)
        )

        # --------------------------------------------------------
        # Ensure the final index remains in [0,1].
        #
        # IMPORTANT:
        # np.clip uses a_min and a_max.
        # --------------------------------------------------------

        health_index_C = np.clip(
            health_index_C,
            a_min=0.0,
            a_max=1.0
        )

        # --------------------------------------------------------
        # Store candidate values
        # --------------------------------------------------------

        candidate_values[
            "health_index_A_equal"
        ].extend(
            health_index_A[
                np.isfinite(health_index_A)
            ].tolist()
        )

        candidate_values[
            "health_index_B_evidence"
        ].extend(
            health_index_B[
                np.isfinite(health_index_B)
            ].tolist()
        )

        candidate_values[
            "health_index_C_dynamic"
        ].extend(
            health_index_C[
                np.isfinite(health_index_C)
            ].tolist()
        )

        available_feature_counts.extend(
            available_count.tolist()
        )

        # --------------------------------------------------------
        # Add candidate columns to chunk
        # --------------------------------------------------------

        chunk[
            "health_index_A_equal"
        ] = health_index_A

        chunk[
            "health_index_B_evidence"
        ] = health_index_B

        chunk[
            "health_index_C_dynamic"
        ] = health_index_C

        chunk[
            "health_features_available"
        ] = available_count

        chunk[
            "weighted_degradation_change"
        ] = weighted_change

        chunk[
            "degradation_penalty"
        ] = degradation_penalty

        # --------------------------------------------------------
        # Write output
        # --------------------------------------------------------

        if first_chunk:

            chunk.to_csv(
                OUTPUT_FILE,
                index=False,
                mode="w"
            )

            first_chunk = False

        else:

            chunk.to_csv(
                OUTPUT_FILE,
                index=False,
                mode="a",
                header=False
            )

    # ============================================================
    # SUMMARY
    # ============================================================

    print("\n")
    print("=" * 70)
    print("CREATING SUMMARY")
    print("=" * 70)

    summary_rows = []

    for candidate_name in candidate_names:

        values = np.asarray(
            candidate_values[candidate_name],
            dtype=float
        )

        if len(values) == 0:

            summary_rows.append({
                "candidate": candidate_name,
                "count": 0,
                "mean": np.nan,
                "median": np.nan,
                "std": np.nan,
                "min": np.nan,
                "p01": np.nan,
                "p99": np.nan,
                "max": np.nan,
            })

            continue

        summary_rows.append({
            "candidate": candidate_name,
            "count": len(values),
            "mean": np.mean(values),
            "median": np.median(values),
            "std": np.std(values),
            "min": np.min(values),
            "p01": np.percentile(values, 1),
            "p99": np.percentile(values, 99),
            "max": np.max(values),
        })

    # ------------------------------------------------------------
    # Availability summary
    # ------------------------------------------------------------

    available_counts = np.asarray(
        available_feature_counts,
        dtype=int
    )

    availability_summary = {
        "candidate": "health_features_available",
        "count": len(available_counts),
        "mean": np.mean(available_counts),
        "median": np.median(available_counts),
        "std": np.std(available_counts),
        "min": np.min(available_counts),
        "p01": np.percentile(available_counts, 1),
        "p99": np.percentile(available_counts, 99),
        "max": np.max(available_counts),
    }

    summary_rows.append(
        availability_summary
    )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False
    )

    # ============================================================
    # FINAL OUTPUT
    # ============================================================

    print("\n")
    print("=" * 70)
    print("HEALTH INDEX CONSTRUCTION COMPLETE")
    print("=" * 70)

    print(
        f"\nTotal rows processed: {total_rows:,}"
    )

    print(
        f"\nCandidate output:"
        f"\n{OUTPUT_FILE}"
    )

    print(
        f"\nSummary output:"
        f"\n{SUMMARY_FILE}"
    )

    print("\nCandidate summary:\n")

    print(
        summary_df.to_string(
            index=False
        )
    )

    print("\n")
    print("=" * 70)
    print("IMPORTANT METHODOLOGICAL NOTE")
    print("=" * 70)

    print(
        "\nCandidate A = equal-weight current health."
    )

    print(
        "Candidate B = evidence-weighted current health."
    )

    print(
        "Candidate C = evidence-weighted health + degradation dynamics."
    )

    print(
        "\nThese are exploratory candidate formulations."
    )

    print(
        "The final health index should be selected only after"
        "\nchronological validation on held-out data."
    )


# ================================================================
# RUN
# ================================================================

if __name__ == "__main__":
    main()