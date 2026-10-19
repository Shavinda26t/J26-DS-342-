import os
import numpy as np
import pandas as pd


# ================================================================
# CONFIGURATION
# ================================================================

INPUT_FILE = r".\data\interim\temporal_features.csv"

OUTPUT_FILE = r".\results\digital_twin_state_v2.csv"

CHUNK_SIZE = 500_000


# ================================================================
# SMART FEATURES
# ================================================================

SMART_FEATURES = [
    "smart_1_normalized",
    "smart_5_normalized",
    "smart_187_normalized",
    "smart_197_normalized",
    "smart_198_normalized",
]


# ================================================================
# 7-DAY TEMPORAL FEATURES
# ================================================================

CHANGE_FEATURES = [
    "smart_1_normalized_change_7d",
    "smart_5_normalized_change_7d",
    "smart_187_normalized_change_7d",
    "smart_197_normalized_change_7d",
    "smart_198_normalized_change_7d",
]


# ================================================================
# LOWER HEALTH REFERENCES
#
# Exploratory transformation:
#
# SMART = 100 -> health = 1
# P05 of below-100 values -> health = 0
#
# These values came from the full-Q1 analysis.
#
# This is NOT the final validated health index.
# ================================================================

LOWER_REFERENCE = {
    "smart_1_normalized": 72.0,
    "smart_5_normalized": 60.0,
    "smart_187_normalized": 34.0,
    "smart_197_normalized": 83.0,
    "smart_198_normalized": 1.0,
}


# ================================================================
# SMART -> HEALTH
# ================================================================

def smart_to_health(value, lower_reference):

    if pd.isna(value):
        return np.nan

    healthy_reference = 100.0

    health = (
        (value - lower_reference)
        /
        (healthy_reference - lower_reference)
    )

    return np.clip(
        health,
        0.0,
        1.0
    )


# ================================================================
# MAIN
# ================================================================

def main():

    print("=" * 70)
    print("DIGITAL TWIN STATE V2 CONSTRUCTION")
    print("=" * 70)

    if not os.path.exists(INPUT_FILE):

        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    # ------------------------------------------------------------
    # Remove old output
    # ------------------------------------------------------------

    if os.path.exists(OUTPUT_FILE):

        os.remove(
            OUTPUT_FILE
        )

    # ------------------------------------------------------------
    # Only request columns that actually exist in
    # temporal_features.csv
    # ------------------------------------------------------------

    required_columns = [
        "date",
        "serial_number",
        "failure",
    ] + SMART_FEATURES + CHANGE_FEATURES

    first_chunk = True

    total_rows = 0

    # ------------------------------------------------------------
    # Process dataset
    # ------------------------------------------------------------

    for chunk_number, df in enumerate(
        pd.read_csv(
            INPUT_FILE,
            usecols=required_columns,
            chunksize=CHUNK_SIZE
        ),
        start=1
    ):

        print(
            f"Processing chunk {chunk_number}..."
        )

        total_rows += len(df)

        # ========================================================
        # CURRENT HEALTH STATE
        # ========================================================

        for feature in SMART_FEATURES:

            health_name = feature.replace(
                "_normalized",
                "_health"
            )

            lower_reference = LOWER_REFERENCE[
                feature
            ]

            values = pd.to_numeric(
                df[feature],
                errors="coerce"
            )

            # Vectorised transformation
            health = (
                (values - lower_reference)
                /
                (100.0 - lower_reference)
            )

            health = health.clip(
                lower=0.0,
                upper=1.0
            )

            # Preserve missing observations
            health.loc[
                values.isna()
            ] = np.nan

            df[health_name] = health

        # ========================================================
        # AVAILABILITY STATE
        # ========================================================
        #
        # Availability is derived directly from whether the
        # corresponding SMART value exists.
        # ========================================================

        availability_columns = []

        for feature in SMART_FEATURES:

            availability_name = (
                feature.replace(
                    "_normalized",
                    "_available"
                )
            )

            available = (
                df[feature]
                .notna()
                .astype("int8")
            )

            df[availability_name] = available

            availability_columns.append(
                availability_name
            )

        # ========================================================
        # TOTAL AVAILABLE HEALTH FEATURES
        # ========================================================

        df[
            "health_features_available"
        ] = df[
            availability_columns
        ].sum(
            axis=1
        )

        # ========================================================
        # OUTPUT COLUMNS
        # ========================================================

        output_columns = [
            "date",
            "serial_number",
            "failure",

            # ----------------------------------------------------
            # Current health state
            # ----------------------------------------------------

            "smart_1_health",
            "smart_5_health",
            "smart_187_health",
            "smart_197_health",
            "smart_198_health",

            # ----------------------------------------------------
            # Temporal degradation state
            # ----------------------------------------------------

            "smart_1_normalized_change_7d",
            "smart_5_normalized_change_7d",
            "smart_187_normalized_change_7d",
            "smart_197_normalized_change_7d",
            "smart_198_normalized_change_7d",

            # ----------------------------------------------------
            # Availability state
            # ----------------------------------------------------

            "smart_1_available",
            "smart_5_available",
            "smart_187_available",
            "smart_197_available",
            "smart_198_available",

            "health_features_available",
        ]

        output_df = df[
            output_columns
        ]

        # ========================================================
        # WRITE OUTPUT
        # ========================================================

        if first_chunk:

            output_df.to_csv(
                OUTPUT_FILE,
                index=False
            )

            first_chunk = False

        else:

            output_df.to_csv(
                OUTPUT_FILE,
                index=False,
                mode="a",
                header=False
            )

    # ============================================================
    # COMPLETE
    # ============================================================

    print("\n")
    print("=" * 70)
    print("DIGITAL TWIN STATE V2 COMPLETE")
    print("=" * 70)

    print(
        f"\nTotal rows processed: "
        f"{total_rows:,}"
    )

    print(
        f"\nOutput file:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        "\nState dimensions: 15"
    )

    print(
        "\n5 current health states"
    )

    print(
        "5 temporal degradation states"
    )

    print(
        "5 observation availability states"
    )


if __name__ == "__main__":
    main()