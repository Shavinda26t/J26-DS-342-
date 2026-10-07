import os
import numpy as np
import pandas as pd


# ============================================================
# Adaptive Digital Twin - Degradation Change Detection
# ============================================================
#
# Purpose:
# Detect unusually strong degradation behaviour in the
# Digital Twin state using robust population statistics.
#
# This is an EXPERIMENTAL adaptive-state detector.
# It is not yet a validated health/failure classifier.
#
# ============================================================


# ============================================================
# Paths
# ============================================================

INPUT_FILE = r".\results\digital_twin_state_v5.csv"

OUTPUT_DIR = r".\results\adaptive_digital_twin"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# Configuration
# ============================================================

CHUNK_SIZE = 500_000

TEMPORAL_FEATURES = [
    "smart_1_normalized_change_7d",
    "smart_1_normalized_change_14d",
    "smart_187_normalized_change_7d",
    "smart_187_normalized_change_14d"
]

AVAILABILITY_FEATURES = [
    "smart_1_available",
    "smart_187_available"
]


# ============================================================
# Robust statistics
# ============================================================

def robust_statistics(values):

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:

        return np.nan, np.nan

    median = np.median(values)

    mad = np.median(
        np.abs(
            values - median
        )
    )

    return median, mad


# ============================================================
# Robust Z-score
# ============================================================

def robust_z_score(
    values,
    median,
    mad
):

    scale = 1.4826 * mad

    if (
        not np.isfinite(scale)
        or scale < 1e-8
    ):

        return np.zeros_like(
            values,
            dtype=np.float64
        )

    return (
        values - median
    ) / scale


# ============================================================
# PASS 1
# Estimate robust population baselines
# ============================================================

print("=" * 70)
print("PASS 1: ESTIMATING ROBUST DEGRADATION BASELINES")
print("=" * 70)

feature_values = {
    feature: []
    for feature in TEMPORAL_FEATURES
}

total_rows = 0


for chunk in pd.read_csv(
    INPUT_FILE,
    usecols=TEMPORAL_FEATURES,
    chunksize=CHUNK_SIZE
):

    total_rows += len(chunk)

    for feature in TEMPORAL_FEATURES:

        values = pd.to_numeric(
            chunk[feature],
            errors="coerce"
        ).dropna().values

        if len(values) == 0:
            continue

        current_count = len(
            feature_values[feature]
        )

        remaining = (
            500_000
            -
            current_count
        )

        if remaining > 0:

            feature_values[feature].extend(
                values[:remaining]
            )


print(
    "\nRows scanned:",
    total_rows
)


# ============================================================
# Calculate robust statistics
# ============================================================

baseline_stats = []


for feature in TEMPORAL_FEATURES:

    values = np.asarray(
        feature_values[feature],
        dtype=np.float64
    )

    median, mad = robust_statistics(
        values
    )

    baseline_stats.append({

        "feature": feature,

        "median": median,

        "MAD": mad,

        "robust_scale":
            (
                1.4826 * mad
                if np.isfinite(mad)
                else np.nan
            ),

        "sample_count":
            len(values)
    })


baseline_df = pd.DataFrame(
    baseline_stats
)


print("\nROBUST BASELINES")

print(
    baseline_df.to_string(
        index=False
    )
)


baseline_file = os.path.join(
    OUTPUT_DIR,
    "robust_degradation_baselines.csv"
)

baseline_df.to_csv(
    baseline_file,
    index=False
)


# ============================================================
# Convert baseline statistics to dictionaries
# ============================================================

medians = dict(
    zip(
        baseline_df["feature"],
        baseline_df["median"]
    )
)

mads = dict(
    zip(
        baseline_df["feature"],
        baseline_df["MAD"]
    )
)


# ============================================================
# PASS 2
# Build adaptive degradation state
# ============================================================

print("\n" + "=" * 70)
print("PASS 2: BUILDING ADAPTIVE DEGRADATION INDICATORS")
print("=" * 70)


output_file = os.path.join(
    OUTPUT_DIR,
    "adaptive_degradation_state.csv"
)


# Remove previous output if it exists
if os.path.exists(output_file):

    os.remove(output_file)


first_write = True


# ============================================================
# Process Digital Twin state in chunks
# ============================================================

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


    # --------------------------------------------------------
    # Convert temporal features to numeric
    # --------------------------------------------------------

    for feature in TEMPORAL_FEATURES:

        chunk[feature] = pd.to_numeric(
            chunk[feature],
            errors="coerce"
        )


    # --------------------------------------------------------
    # SMART 1 - 7 day degradation
    # --------------------------------------------------------

    smart1_change_7 = (
        chunk[
            "smart_1_normalized_change_7d"
        ]
    )

    smart1_filled = (
        smart1_change_7.fillna(
            medians[
                "smart_1_normalized_change_7d"
            ]
        )
    )

    smart1_z = robust_z_score(
        smart1_filled.values,
        medians[
            "smart_1_normalized_change_7d"
        ],
        mads[
            "smart_1_normalized_change_7d"
        ]
    )


    # Negative change means deterioration
    smart1_degradation = np.maximum(
        -smart1_z,
        0
    )


    # --------------------------------------------------------
    # SMART 187 - 7 day degradation
    # --------------------------------------------------------

    smart187_change_7 = (
        chunk[
            "smart_187_normalized_change_7d"
        ]
    )

    smart187_filled = (
        smart187_change_7.fillna(
            medians[
                "smart_187_normalized_change_7d"
            ]
        )
    )

    smart187_z = robust_z_score(
        smart187_filled.values,
        medians[
            "smart_187_normalized_change_7d"
        ],
        mads[
            "smart_187_normalized_change_7d"
        ]
    )


    # Negative change means deterioration
    smart187_degradation = np.maximum(
        -smart187_z,
        0
    )


    # --------------------------------------------------------
    # Availability
    # --------------------------------------------------------

    smart1_available = (
        chunk[
            "smart_1_available"
        ]
        .fillna(0)
        .astype(int)
        .values
    )

    smart187_available = (
        chunk[
            "smart_187_available"
        ]
        .fillna(0)
        .astype(int)
        .values
    )


    # --------------------------------------------------------
    # Missing observations
    #
    # Missing SMART data does NOT mean healthy.
    # It simply provides no degradation evidence.
    # --------------------------------------------------------

    smart1_degradation[
        smart1_available == 0
    ] = np.nan

    smart187_degradation[
        smart187_available == 0
    ] = np.nan


    # --------------------------------------------------------
    # Combine available degradation evidence
    # --------------------------------------------------------

    degradation_matrix = np.column_stack(
        [
            smart1_degradation,
            smart187_degradation
        ]
    )

    combined_score = np.nanmean(
        degradation_matrix,
        axis=1
    )


    # np.nanmean produces a warning when
    # both values are missing.
    # Those rows contain no degradation evidence.

    no_evidence = np.isnan(
        combined_score
    )

    combined_score[
        no_evidence
    ] = 0.0


    # --------------------------------------------------------
    # Experimental adaptive states
    #
    # 0 = normal / no strong evidence
    # 1 = elevated degradation
    # 2 = strong degradation
    #
    # These thresholds are experimental and will
    # be validated against failure trajectories.
    # --------------------------------------------------------

    adaptive_state = np.select(

        [
            combined_score >= 3.0,

            combined_score >= 2.0
        ],

        [
            2,

            1
        ],

        default=0
    )


    # --------------------------------------------------------
    # Degradation-change flag
    # --------------------------------------------------------

    degradation_change_detected = (
        combined_score >= 2.0
    ).astype(int)


    # --------------------------------------------------------
    # Build output dataframe
    # --------------------------------------------------------

    output = pd.DataFrame({

        "date":
            chunk["date"],

        "serial_number":
            chunk["serial_number"],

        "failure":
            chunk["failure"],


        "smart_1_normalized_change_7d":
            chunk[
                "smart_1_normalized_change_7d"
            ],

        "smart_1_normalized_change_14d":
            chunk[
                "smart_1_normalized_change_14d"
            ],


        "smart_187_normalized_change_7d":
            chunk[
                "smart_187_normalized_change_7d"
            ],

        "smart_187_normalized_change_14d":
            chunk[
                "smart_187_normalized_change_14d"
            ],


        "smart_1_available":
            smart1_available,

        "smart_187_available":
            smart187_available,


        "smart_1_degradation_score":
            smart1_degradation,

        "smart_187_degradation_score":
            smart187_degradation,


        "combined_degradation_score":
            combined_score,


        "adaptive_state":
            adaptive_state,


        "degradation_change_detected":
            degradation_change_detected
    })


    # --------------------------------------------------------
    # Save chunk
    # --------------------------------------------------------

    output.to_csv(

        output_file,

        mode=(
            "w"
            if first_write
            else "a"
        ),

        header=first_write,

        index=False
    )


    first_write = False


# ============================================================
# Completion
# ============================================================

print("\n" + "=" * 70)
print("ADAPTIVE DEGRADATION ANALYSIS COMPLETED")
print("=" * 70)

print(
    "Baseline file:",
    baseline_file
)

print(
    "Adaptive state file:",
    output_file
)