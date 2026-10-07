import os
import numpy as np
import pandas as pd


INPUT_FILE = r".\results\digital_twin_state_v5.csv"

OUTPUT_DIR = r".\results\adaptive_digital_twin"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "degradation_threshold_analysis.csv"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

CHUNK_SIZE = 500_000


FEATURES = [
    "smart_1_normalized_change_7d",
    "smart_1_normalized_change_14d",
    "smart_187_normalized_change_7d",
    "smart_187_normalized_change_14d"
]


# Store only non-zero negative degradation values
negative_values = {
    feature: []
    for feature in FEATURES
}


total_rows = 0


print("=" * 70)
print("DEGRADATION THRESHOLD ANALYSIS")
print("=" * 70)

print("\nReading Digital Twin V5 state...")


for chunk_number, chunk in enumerate(
    pd.read_csv(
        INPUT_FILE,
        usecols=FEATURES,
        chunksize=CHUNK_SIZE
    ),
    start=1
):

    total_rows += len(chunk)

    print(
        f"Processing chunk {chunk_number}..."
    )

    for feature in FEATURES:

        values = pd.to_numeric(
            chunk[feature],
            errors="coerce"
        ).dropna().values

        # Only negative changes represent
        # downward movement for these SMART features.
        degradation = values[
            values < 0
        ]

        if len(degradation) > 0:

            negative_values[feature].extend(
                degradation.tolist()
            )


print("\nTotal rows:", total_rows)


# ============================================================
# Calculate empirical degradation statistics
# ============================================================

results = []


for feature in FEATURES:

    values = np.asarray(
        negative_values[feature],
        dtype=np.float64
    )

    if len(values) == 0:

        print(
            f"\n{feature}: NO NEGATIVE VALUES"
        )

        continue


    percentiles = np.percentile(
        values,
        [
            1,
            5,
            10,
            25,
            50,
            75,
            90,
            95,
            99
        ]
    )


    results.append({

        "feature": feature,

        "negative_count": len(values),

        "minimum":
            np.min(values),

        "P1":
            percentiles[0],

        "P5":
            percentiles[1],

        "P10":
            percentiles[2],

        "P25":
            percentiles[3],

        "P50":
            percentiles[4],

        "P75":
            percentiles[5],

        "P90":
            percentiles[6],

        "P95":
            percentiles[7],

        "P99":
            percentiles[8],

        "mean":
            np.mean(values),

        "std":
            np.std(values)
    })


# ============================================================
# Save results
# ============================================================

results_df = pd.DataFrame(
    results
)


print("\n")
print("=" * 70)
print("NEGATIVE DEGRADATION DISTRIBUTIONS")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n")
print("=" * 70)
print("ANALYSIS COMPLETED")
print("=" * 70)

print(
    "Output:",
    OUTPUT_FILE
)