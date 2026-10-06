import pandas as pd
import numpy as np

INPUT_FILE = r".\data\interim\temporal_features.csv"

FEATURES = [
    "smart_1_normalized",
    "smart_5_normalized",
    "smart_187_normalized",
    "smart_197_normalized",
    "smart_198_normalized",
]

CHUNK_SIZE = 500_000

# Store values below the healthy baseline of 100
below_100 = {
    feature: []
    for feature in FEATURES
}

for chunk_number, df in enumerate(
    pd.read_csv(
        INPUT_FILE,
        usecols=FEATURES,
        chunksize=CHUNK_SIZE
    ),
    start=1
):

    print(f"Processing chunk {chunk_number}...")

    for feature in FEATURES:

        values = pd.to_numeric(
            df[feature],
            errors="coerce"
        ).dropna()

        # Normalized SMART values below 100
        values = values[values < 100]

        if len(values) == 0:
            continue

        # Convert to NumPy once
        values_array = values.to_numpy()

        # Keep at most 50,000 observations per chunk
        if len(values_array) > 50_000:

            rng = np.random.default_rng(
                seed=100 + chunk_number
            )

            values_array = rng.choice(
                values_array,
                size=50_000,
                replace=False
            )

        below_100[feature].append(
            values_array
        )


print("\n")
print("=" * 70)
print("DIRECTION-AWARE SMART DEGRADATION ANALYSIS")
print("=" * 70)

for feature in FEATURES:

    if not below_100[feature]:

        print(
            f"\n{feature}: no values below 100"
        )

        continue

    values = np.concatenate(
        below_100[feature]
    )

    print(f"\n{feature}")
    print("-" * 50)

    print(
        f"Count below 100 : {len(values):,}"
    )

    print(
        f"Minimum         : {values.min():.2f}"
    )

    print(
        f"P01             : "
        f"{np.percentile(values, 1):.2f}"
    )

    print(
        f"P05             : "
        f"{np.percentile(values, 5):.2f}"
    )

    print(
        f"P10             : "
        f"{np.percentile(values, 10):.2f}"
    )

    print(
        f"P25             : "
        f"{np.percentile(values, 25):.2f}"
    )

    print(
        f"Median          : "
        f"{np.median(values):.2f}"
    )

    print(
        f"P75             : "
        f"{np.percentile(values, 75):.2f}"
    )

    print(
        f"P90             : "
        f"{np.percentile(values, 90):.2f}"
    )

    print(
        f"P95             : "
        f"{np.percentile(values, 95):.2f}"
    )

    print(
        f"P99             : "
        f"{np.percentile(values, 99):.2f}"
    )

    print(
        f"Maximum         : "
        f"{values.max():.2f}"
    )


print("\n")
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)