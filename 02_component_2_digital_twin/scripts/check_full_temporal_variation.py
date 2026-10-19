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

values = {
    feature: []
    for feature in FEATURES
}

total_rows = 0

for chunk_number, df in enumerate(
    pd.read_csv(
        INPUT_FILE,
        usecols=FEATURES,
        chunksize=500_000
    ),
    start=1
):

    print(f"Processing chunk {chunk_number}...")

    total_rows += len(df)

    for feature in FEATURES:

        v = pd.to_numeric(
            df[feature],
            errors="coerce"
        ).dropna()

        if len(v) > 0:
            values[feature].append(
                v.to_numpy()
            )

print("\n" + "=" * 70)
print("FULL TEMPORAL FEATURE DISTRIBUTIONS")
print("=" * 70)

print(f"\nTotal rows: {total_rows:,}\n")

for feature in FEATURES:

    if not values[feature]:
        print(f"{feature}: NO DATA")
        continue

    v = np.concatenate(values[feature])

    print(
        f"{feature}\n"
        f"  Count : {len(v):,}\n"
        f"  Min   : {v.min():.6f}\n"
        f"  P01   : {np.percentile(v,1):.6f}\n"
        f"  Median: {np.median(v):.6f}\n"
        f"  Mean  : {v.mean():.6f}\n"
        f"  P99   : {np.percentile(v,99):.6f}\n"
        f"  Max   : {v.max():.6f}\n"
        f"  Std   : {v.std():.6f}\n"
    )