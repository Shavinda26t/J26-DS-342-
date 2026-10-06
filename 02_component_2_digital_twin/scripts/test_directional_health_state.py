import pandas as pd
import numpy as np


INPUT_FILE = r".\data\interim\temporal_features.csv"


FEATURES = {
    "smart_1_normalized": 72.0,
    "smart_5_normalized": 60.0,
    "smart_187_normalized": 34.0,
    "smart_197_normalized": 83.0,
    "smart_198_normalized": 1.0,
}


def calculate_health(value, lower_reference):

    if pd.isna(value):
        return np.nan

    # Healthy baseline
    healthy_reference = 100.0

    # Convert SMART value to health state.
    #
    # 100 -> 1
    # lower_reference -> 0
    #
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


# ---------------------------------------------------------------
# Read a representative sample from the full dataset.
# ---------------------------------------------------------------

df = pd.read_csv(
    INPUT_FILE,
    usecols=list(FEATURES.keys()),
    nrows=500_000
)


print("=" * 70)
print("DIRECTIONAL HEALTH-STATE TEST")
print("=" * 70)


for feature, lower_reference in FEATURES.items():

    health = df[feature].apply(
        lambda x: calculate_health(
            x,
            lower_reference
        )
    )

    print(f"\n{feature}")

    print(
        f"Reference lower value: {lower_reference}"
    )

    print(
        f"Valid observations: "
        f"{health.notna().sum():,}"
    )

    if health.notna().sum() > 0:

        print(
            f"Health mean: "
            f"{health.mean():.4f}"
        )

        print(
            f"Health median: "
            f"{health.median():.4f}"
        )

        print(
            f"Health min: "
            f"{health.min():.4f}"
        )

        print(
            f"Health max: "
            f"{health.max():.4f}"
        )


# ---------------------------------------------------------------
# Show example conversions.
# ---------------------------------------------------------------

print("\n")
print("=" * 70)
print("EXAMPLE SMART VALUE -> HEALTH")
print("=" * 70)

test_values = [
    100,
    99,
    95,
    90,
    85,
    80,
    70,
    60,
    50,
    25,
    1,
]

for feature, lower_reference in FEATURES.items():

    print(f"\n{feature}")

    for value in test_values:

        health = calculate_health(
            value,
            lower_reference
        )

        print(
            f"SMART {value:>3} "
            f"-> Health {health:.3f}"
        )