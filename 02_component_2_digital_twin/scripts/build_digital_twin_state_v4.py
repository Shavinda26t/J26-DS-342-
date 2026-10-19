import os
import pandas as pd

INPUT_FILE = r".\data\interim\temporal_features.csv"
OUTPUT_FILE = r".\results\digital_twin_state_v4.csv"

CHUNK_SIZE = 500_000

SMART_IDS = [1, 5, 187, 197, 198]

CURRENT_COLS = [
    f"smart_{i}_normalized"
    for i in SMART_IDS
]

CHANGE_7_COLS = [
    f"smart_{i}_normalized_change_7d"
    for i in SMART_IDS
]

CHANGE_14_COLS = [
    f"smart_{i}_normalized_change_14d"
    for i in SMART_IDS
]

ROLLING_STD_COLS = [
    f"smart_{i}_normalized_rolling_std_7"
    for i in SMART_IDS
]

AVAILABILITY_COLS = [
    f"smart_{i}_available"
    for i in SMART_IDS
]

USE_COLS = (
    ["date", "serial_number", "failure"]
    + CURRENT_COLS
    + CHANGE_7_COLS
    + CHANGE_14_COLS
    + ROLLING_STD_COLS
)

OUTPUT_COLS = (
    ["date", "serial_number", "failure"]
    + CURRENT_COLS
    + CHANGE_7_COLS
    + CHANGE_14_COLS
    + ROLLING_STD_COLS
    + AVAILABILITY_COLS
    + ["health_features_available"]
)

print("=" * 70)
print("DIGITAL TWIN STATE V4 CONSTRUCTION")
print("=" * 70)

print("\nState representation:")
print("5 current SMART states")
print("5 short-term degradation states")
print("5 longer-term degradation states")
print("5 rolling volatility states")
print("5 observation availability states")

print("\nTotal dynamic state features: 25")

if os.path.exists(OUTPUT_FILE):
    print("\nRemoving previous V4 output...")
    os.remove(OUTPUT_FILE)

total_rows = 0
chunk_number = 0

for df in pd.read_csv(
    INPUT_FILE,
    usecols=USE_COLS,
    chunksize=CHUNK_SIZE
):

    chunk_number += 1
    print(f"Processing chunk {chunk_number}...")

    # Create availability from current SMART observations
    for smart_id in SMART_IDS:

        current_col = f"smart_{smart_id}_normalized"
        availability_col = f"smart_{smart_id}_available"

        df[availability_col] = (
            df[current_col].notna().astype("int8")
        )

    # Count available SMART features
    df["health_features_available"] = (
        df[AVAILABILITY_COLS]
        .sum(axis=1)
        .astype("int8")
    )

    # Arrange final state
    df = df[OUTPUT_COLS]

    df.to_csv(
        OUTPUT_FILE,
        mode="a",
        header=(chunk_number == 1),
        index=False
    )

    total_rows += len(df)

print("\n" + "=" * 70)
print("DIGITAL TWIN STATE V4 COMPLETE")
print("=" * 70)

print(f"\nTotal rows processed: {total_rows:,}")

print("\nOutput:")
print(OUTPUT_FILE)

print("\nDynamic state dimensions: 25")

print("\n5  Current SMART measurements")
print("5  Short-term degradation features")
print("5  Long-term degradation features")
print("5  Rolling volatility features")
print("5  Observation availability features")

print("\nNo manually weighted health index was created.")
print("No SMART values were artificially converted into health scores.")