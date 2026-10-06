import os
import pandas as pd

INPUT_FILE = r".\data\interim\temporal_features.csv"
OUTPUT_FILE = r".\results\digital_twin_state_v3.csv"

CHUNK_SIZE = 500_000

SMART_IDS = [1, 5, 187, 197, 198]

CURRENT_COLS = [
    f"smart_{i}_normalized"
    for i in SMART_IDS
]

CHANGE_COLS = [
    f"smart_{i}_normalized_change_7d"
    for i in SMART_IDS
]

USE_COLS = (
    ["date", "serial_number", "failure"]
    + CURRENT_COLS
    + CHANGE_COLS
)

OUTPUT_COLS = (
    ["date", "serial_number", "failure"]
    + CURRENT_COLS
    + CHANGE_COLS
    + [f"smart_{i}_available" for i in SMART_IDS]
    + ["health_features_available"]
)


print("=" * 70)
print("DIGITAL TWIN STATE V3 CONSTRUCTION")
print("=" * 70)

print("\nInput:")
print(INPUT_FILE)

print("\nOutput:")
print(OUTPUT_FILE)

print("\nState design:")
print("5 current SMART states")
print("5 temporal degradation states")
print("5 observation availability states")
print("1 total availability count")
print("Total = 16 state-related values + identifiers/target")

# Remove previous output if it exists
if os.path.exists(OUTPUT_FILE):
    print("\nRemoving previous V3 output...")
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

    # Ensure date is represented consistently
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Create observation availability indicators
    availability_cols = []

    for smart_id in SMART_IDS:

        current_col = f"smart_{smart_id}_normalized"
        availability_col = f"smart_{smart_id}_available"

        df[availability_col] = (
            df[current_col].notna().astype("int8")
        )

        availability_cols.append(availability_col)

    # Number of currently available SMART features
    df["health_features_available"] = (
        df[availability_cols]
        .sum(axis=1)
        .astype("int8")
    )

    # Keep only the Digital Twin state columns
    df = df[OUTPUT_COLS]

    # Append to output
    df.to_csv(
        OUTPUT_FILE,
        mode="a",
        header=(chunk_number == 1),
        index=False
    )

    total_rows += len(df)

print("\n" + "=" * 70)
print("DIGITAL TWIN STATE V3 COMPLETE")
print("=" * 70)

print(f"\nTotal rows processed: {total_rows:,}")

print("\nOutput file:")
print(OUTPUT_FILE)

print("\nDigital Twin state:")
print("Current SMART measurements       : 5")
print("Temporal degradation features   : 5")
print("Observation availability        : 5")
print("Total available-feature count   : 1")

print("\nNo manually weighted health index was created.")
print("SMART values are preserved for downstream temporal AI modelling.")