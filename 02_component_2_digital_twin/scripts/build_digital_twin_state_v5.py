import os
import pandas as pd

INPUT_FILE = r".\data\interim\temporal_features.csv"
OUTPUT_FILE = r".\results\digital_twin_state_v5.csv"

CHUNK_SIZE = 500_000

SMART_IDS = [1, 5, 187, 197, 198]

# ------------------------------------------------------------
# CURRENT SMART STATE
# ------------------------------------------------------------

CURRENT_COLS = [
    "smart_1_normalized",
    "smart_5_normalized",
    "smart_187_normalized",
    "smart_197_normalized",
    "smart_198_normalized"
]

# ------------------------------------------------------------
# SELECTED TEMPORAL DEGRADATION STATE
# ------------------------------------------------------------

TEMPORAL_COLS = [
    "smart_1_normalized_change_7d",
    "smart_1_normalized_change_14d",
    "smart_1_normalized_rolling_std_7",

    "smart_187_normalized_change_7d",
    "smart_187_normalized_change_14d",
    "smart_187_normalized_rolling_std_7"
]

# ------------------------------------------------------------
# SOURCE COLUMNS
# ------------------------------------------------------------

USE_COLS = (
    ["date", "serial_number", "failure"]
    + CURRENT_COLS
    + TEMPORAL_COLS
)

# ------------------------------------------------------------
# OUTPUT COLUMNS
# ------------------------------------------------------------

AVAILABILITY_COLS = [
    "smart_1_available",
    "smart_5_available",
    "smart_187_available",
    "smart_197_available",
    "smart_198_available"
]

OUTPUT_COLS = (
    ["date", "serial_number", "failure"]
    + CURRENT_COLS
    + TEMPORAL_COLS
    + AVAILABILITY_COLS
    + ["health_features_available"]
)

print("=" * 70)
print("DIGITAL TWIN STATE V5 CONSTRUCTION")
print("=" * 70)

print("\nFinal compact Digital Twin state")

print("\nCurrent SMART features: 5")
for col in CURRENT_COLS:
    print("  -", col)

print("\nTemporal degradation features: 6")
for col in TEMPORAL_COLS:
    print("  -", col)

print("\nObservation availability features: 5")
for col in AVAILABILITY_COLS:
    print("  -", col)

print("\nTotal dynamic state features: 16")

# ------------------------------------------------------------
# REMOVE OLD OUTPUT
# ------------------------------------------------------------

if os.path.exists(OUTPUT_FILE):
    print("\nRemoving existing V5 output...")
    os.remove(OUTPUT_FILE)

total_rows = 0
chunk_number = 0

# ------------------------------------------------------------
# PROCESS DATA
# ------------------------------------------------------------

for df in pd.read_csv(
    INPUT_FILE,
    usecols=USE_COLS,
    chunksize=CHUNK_SIZE
):

    chunk_number += 1

    print(f"Processing chunk {chunk_number}...")

    # --------------------------------------------------------
    # CREATE AVAILABILITY INDICATORS
    # --------------------------------------------------------

    for smart_id in SMART_IDS:

        current_col = f"smart_{smart_id}_normalized"
        availability_col = f"smart_{smart_id}_available"

        df[availability_col] = (
            df[current_col].notna()
            .astype("int8")
        )

    # --------------------------------------------------------
    # TOTAL AVAILABLE SMART FEATURES
    # --------------------------------------------------------

    df["health_features_available"] = (
        df[AVAILABILITY_COLS]
        .sum(axis=1)
        .astype("int8")
    )

    # --------------------------------------------------------
    # FINAL STATE
    # --------------------------------------------------------

    df = df[OUTPUT_COLS]

    # --------------------------------------------------------
    # WRITE OUTPUT
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        mode="a",
        header=(chunk_number == 1),
        index=False
    )

    total_rows += len(df)

# ------------------------------------------------------------
# COMPLETE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DIGITAL TWIN STATE V5 COMPLETE")
print("=" * 70)

print(f"\nTotal rows processed: {total_rows:,}")

print("\nOutput:")
print(OUTPUT_FILE)

print("\nDynamic state dimensions: 16")

print("\n5 current SMART states")
print("6 temporal degradation states")
print("5 observation availability states")

print("\nNo manually weighted health index.")
print("No artificial SMART-to-health conversion.")
print("V5 is ready for temporal-state validation.")