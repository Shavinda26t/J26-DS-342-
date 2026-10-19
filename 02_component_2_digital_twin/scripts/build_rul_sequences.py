import os
import numpy as np
import pandas as pd

# ============================================================
# Configuration
# ============================================================

INPUT_FILE = r".\results\digital_twin_state_v5.csv"

OUTPUT_DIR = r".\results\rul_sequences"

SEQUENCE_LENGTH = 14

# 16 Digital Twin state features
STATE_FEATURES = [
    "smart_1_normalized",
    "smart_5_normalized",
    "smart_187_normalized",
    "smart_197_normalized",
    "smart_198_normalized",

    "smart_1_normalized_change_7d",
    "smart_1_normalized_change_14d",
    "smart_1_normalized_rolling_std_7",

    "smart_187_normalized_change_7d",
    "smart_187_normalized_change_14d",
    "smart_187_normalized_rolling_std_7",

    "smart_1_available",
    "smart_5_available",
    "smart_187_available",
    "smart_197_available",
    "smart_198_available",
]

META_COLUMNS = [
    "date",
    "serial_number",
    "failure",
]

ALL_COLUMNS = META_COLUMNS + STATE_FEATURES


# ============================================================
# Prepare output directory
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Load V5 Digital Twin state
# ============================================================

print("=" * 70)
print("BUILDING RUL SEQUENCES")
print("=" * 70)

print("\nLoading Digital Twin V5 state...")

df = pd.read_csv(
    INPUT_FILE,
    usecols=ALL_COLUMNS
)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    ["serial_number", "date"]
).reset_index(drop=True)

print(f"Rows loaded: {len(df):,}")
print(f"Unique HDDs: {df['serial_number'].nunique():,}")


# ============================================================
# Identify failed HDDs
# ============================================================

failed_ids = df.loc[
    df["failure"] == 1,
    "serial_number"
].unique()

print(f"Failed HDDs: {len(failed_ids):,}")


failed_df = df[
    df["serial_number"].isin(failed_ids)
].copy()


# ============================================================
# Determine failure date for each HDD
# ============================================================

failure_dates = (
    failed_df.loc[failed_df["failure"] == 1]
    .groupby("serial_number")["date"]
    .min()
)

failed_df["failure_date"] = failed_df[
    "serial_number"
].map(failure_dates)


# ============================================================
# Keep only observations BEFORE failure
# ============================================================

failed_df = failed_df[
    failed_df["date"] < failed_df["failure_date"]
].copy()

failed_df["rul_days"] = (
    failed_df["failure_date"] - failed_df["date"]
).dt.days


# ============================================================
# Build sequences
# ============================================================

X_sequences = []
y_rul = []

sequence_hdd_ids = []
sequence_end_dates = []
sequence_failure_dates = []

total_hdds = 0
eligible_hdds = 0

print("\nBuilding sequences...")

for serial_number, group in failed_df.groupby(
    "serial_number",
    sort=False
):

    total_hdds += 1

    group = group.sort_values("date").reset_index(drop=True)

    # Need at least 14 observations
    if len(group) < SEQUENCE_LENGTH:
        continue

    eligible_hdds += 1

    values = group[STATE_FEATURES].to_numpy(
        dtype=np.float32
    )

    rul_values = group["rul_days"].to_numpy(
        dtype=np.float32
    )

    dates = group["date"].to_numpy()

    failure_date = group["failure_date"].iloc[0]

    # --------------------------------------------------------
    # Sliding windows
    # --------------------------------------------------------

    for end_idx in range(
        SEQUENCE_LENGTH - 1,
        len(group)
    ):

        start_idx = end_idx - SEQUENCE_LENGTH + 1

        sequence = values[
            start_idx:end_idx + 1
        ]

        rul = rul_values[end_idx]

        # Safety check: only positive RUL
        if rul <= 0:
            continue

        X_sequences.append(sequence)
        y_rul.append(rul)

        sequence_hdd_ids.append(serial_number)
        sequence_end_dates.append(dates[end_idx])
        sequence_failure_dates.append(failure_date)


# ============================================================
# Convert to NumPy arrays
# ============================================================

X = np.stack(X_sequences).astype(np.float32)

y = np.asarray(
    y_rul,
    dtype=np.float32
)

print("\n" + "=" * 70)
print("SEQUENCE DATASET SUMMARY")
print("=" * 70)

print(f"Failed HDDs:             {total_hdds:,}")
print(f"Eligible HDDs:           {eligible_hdds:,}")
print(f"Sequences:               {len(X):,}")
print(f"Sequence shape:          {X.shape}")
print(f"Target shape:            {y.shape}")

print("\nRUL statistics:")
print(f"Minimum RUL:             {y.min():.0f} days")
print(f"Maximum RUL:             {y.max():.0f} days")
print(f"Mean RUL:                {y.mean():.2f} days")
print(f"Median RUL:              {np.median(y):.2f} days")


# ============================================================
# Missing-value analysis
# ============================================================

print("\nMissing-value analysis:")

missing_total = np.isnan(X).sum()

print(
    f"Total NaN values in sequences: {missing_total:,}"
)

for i, feature in enumerate(STATE_FEATURES):

    missing = np.isnan(X[:, :, i]).sum()

    print(
        f"{feature:<45} "
        f"{missing:,}"
    )


# ============================================================
# Save arrays
# ============================================================

np.save(
    os.path.join(
        OUTPUT_DIR,
        "X_rul_sequences.npy"
    ),
    X
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "y_rul.npy"
    ),
    y
)


# ============================================================
# Save sequence metadata
# ============================================================

metadata = pd.DataFrame({
    "serial_number": sequence_hdd_ids,
    "sequence_end_date": pd.to_datetime(
        sequence_end_dates
    ),
    "failure_date": pd.to_datetime(
        sequence_failure_dates
    ),
    "rul_days": y,
})

metadata.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "rul_sequence_metadata.csv"
    ),
    index=False
)


# ============================================================
# Save feature list
# ============================================================

with open(
    os.path.join(
        OUTPUT_DIR,
        "rul_sequence_features.txt"
    ),
    "w"
) as f:

    for feature in STATE_FEATURES:
        f.write(feature + "\n")


print("\nFiles created:")

print(
    os.path.join(
        OUTPUT_DIR,
        "X_rul_sequences.npy"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "y_rul.npy"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "rul_sequence_metadata.csv"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "rul_sequence_features.txt"
    )
)

print("\nSequence construction completed.")