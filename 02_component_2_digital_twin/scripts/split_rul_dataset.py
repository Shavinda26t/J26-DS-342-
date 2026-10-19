import os
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

INPUT_DIR = r".\results\rul_sequences"

METADATA_FILE = os.path.join(
    INPUT_DIR,
    "rul_sequence_metadata.csv"
)

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# Load metadata
# ============================================================

print("=" * 70)
print("RUL HDD-LEVEL DATASET SPLIT")
print("=" * 70)

metadata = pd.read_csv(METADATA_FILE)

print(f"\nTotal sequences: {len(metadata):,}")
print(
    f"Unique HDDs: {metadata['serial_number'].nunique():,}"
)


# ============================================================
# Get unique HDDs
# ============================================================

hdds = metadata["serial_number"].unique()

rng = np.random.default_rng(RANDOM_SEED)

hdds = rng.permutation(hdds)

n_hdds = len(hdds)

n_train = int(n_hdds * TRAIN_RATIO)
n_val = int(n_hdds * VAL_RATIO)

train_hdds = hdds[:n_train]

val_hdds = hdds[
    n_train:n_train + n_val
]

test_hdds = hdds[
    n_train + n_val:
]


# ============================================================
# Assign split
# ============================================================

metadata["split"] = "test"

metadata.loc[
    metadata["serial_number"].isin(train_hdds),
    "split"
] = "train"

metadata.loc[
    metadata["serial_number"].isin(val_hdds),
    "split"
] = "validation"


# ============================================================
# Safety checks
# ============================================================

train_set = set(train_hdds)
val_set = set(val_hdds)
test_set = set(test_hdds)

assert len(train_set & val_set) == 0
assert len(train_set & test_set) == 0
assert len(val_set & test_set) == 0

assert len(train_set) + len(val_set) + len(test_set) == n_hdds


# ============================================================
# Summary
# ============================================================

print("\nHDD split:")
print(f"Train HDDs:       {len(train_set):,}")
print(f"Validation HDDs:  {len(val_set):,}")
print(f"Test HDDs:        {len(test_set):,}")


print("\nSequence split:")

for split in ["train", "validation", "test"]:

    subset = metadata[
        metadata["split"] == split
    ]

    print(
        f"{split:<12} "
        f"{len(subset):,} sequences | "
        f"{subset['serial_number'].nunique():,} HDDs"
    )


# ============================================================
# RUL distribution
# ============================================================

print("\nRUL distribution by split:")

for split in ["train", "validation", "test"]:

    subset = metadata[
        metadata["split"] == split
    ]

    print(f"\n{split.upper()}")

    print(
        subset["rul_days"]
        .describe()
        .to_string()
    )


# ============================================================
# RUL ranges
# ============================================================

print("\nRUL ranges by split:")

for split in ["train", "validation", "test"]:

    subset = metadata[
        metadata["split"] == split
    ]

    print(f"\n{split.upper()}")

    print(
        "1-7 days:",
        ((subset["rul_days"] >= 1) &
         (subset["rul_days"] <= 7)).sum()
    )

    print(
        "8-14 days:",
        ((subset["rul_days"] >= 8) &
         (subset["rul_days"] <= 14)).sum()
    )

    print(
        "15-30 days:",
        ((subset["rul_days"] >= 15) &
         (subset["rul_days"] <= 30)).sum()
    )

    print(
        "31-60 days:",
        ((subset["rul_days"] >= 31) &
         (subset["rul_days"] <= 60)).sum()
    )

    print(
        ">60 days:",
        (subset["rul_days"] > 60).sum()
    )


# ============================================================
# Save split metadata
# ============================================================

output_file = os.path.join(
    INPUT_DIR,
    "rul_sequence_metadata_split.csv"
)

metadata.to_csv(
    output_file,
    index=False
)


# ============================================================
# Save HDD lists
# ============================================================

pd.Series(
    train_hdds,
    name="serial_number"
).to_csv(
    os.path.join(INPUT_DIR, "train_hdds.csv"),
    index=False
)

pd.Series(
    val_hdds,
    name="serial_number"
).to_csv(
    os.path.join(INPUT_DIR, "validation_hdds.csv"),
    index=False
)

pd.Series(
    test_hdds,
    name="serial_number"
).to_csv(
    os.path.join(INPUT_DIR, "test_hdds.csv"),
    index=False
)


print("\nFiles created:")
print(output_file)
print(os.path.join(INPUT_DIR, "train_hdds.csv"))
print(os.path.join(INPUT_DIR, "validation_hdds.csv"))
print(os.path.join(INPUT_DIR, "test_hdds.csv"))

print("\nHDD-level split completed.")