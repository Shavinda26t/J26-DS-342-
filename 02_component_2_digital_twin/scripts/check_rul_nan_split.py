import numpy as np
import pandas as pd

# ============================================================
# Configuration
# ============================================================

INPUT_DIR = r".\results\rul_sequences"

X_FILE = INPUT_DIR + r"\X_rul_sequences.npy"
META_FILE = INPUT_DIR + r"\rul_sequence_metadata_split.csv"
FEATURE_FILE = INPUT_DIR + r"\rul_sequence_features.txt"


# ============================================================
# Load data
# ============================================================

X = np.load(
    X_FILE,
    mmap_mode="r"
)

metadata = pd.read_csv(
    META_FILE
)

with open(FEATURE_FILE, "r") as f:
    features = [
        line.strip()
        for line in f
        if line.strip()
    ]


print("=" * 90)
print("RUL SEQUENCE MISSING-VALUE CHECK BY SPLIT")
print("=" * 90)

print(f"\nX shape: {X.shape}")
print(f"Features: {len(features)}")


# ============================================================
# Split masks
# ============================================================

split_values = metadata["split"].to_numpy()

train_mask = split_values == "train"
val_mask = split_values == "validation"
test_mask = split_values == "test"


print("\nSequence counts:")
print(f"Train:       {train_mask.sum():,}")
print(f"Validation:  {val_mask.sum():,}")
print(f"Test:        {test_mask.sum():,}")


# ============================================================
# Header
# ============================================================

print("\n" + "-" * 90)

print(
    f"{'Feature':<45}"
    f"{'Train %':>12}"
    f"{'Val %':>12}"
    f"{'Test %':>12}"
)

print("-" * 90)


# ============================================================
# Missing percentage
# ============================================================

for i, feature in enumerate(features):

    train_missing = np.isnan(
        X[train_mask, :, i]
    ).sum()

    val_missing = np.isnan(
        X[val_mask, :, i]
    ).sum()

    test_missing = np.isnan(
        X[test_mask, :, i]
    ).sum()

    train_total = train_mask.sum() * X.shape[1]
    val_total = val_mask.sum() * X.shape[1]
    test_total = test_mask.sum() * X.shape[1]

    train_pct = (
        train_missing / train_total * 100
    )

    val_pct = (
        val_missing / val_total * 100
    )

    test_pct = (
        test_missing / test_total * 100
    )

    print(
        f"{feature:<45}"
        f"{train_pct:>12.3f}"
        f"{val_pct:>12.3f}"
        f"{test_pct:>12.3f}"
    )


print("-" * 90)

print("\nMissing-value check completed.")