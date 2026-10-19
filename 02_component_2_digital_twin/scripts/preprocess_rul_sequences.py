import os
import json
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

INPUT_DIR = r".\results\rul_sequences"

X_FILE = os.path.join(
    INPUT_DIR,
    "X_rul_sequences.npy"
)

Y_FILE = os.path.join(
    INPUT_DIR,
    "y_rul.npy"
)

META_FILE = os.path.join(
    INPUT_DIR,
    "rul_sequence_metadata_split.csv"
)

OUTPUT_DIR = os.path.join(
    INPUT_DIR,
    "preprocessed"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# Original 16 features
# ============================================================

FEATURES = [
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


# ============================================================
# Feature groups
# ============================================================

CURRENT_FEATURES = [
    "smart_1_normalized",
    "smart_5_normalized",
    "smart_187_normalized",
    "smart_197_normalized",
    "smart_198_normalized",
]

TEMPORAL_FEATURES = [
    "smart_1_normalized_change_7d",
    "smart_1_normalized_change_14d",
    "smart_1_normalized_rolling_std_7",
    "smart_187_normalized_change_7d",
    "smart_187_normalized_change_14d",
    "smart_187_normalized_rolling_std_7",
]

AVAILABILITY_FEATURES = [
    "smart_1_available",
    "smart_5_available",
    "smart_187_available",
    "smart_197_available",
    "smart_198_available",
]


# ============================================================
# Load data
# ============================================================

print("=" * 80)
print("RUL SEQUENCE PREPROCESSING")
print("=" * 80)

X = np.load(
    X_FILE,
    mmap_mode="r"
)

y = np.load(
    Y_FILE,
    mmap_mode="r"
)

metadata = pd.read_csv(
    META_FILE
)

print(f"\nOriginal X shape: {X.shape}")
print(f"Original y shape: {y.shape}")
print(f"Metadata rows:    {len(metadata):,}")


# ============================================================
# Verify alignment
# ============================================================

assert len(X) == len(y)
assert len(X) == len(metadata)

assert list(metadata["split"].unique())

print("\nAlignment check: PASSED")


# ============================================================
# Split masks
# ============================================================

split_values = metadata["split"].to_numpy()

train_mask = split_values == "train"
val_mask = split_values == "validation"
test_mask = split_values == "test"


print("\nSplit sizes:")
print(f"Train:       {train_mask.sum():,}")
print(f"Validation:  {val_mask.sum():,}")
print(f"Test:        {test_mask.sum():,}")


# ============================================================
# Convert split data to writable arrays
# ============================================================

X_train = np.array(
    X[train_mask],
    dtype=np.float32
)

X_val = np.array(
    X[val_mask],
    dtype=np.float32
)

X_test = np.array(
    X[test_mask],
    dtype=np.float32
)

y_train = np.array(
    y[train_mask],
    dtype=np.float32
)

y_val = np.array(
    y[val_mask],
    dtype=np.float32
)

y_test = np.array(
    y[test_mask],
    dtype=np.float32
)


# ============================================================
# Build 6 temporal validity indicators
# ============================================================

print("\nCreating temporal validity indicators...")


def add_temporal_validity(X_array):

    validity = []

    for feature in TEMPORAL_FEATURES:

        index = FEATURES.index(feature)

        valid = (
            ~np.isnan(
                X_array[:, :, index]
            )
        ).astype(np.float32)

        validity.append(valid)

    return np.stack(
        validity,
        axis=2
    )


train_validity = add_temporal_validity(X_train)
val_validity = add_temporal_validity(X_val)
test_validity = add_temporal_validity(X_test)


print(
    "Temporal validity shape:",
    train_validity.shape
)


# ============================================================
# Learn preprocessing parameters from TRAIN ONLY
# ============================================================

print("\nLearning preprocessing parameters from TRAIN only...")


continuous_features = (
    CURRENT_FEATURES +
    TEMPORAL_FEATURES
)

preprocessing = {}


for feature in continuous_features:

    index = FEATURES.index(feature)

    values = X_train[:, :, index]

    valid_values = values[
        ~np.isnan(values)
    ]

    median = float(
        np.median(valid_values)
    )

    mean = float(
        np.mean(valid_values)
    )

    std = float(
        np.std(valid_values)
    )

    # Prevent division by zero
    if std < 1e-8:
        std = 1.0

    preprocessing[feature] = {
        "median": median,
        "mean": mean,
        "std": std
    }

    print(
        f"{feature:<45} "
        f"median={median:10.4f} "
        f"mean={mean:10.4f} "
        f"std={std:10.4f}"
    )


# ============================================================
# Apply imputation + standardization
# ============================================================

print("\nApplying train-fitted preprocessing...")


def preprocess_continuous(
    X_array,
    parameters
):

    output = np.empty_like(
        X_array,
        dtype=np.float32
    )

    # Continuous features
    for feature in continuous_features:

        index = FEATURES.index(feature)

        values = X_array[:, :, index].copy()

        median = parameters[
            feature
        ]["median"]

        mean = parameters[
            feature
        ]["mean"]

        std = parameters[
            feature
        ]["std"]

        # Median imputation
        values[np.isnan(values)] = median

        # Standardization
        values = (
            values - mean
        ) / std

        output[:, :, index] = values.astype(
            np.float32
        )

    # Availability features
    for feature in AVAILABILITY_FEATURES:

        index = FEATURES.index(feature)

        values = X_array[:, :, index].copy()

        # Availability should never be NaN.
        # Keep as 0/1.
        values[np.isnan(values)] = 0.0

        output[:, :, index] = values.astype(
            np.float32
        )

    return output


X_train_base = preprocess_continuous(
    X_train,
    preprocessing
)

X_val_base = preprocess_continuous(
    X_val,
    preprocessing
)

X_test_base = preprocess_continuous(
    X_test,
    preprocessing
)


# ============================================================
# Append temporal validity features
# ============================================================

X_train_processed = np.concatenate(
    [
        X_train_base,
        train_validity
    ],
    axis=2
)

X_val_processed = np.concatenate(
    [
        X_val_base,
        val_validity
    ],
    axis=2
)

X_test_processed = np.concatenate(
    [
        X_test_base,
        test_validity
    ],
    axis=2
)


# ============================================================
# Final feature names
# ============================================================

PROCESSED_FEATURES = (
    FEATURES +
    [
        feature + "_valid"
        for feature in TEMPORAL_FEATURES
    ]
)


assert len(PROCESSED_FEATURES) == 22


# ============================================================
# Final validation
# ============================================================

print("\n" + "=" * 80)
print("FINAL PREPROCESSING CHECK")
print("=" * 80)

print(
    f"Processed train shape: {X_train_processed.shape}"
)

print(
    f"Processed validation shape: {X_val_processed.shape}"
)

print(
    f"Processed test shape: {X_test_processed.shape}"
)

print(
    f"Processed feature count: {len(PROCESSED_FEATURES)}"
)

print(
    f"Train NaNs: {np.isnan(X_train_processed).sum():,}"
)

print(
    f"Validation NaNs: {np.isnan(X_val_processed).sum():,}"
)

print(
    f"Test NaNs: {np.isnan(X_test_processed).sum():,}"
)


# ============================================================
# Save processed datasets
# ============================================================

np.save(
    os.path.join(
        OUTPUT_DIR,
        "X_train.npy"
    ),
    X_train_processed
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "X_validation.npy"
    ),
    X_val_processed
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "X_test.npy"
    ),
    X_test_processed
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "y_train.npy"
    ),
    y_train
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "y_validation.npy"
    ),
    y_val
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "y_test.npy"
    ),
    y_test
)


# ============================================================
# Save preprocessing parameters
# ============================================================

with open(
    os.path.join(
        OUTPUT_DIR,
        "preprocessing_parameters.json"
    ),
    "w"
) as f:

    json.dump(
        preprocessing,
        f,
        indent=4
    )


# ============================================================
# Save feature list
# ============================================================

with open(
    os.path.join(
        OUTPUT_DIR,
        "processed_features.txt"
    ),
    "w"
) as f:

    for i, feature in enumerate(
        PROCESSED_FEATURES,
        start=1
    ):

        f.write(
            f"{i}. {feature}\n"
        )


# ============================================================
# Save summary
# ============================================================

summary = pd.DataFrame({
    "split": [
        "train",
        "validation",
        "test"
    ],

    "sequences": [
        len(X_train_processed),
        len(X_val_processed),
        len(X_test_processed)
    ],

    "sequence_length": [
        X_train_processed.shape[1],
        X_val_processed.shape[1],
        X_test_processed.shape[1]
    ],

    "features": [
        X_train_processed.shape[2],
        X_val_processed.shape[2],
        X_test_processed.shape[2]
    ],

    "mean_rul": [
        y_train.mean(),
        y_val.mean(),
        y_test.mean()
    ],

    "median_rul": [
        np.median(y_train),
        np.median(y_val),
        np.median(y_test)
    ]
})

summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "preprocessing_summary.csv"
    ),
    index=False
)


print("\nFiles created in:")
print(OUTPUT_DIR)

print("\nPreprocessing completed successfully.")