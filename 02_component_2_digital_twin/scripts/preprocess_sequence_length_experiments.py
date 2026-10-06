import os
import json
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

BASE = r".\results\rul_sequences\sequence_length_experiments"

SEQUENCE_LENGTHS = [7, 14, 21, 30]


# ============================================================
# Feature definitions
# ============================================================

BASE_FEATURES = [
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
    "smart_198_available"
]


# Six temporal validity indicators.
# These are created from the six temporal features.

TEMPORAL_FEATURES = [
    "smart_1_normalized_change_7d",
    "smart_1_normalized_change_14d",
    "smart_1_normalized_rolling_std_7",

    "smart_187_normalized_change_7d",
    "smart_187_normalized_change_14d",
    "smart_187_normalized_rolling_std_7"
]


FINAL_FEATURES = BASE_FEATURES + [
    f"{feature}_valid"
    for feature in TEMPORAL_FEATURES
]


print("Base features:", len(BASE_FEATURES))
print("Validity features:", len(TEMPORAL_FEATURES))
print("Final features:", len(FINAL_FEATURES))


# ============================================================
# Preprocessing function
# ============================================================

def preprocess_split(
    X,
    medians,
    means,
    stds
):

    X = X.astype(np.float32, copy=True)

    n_samples, sequence_length, n_features = X.shape

    # --------------------------------------------------------
    # Create validity flags BEFORE imputation
    # --------------------------------------------------------

    temporal_indices = [
        BASE_FEATURES.index(feature)
        for feature in TEMPORAL_FEATURES
    ]

    validity_flags = np.zeros(
        (
            n_samples,
            sequence_length,
            len(TEMPORAL_FEATURES)
        ),
        dtype=np.float32
    )

    for i, feature_index in enumerate(
        temporal_indices
    ):

        validity_flags[:, :, i] = (
            ~np.isnan(
                X[:, :, feature_index]
            )
        ).astype(np.float32)


    # --------------------------------------------------------
    # Median imputation
    # --------------------------------------------------------

    for j in range(n_features):

        missing = np.isnan(X[:, :, j])

        if np.any(missing):

            X[:, :, j][missing] = medians[j]


    # --------------------------------------------------------
    # Train-only standardisation
    # --------------------------------------------------------

    for j in range(n_features):

        X[:, :, j] = (
            X[:, :, j] - means[j]
        ) / stds[j]


    # --------------------------------------------------------
    # Combine original 16 features + 6 validity flags
    # --------------------------------------------------------

    X_final = np.concatenate(
        [
            X,
            validity_flags
        ],
        axis=2
    )


    return X_final


# ============================================================
# Process each sequence length
# ============================================================

summary = []


for sequence_length in SEQUENCE_LENGTHS:

    print("\n" + "=" * 70)
    print(
        f"PROCESSING SEQUENCE LENGTH: {sequence_length}"
    )
    print("=" * 70)


    input_dir = os.path.join(
        BASE,
        f"seq_{sequence_length}"
    )

    output_dir = os.path.join(
        input_dir,
        "preprocessed"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Load raw sequences
    # --------------------------------------------------------

    X_train = np.load(
        os.path.join(
            input_dir,
            "X_train.npy"
        )
    )

    y_train = np.load(
        os.path.join(
            input_dir,
            "y_train.npy"
        )
    )

    X_validation = np.load(
        os.path.join(
            input_dir,
            "X_validation.npy"
        )
    )

    y_validation = np.load(
        os.path.join(
            input_dir,
            "y_validation.npy"
        )
    )

    X_test = np.load(
        os.path.join(
            input_dir,
            "X_test.npy"
        )
    )

    y_test = np.load(
        os.path.join(
            input_dir,
            "y_test.npy"
        )
    )


    print(
        "Raw train:",
        X_train.shape
    )

    print(
        "Raw validation:",
        X_validation.shape
    )

    print(
        "Raw test:",
        X_test.shape
    )


    # ========================================================
    # Calculate preprocessing parameters
    # USING TRAINING DATA ONLY
    # ========================================================

    n_features = X_train.shape[2]

    medians = np.zeros(
        n_features,
        dtype=np.float32
    )

    means = np.zeros(
        n_features,
        dtype=np.float32
    )

    stds = np.ones(
        n_features,
        dtype=np.float32
    )


    print(
        "\nCalculating train-only preprocessing parameters..."
    )


    for j in range(n_features):

        values = X_train[:, :, j].reshape(-1)

        valid_values = values[
            ~np.isnan(values)
        ]

        if len(valid_values) == 0:

            medians[j] = 0.0
            means[j] = 0.0
            stds[j] = 1.0

        else:

            medians[j] = np.median(
                valid_values
            )

            means[j] = np.mean(
                valid_values
            )

            stds[j] = np.std(
                valid_values
            )

            # Avoid division by zero
            if stds[j] == 0:

                stds[j] = 1.0


    # ========================================================
    # Print train-only parameters
    # ========================================================

    print("\nTrain-only parameters:")

    for j, feature in enumerate(BASE_FEATURES):

        print(
            f"{feature}: "
            f"median={medians[j]:.4f}, "
            f"mean={means[j]:.4f}, "
            f"std={stds[j]:.4f}"
        )


    # ========================================================
    # Preprocess all splits
    # ========================================================

    print("\nPreprocessing train...")

    X_train_processed = preprocess_split(
        X_train,
        medians,
        means,
        stds
    )


    print("Preprocessing validation...")

    X_validation_processed = preprocess_split(
        X_validation,
        medians,
        means,
        stds
    )


    print("Preprocessing test...")

    X_test_processed = preprocess_split(
        X_test,
        medians,
        means,
        stds
    )


    # ========================================================
    # Check NaNs
    # ========================================================

    train_nan = np.isnan(
        X_train_processed
    ).sum()

    validation_nan = np.isnan(
        X_validation_processed
    ).sum()

    test_nan = np.isnan(
        X_test_processed
    ).sum()


    print("\nNaN check:")

    print(
        "Train:",
        train_nan
    )

    print(
        "Validation:",
        validation_nan
    )

    print(
        "Test:",
        test_nan
    )


    # ========================================================
    # Save processed arrays
    # ========================================================

    np.save(
        os.path.join(
            output_dir,
            "X_train.npy"
        ),
        X_train_processed
    )

    np.save(
        os.path.join(
            output_dir,
            "y_train.npy"
        ),
        y_train
    )

    np.save(
        os.path.join(
            output_dir,
            "X_validation.npy"
        ),
        X_validation_processed
    )

    np.save(
        os.path.join(
            output_dir,
            "y_validation.npy"
        ),
        y_validation
    )

    np.save(
        os.path.join(
            output_dir,
            "X_test.npy"
        ),
        X_test_processed
    )

    np.save(
        os.path.join(
            output_dir,
            "y_test.npy"
        ),
        y_test
    )


    # ========================================================
    # Save preprocessing parameters
    # ========================================================

    parameters = {
        "sequence_length": sequence_length,
        "base_features": BASE_FEATURES,
        "temporal_features": TEMPORAL_FEATURES,
        "final_feature_count": len(FINAL_FEATURES),
        "medians": medians.tolist(),
        "means": means.tolist(),
        "stds": stds.tolist()
    }


    with open(
        os.path.join(
            output_dir,
            "preprocessing_parameters.json"
        ),
        "w"
    ) as f:

        json.dump(
            parameters,
            f,
            indent=2
        )


    # ========================================================
    # Save feature list
    # ========================================================

    with open(
        os.path.join(
            output_dir,
            "processed_features.txt"
        ),
        "w"
    ) as f:

        for i, feature in enumerate(
            FINAL_FEATURES,
            start=1
        ):

            f.write(
                f"{i}. {feature}\n"
            )


    # ========================================================
    # Summary
    # ========================================================

    summary.append({
        "sequence_length": sequence_length,

        "train_sequences": len(X_train_processed),
        "validation_sequences": len(X_validation_processed),
        "test_sequences": len(X_test_processed),

        "train_shape": str(
            X_train_processed.shape
        ),

        "validation_shape": str(
            X_validation_processed.shape
        ),

        "test_shape": str(
            X_test_processed.shape
        ),

        "final_features": X_train_processed.shape[2],

        "train_nan": train_nan,
        "validation_nan": validation_nan,
        "test_nan": test_nan
    })


    print("\nProcessed shapes:")

    print(
        "Train:",
        X_train_processed.shape
    )

    print(
        "Validation:",
        X_validation_processed.shape
    )

    print(
        "Test:",
        X_test_processed.shape
    )


# ============================================================
# Save overall summary
# ============================================================

summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    os.path.join(
        BASE,
        "sequence_length_preprocessing_summary.csv"
    ),
    index=False
)


print("\n" + "=" * 70)
print("ALL PREPROCESSING COMPLETED")
print("=" * 70)

print(summary_df.to_string(index=False))