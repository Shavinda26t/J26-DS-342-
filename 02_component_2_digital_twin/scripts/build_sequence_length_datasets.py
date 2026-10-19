import os
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

BASE = r".\results\rul_sequences"

STATE_FILE = r".\results\digital_twin_state_v5.csv"

OUTPUT_DIR = os.path.join(
    BASE,
    "sequence_length_experiments"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

SEQUENCE_LENGTHS = [7, 14, 21, 30]

SEED = 42

np.random.seed(SEED)


# ============================================================
# Load HDD-level split
# ============================================================

train_hdds = set(
    pd.read_csv(
        os.path.join(BASE, "train_hdds.csv")
    )["serial_number"]
)

validation_hdds = set(
    pd.read_csv(
        os.path.join(BASE, "validation_hdds.csv")
    )["serial_number"]
)

test_hdds = set(
    pd.read_csv(
        os.path.join(BASE, "test_hdds.csv")
    )["serial_number"]
)

print("Train HDDs:", len(train_hdds))
print("Validation HDDs:", len(validation_hdds))
print("Test HDDs:", len(test_hdds))


# ============================================================
# Actual V5 Digital Twin features
# 16 dynamic state features
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
    "smart_198_available"
]


# ============================================================
# Read Digital Twin state
# ============================================================

columns = [
    "date",
    "serial_number",
    "failure"
] + FEATURES

print("\nReading Digital Twin state...")

df = pd.read_csv(
    STATE_FILE,
    usecols=columns
)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    ["serial_number", "date"]
).reset_index(drop=True)

print("Total observations:", len(df))


# ============================================================
# Find failure dates
# ============================================================

failure_dates = (
    df[df["failure"] == 1]
    [["serial_number", "date"]]
    .drop_duplicates()
)

failure_dates = dict(
    zip(
        failure_dates["serial_number"],
        failure_dates["date"]
    )
)

print("Failed HDDs:", len(failure_dates))


# ============================================================
# Process sequence lengths
# ============================================================

for sequence_length in SEQUENCE_LENGTHS:

    print("\n" + "=" * 60)
    print("SEQUENCE LENGTH:", sequence_length)
    print("=" * 60)

    X_train = []
    y_train = []

    X_validation = []
    y_validation = []

    X_test = []
    y_test = []

    metadata = []

    # --------------------------------------------------------
    # Group by HDD
    # --------------------------------------------------------

    grouped = df.groupby(
        "serial_number",
        sort=False
    )

    for serial_number, group in grouped:

        if serial_number not in failure_dates:
            continue

        failure_date = failure_dates[
            serial_number
        ]

        # IMPORTANT:
        # Failure-day observation is excluded.
        history = group[
            group["date"] < failure_date
        ].copy()

        if len(history) < sequence_length:
            continue

        history = history.sort_values("date")


        # ----------------------------------------------------
        # Sliding windows
        # ----------------------------------------------------

        for end_idx in range(
            sequence_length - 1,
            len(history)
        ):

            start_idx = (
                end_idx - sequence_length + 1
            )

            window = history.iloc[
                start_idx:end_idx + 1
            ]

            sequence_end_date = (
                window.iloc[-1]["date"]
            )

            rul = (
                failure_date -
                sequence_end_date
            ).days

            if rul <= 0:
                continue

            X = window[
                FEATURES
            ].values.astype(np.float32)


            # ------------------------------------------------
            # Assign to HDD-level split
            # ------------------------------------------------

            if serial_number in train_hdds:

                X_train.append(X)
                y_train.append(rul)

                split = "train"

            elif serial_number in validation_hdds:

                X_validation.append(X)
                y_validation.append(rul)

                split = "validation"

            elif serial_number in test_hdds:

                X_test.append(X)
                y_test.append(rul)

                split = "test"

            else:
                continue


            metadata.append({
                "serial_number": serial_number,
                "sequence_end": sequence_end_date,
                "failure_date": failure_date,
                "rul": rul,
                "split": split
            })


    # ========================================================
    # Convert to NumPy
    # ========================================================

    X_train = np.asarray(
        X_train,
        dtype=np.float32
    )

    y_train = np.asarray(
        y_train,
        dtype=np.float32
    )

    X_validation = np.asarray(
        X_validation,
        dtype=np.float32
    )

    y_validation = np.asarray(
        y_validation,
        dtype=np.float32
    )

    X_test = np.asarray(
        X_test,
        dtype=np.float32
    )

    y_test = np.asarray(
        y_test,
        dtype=np.float32
    )


    # ========================================================
    # Output directory
    # ========================================================

    out = os.path.join(
        OUTPUT_DIR,
        f"seq_{sequence_length}"
    )

    os.makedirs(
        out,
        exist_ok=True
    )


    # ========================================================
    # Save
    # ========================================================

    np.save(
        os.path.join(out, "X_train.npy"),
        X_train
    )

    np.save(
        os.path.join(out, "y_train.npy"),
        y_train
    )

    np.save(
        os.path.join(out, "X_validation.npy"),
        X_validation
    )

    np.save(
        os.path.join(out, "y_validation.npy"),
        y_validation
    )

    np.save(
        os.path.join(out, "X_test.npy"),
        X_test
    )

    np.save(
        os.path.join(out, "y_test.npy"),
        y_test
    )

    pd.DataFrame(metadata).to_csv(
        os.path.join(
            out,
            "sequence_metadata.csv"
        ),
        index=False
    )


    # ========================================================
    # Summary
    # ========================================================

    print(
        "Train:",
        X_train.shape,
        y_train.shape
    )

    print(
        "Validation:",
        X_validation.shape,
        y_validation.shape
    )

    print(
        "Test:",
        X_test.shape,
        y_test.shape
    )

    print(
        "Total sequences:",
        len(metadata)
    )

    if len(y_train) > 0:

        print(
            "Train RUL:",
            y_train.min(),
            "-",
            y_train.max()
        )

    if len(y_test) > 0:

        print(
            "Test RUL:",
            y_test.min(),
            "-",
            y_test.max()
        )


print("\nAll sequence lengths completed.")