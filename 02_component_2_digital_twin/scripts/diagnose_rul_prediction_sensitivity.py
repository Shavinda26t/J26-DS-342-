import os
import glob
import pandas as pd
import numpy as np

from digital_twin_rul_predictor import DigitalTwinRULPredictor


# ============================================================
# Configuration
# ============================================================

DATA_DIR = r"D:\db module 4th yera 1sem\data_Q1_2023\data_Q1_2023"

SERIAL_NUMBER = "1050A098F97G"

CUTOFFS = [45, 30, 21, 14, 7, 3, 1]

# Features that should change as the HDD degrades
IMPORTANT_FEATURES = [
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
]


# ============================================================
# Load HDD
# ============================================================

print("=" * 75)
print("RUL PREDICTION SENSITIVITY DIAGNOSTIC")
print("=" * 75)

print("\nHDD:", SERIAL_NUMBER)

required_columns = [
    "date",
    "serial_number",
    "failure",
    "smart_1_normalized",
    "smart_5_normalized",
    "smart_187_normalized",
    "smart_197_normalized",
    "smart_198_normalized",
]

records = []

files = sorted(
    glob.glob(
        os.path.join(DATA_DIR, "*.csv")
    )
)

for file in files:

    chunk = pd.read_csv(
        file,
        usecols=required_columns
    )

    match = chunk[
        chunk["serial_number"] == SERIAL_NUMBER
    ]

    if not match.empty:
        records.append(match)


if not records:

    raise RuntimeError(
        "HDD was not found."
    )


hdd = pd.concat(
    records,
    ignore_index=True
)

hdd["date"] = pd.to_datetime(
    hdd["date"]
)

hdd = hdd.sort_values(
    "date"
).reset_index(drop=True)


# ============================================================
# Find failure
# ============================================================

failure_rows = hdd[
    hdd["failure"] == 1
]

if failure_rows.empty:

    raise RuntimeError(
        "No failure record found."
    )


failure_date = failure_rows[
    "date"
].iloc[0]


pre_failure = hdd[
    hdd["date"] < failure_date
].copy()

pre_failure = pre_failure.sort_values(
    "date"
).reset_index(drop=True)


print(
    "\nFailure date:",
    failure_date.date()
)

print(
    "Pre-failure observations:",
    len(pre_failure)
)


# ============================================================
# Build the SAME temporal features used by the model
# ============================================================

pre_failure[
    "smart_1_normalized_change_7d"
] = pre_failure[
    "smart_1_normalized"
].diff(7)

pre_failure[
    "smart_1_normalized_change_14d"
] = pre_failure[
    "smart_1_normalized"
].diff(14)

pre_failure[
    "smart_1_normalized_rolling_std_7"
] = pre_failure[
    "smart_1_normalized"
].rolling(
    window=7,
    min_periods=2
).std()


pre_failure[
    "smart_187_normalized_change_7d"
] = pre_failure[
    "smart_187_normalized"
].diff(7)

pre_failure[
    "smart_187_normalized_change_14d"
] = pre_failure[
    "smart_187_normalized"
].diff(14)

pre_failure[
    "smart_187_normalized_rolling_std_7"
] = pre_failure[
    "smart_187_normalized"
].rolling(
    window=7,
    min_periods=2
).std()


# ============================================================
# Availability features
# ============================================================

availability_features = [
    "smart_1_normalized",
    "smart_5_normalized",
    "smart_187_normalized",
    "smart_197_normalized",
    "smart_198_normalized",
]

for feature in availability_features:

    availability_name = feature.replace(
        "_normalized",
        "_available"
    )

    pre_failure[
        availability_name
    ] = (
        pre_failure[feature]
        .notna()
        .astype(int)
    )


# ============================================================
# Load predictor
# ============================================================

predictor = DigitalTwinRULPredictor()


# ============================================================
# Analyse each cutoff
# ============================================================

results = []

previous_sequence = None


for cutoff in CUTOFFS:

    cutoff_index = len(pre_failure) - cutoff

    if cutoff_index < 30:

        continue


    available_history = pre_failure.iloc[
        :cutoff_index
    ].copy()

    sequence = available_history.tail(
        30
    ).copy()


    sequence_end = sequence[
        "date"
    ].iloc[-1]


    actual_rul = (
        failure_date -
        sequence_end
    ).days


    # --------------------------------------------------------
    # Get the exact processed 22-feature input
    # --------------------------------------------------------

    processed = predictor.preprocess_sequence(
        sequence
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predicted_rul = predictor.predict(
        sequence
    )


    # --------------------------------------------------------
    # Input statistics
    # --------------------------------------------------------

    feature_means = np.nanmean(
        processed,
        axis=0
    )


    feature_std = np.nanstd(
        processed,
        axis=0
    )


    # --------------------------------------------------------
    # Compare input with previous sequence
    # --------------------------------------------------------

    input_change = np.nan

    if previous_sequence is not None:

        input_change = np.mean(
            np.abs(
                processed -
                previous_sequence
            )
        )


    previous_sequence = processed.copy()


    results.append(
        {
            "cutoff": cutoff,
            "sequence_end": sequence_end,
            "actual_rul": actual_rul,
            "predicted_rul": predicted_rul,
            "prediction_change": np.nan,
            "mean_absolute_input_change": input_change,
            "mean_processed_feature_value":
                np.mean(feature_means),
            "mean_processed_feature_std":
                np.mean(feature_std),
        }
    )


# ============================================================
# Calculate prediction changes
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    "actual_rul",
    ascending=False
).reset_index(drop=True)


results_df[
    "prediction_change"
] = results_df[
    "predicted_rul"
].diff().abs()


# ============================================================
# Print results
# ============================================================

print("\n" + "=" * 75)
print("INPUT / PREDICTION SENSITIVITY")
print("=" * 75)

print(
    results_df.to_string(
        index=False,
        formatters={
            "predicted_rul":
                "{:.4f}".format,
            "prediction_change":
                "{:.4f}".format,
            "mean_absolute_input_change":
                "{:.6f}".format,
            "mean_processed_feature_value":
                "{:.4f}".format,
            "mean_processed_feature_std":
                "{:.4f}".format,
        }
    )
)


# ============================================================
# Detailed feature comparison
# ============================================================

print("\n" + "=" * 75)
print("LATEST VALUES OF IMPORTANT FEATURES")
print("=" * 75)

for cutoff in CUTOFFS:

    cutoff_index = len(pre_failure) - cutoff

    if cutoff_index < 30:
        continue

    sequence = pre_failure.iloc[
        :cutoff_index
    ].tail(30)

    latest = sequence.iloc[-1]

    print(
        f"\nActual RUL approximately "
        f"{(failure_date - latest['date']).days} days "
        f"| Date: {latest['date'].date()}"
    )

    for feature in IMPORTANT_FEATURES:

        value = latest[feature]

        print(
            f"  {feature:<45} {value}"
        )


# ============================================================
# Save diagnostic results
# ============================================================

output_dir = (
    r".\results\digital_twin_rul"
)

os.makedirs(
    output_dir,
    exist_ok=True
)

output_file = os.path.join(
    output_dir,
    f"{SERIAL_NUMBER}_rul_sensitivity.csv"
)

results_df.to_csv(
    output_file,
    index=False
)


print("\n" + "=" * 75)

print(
    "Diagnostic results saved to:"
)

print(
    output_file
)

print("=" * 75)