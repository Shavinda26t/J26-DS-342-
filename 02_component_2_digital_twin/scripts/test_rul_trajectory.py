import os
import glob
import pandas as pd

from digital_twin_rul_predictor import DigitalTwinRULPredictor


# ============================================================
# Configuration
# ============================================================

DATA_DIR = r"D:\db module 4th yera 1sem\data_Q1_2023\data_Q1_2023"

SERIAL_NUMBER = "1050A098F97G"

# Number of observations before the final pre-failure point.
# These are observation-based cut-offs.
CUTOFFS = [60, 45, 30, 21, 14, 7, 3, 1]


# ============================================================
# Load HDD
# ============================================================

print("=" * 70)
print("DIGITAL TWIN RUL TRAJECTORY TEST")
print("=" * 70)

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

hdd_records = []

files = sorted(
    glob.glob(
        os.path.join(
            DATA_DIR,
            "*.csv"
        )
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
        hdd_records.append(match)


if not hdd_records:
    raise RuntimeError(
        "HDD was not found."
    )


hdd = pd.concat(
    hdd_records,
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
# Build temporal features
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
# Load Digital Twin RUL predictor
# ============================================================

predictor = DigitalTwinRULPredictor()


# ============================================================
# Test multiple cut-off points
# ============================================================

results = []

total_observations = len(pre_failure)


for cutoff in CUTOFFS:

    # Number of observations that should remain
    # after removing the final cutoff observations.
    cutoff_index = total_observations - cutoff

    if cutoff_index < 30:
        print(
            f"\nSkipping cutoff {cutoff}: "
            "not enough observations for 30-step input."
        )
        continue


    available_history = pre_failure.iloc[
        :cutoff_index
    ].copy()


    sequence = available_history.tail(
        30
    ).copy()


    sequence_end_date = sequence[
        "date"
    ].iloc[-1]


    actual_rul = (
        failure_date -
        sequence_end_date
    ).days


    predicted_rul = predictor.predict(
        sequence
    )


    error = (
        predicted_rul -
        actual_rul
    )


    absolute_error = abs(
        error
    )


    results.append(
        {
            "cutoff_observations_before_failure": cutoff,
            "sequence_start_date": sequence[
                "date"
            ].iloc[0],
            "sequence_end_date": sequence_end_date,
            "failure_date": failure_date,
            "actual_rul_days": actual_rul,
            "predicted_rul_days": predicted_rul,
            "error_days": error,
            "absolute_error_days": absolute_error,
        }
    )


# ============================================================
# Results table
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    "actual_rul_days",
    ascending=False
).reset_index(
    drop=True
)


print("\n" + "=" * 70)
print("RUL TRAJECTORY RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        formatters={
            "predicted_rul_days":
                "{:.2f}".format,
            "error_days":
                "{:.2f}".format,
            "absolute_error_days":
                "{:.2f}".format,
        }
    )
)


# ============================================================
# Save results
# ============================================================

output_dir = r".\results\digital_twin_rul"

os.makedirs(
    output_dir,
    exist_ok=True
)

output_file = os.path.join(
    output_dir,
    f"{SERIAL_NUMBER}_rul_trajectory.csv"
)

results_df.to_csv(
    output_file,
    index=False
)


print(
    "\nResults saved to:"
)

print(
    output_file
)

print("=" * 70)