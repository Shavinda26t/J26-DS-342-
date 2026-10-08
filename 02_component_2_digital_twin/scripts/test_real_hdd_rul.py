import os
import glob
import pandas as pd

from digital_twin_rul_predictor import DigitalTwinRULPredictor


# ============================================================
# Configuration
# ============================================================

DATA_DIR = r"D:\db module 4th yera 1sem\data_Q1_2023\data_Q1_2023"

# A failed HDD known to have enough pre-failure observations.
SERIAL_NUMBER = "1050A098F97G"


# ============================================================
# Load all Q1 observations for the selected HDD
# ============================================================

print("=" * 70)
print("REAL HDD DIGITAL TWIN RUL TEST")
print("=" * 70)

print("\nSearching for HDD:", SERIAL_NUMBER)

files = sorted(
    glob.glob(
        os.path.join(
            DATA_DIR,
            "*.csv"
        )
    )
)

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


for file in files:

    chunk = pd.read_csv(
        file,
        usecols=required_columns
    )

    match = chunk[
        chunk["serial_number"]
        == SERIAL_NUMBER
    ]

    if not match.empty:

        hdd_records.append(
            match
        )


if not hdd_records:

    raise RuntimeError(
        "Selected HDD was not found."
    )


hdd = pd.concat(
    hdd_records,
    ignore_index=True
)


# ============================================================
# Sort chronologically
# ============================================================

hdd["date"] = pd.to_datetime(
    hdd["date"]
)

hdd = hdd.sort_values(
    "date"
).reset_index(
    drop=True
)


# ============================================================
# Find failure observation
# ============================================================

failure_rows = hdd[
    hdd["failure"] == 1
]


if failure_rows.empty:

    raise RuntimeError(
        "Selected HDD does not contain "
        "a recorded failure."
    )


failure_date = failure_rows[
    "date"
].iloc[0]


failure_index = hdd.index[
    hdd["date"] == failure_date
][0]


# Only observations BEFORE failure.
pre_failure = hdd[
    hdd["date"] < failure_date
].copy()


print(
    "\nFailure date:",
    failure_date.date()
)

print(
    "Pre-failure observations:",
    len(pre_failure)
)


if len(pre_failure) < 30:

    raise RuntimeError(
        "HDD does not have enough "
        "pre-failure observations."
    )


# ============================================================
# Calculate actual RUL
# ============================================================

latest_date = pre_failure[
    "date"
].iloc[-1]

actual_rul = (
    failure_date - latest_date
).days


print(
    "Latest observation:",
    latest_date.date()
)

print(
    "Actual RUL:",
    actual_rul,
    "days"
)


# ============================================================
# Build temporal features
# ============================================================

pre_failure = pre_failure.copy()

pre_failure = pre_failure.sort_values(
    "date"
)


# SMART 1 temporal features

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


# SMART 187 temporal features

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

    availability_name = (
        feature.replace(
            "_normalized",
            "_available"
        )
    )

    pre_failure[
        availability_name
    ] = (
        pre_failure[feature]
        .notna()
        .astype(int)
    )


# ============================================================
# Keep the latest 30 observations
# ============================================================

latest_30 = pre_failure.tail(
    30
).copy()


print(
    "\n30-observation sequence:",
    latest_30["date"].iloc[0].date(),
    "to",
    latest_30["date"].iloc[-1].date()
)


# ============================================================
# RUL prediction
# ============================================================

predictor = DigitalTwinRULPredictor()

predicted_rul = predictor.predict(
    latest_30
)


# ============================================================
# Results
# ============================================================

error = (
    predicted_rul
    - actual_rul
)

absolute_error = abs(
    error
)


print("\n" + "=" * 70)
print("REAL HDD RUL RESULT")
print("=" * 70)

print(
    "HDD:",
    SERIAL_NUMBER
)

print(
    "Failure date:",
    failure_date.date()
)

print(
    "Latest observation:",
    latest_date.date()
)

print(
    "Actual RUL:",
    actual_rul,
    "days"
)

print(
    "Predicted RUL:",
    round(
        predicted_rul,
        2
    ),
    "days"
)

print(
    "Error:",
    round(
        error,
        2
    ),
    "days"
)

print(
    "Absolute error:",
    round(
        absolute_error,
        2
    ),
    "days"
)

print("=" * 70)