import os
import pandas as pd

from digital_twin_rul_predictor import DigitalTwinRULPredictor


# ============================================================
# Configuration
# ============================================================

V5_FILE = r".\results\digital_twin_state_v5.csv"

SERIAL_NUMBER = "1050A098F97G"

CUTOFFS = [45, 30, 21, 14, 7, 3, 1]


# ============================================================
# V5 Digital Twin columns
# ============================================================

USECOLS = [
    "date",
    "serial_number",
    "failure",

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

    "health_features_available",
]


# ============================================================
# Load only the selected HDD from V5
# ============================================================

print("=" * 75)
print("REAL DIGITAL TWIN V5 RUL TRAJECTORY TEST")
print("=" * 75)

print("\nV5 state file:")
print(V5_FILE)

print("\nHDD:", SERIAL_NUMBER)

records = []

for chunk in pd.read_csv(
    V5_FILE,
    usecols=USECOLS,
    chunksize=500_000
):

    match = chunk[
        chunk["serial_number"].astype(str)
        == SERIAL_NUMBER
    ]

    if not match.empty:
        records.append(match)


if not records:
    raise RuntimeError(
        "Selected HDD was not found in V5."
    )


hdd = pd.concat(
    records,
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
).reset_index(drop=True)


print(
    "\nTotal V5 observations:",
    len(hdd)
)


# ============================================================
# Find failure
# ============================================================

failure_rows = hdd[
    hdd["failure"] == 1
]

if failure_rows.empty:
    raise RuntimeError(
        "No failure record found for this HDD."
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
    "Failure date:",
    failure_date.date()
)

print(
    "Pre-failure observations:",
    len(pre_failure)
)


# ============================================================
# Load predictor
# ============================================================

predictor = DigitalTwinRULPredictor()


# ============================================================
# Test V5 state at multiple historical cut-offs
# ============================================================

results = []

previous_processed = None


for cutoff in CUTOFFS:

    cutoff_index = (
        len(pre_failure) - cutoff
    )


    if cutoff_index < 30:

        print(
            f"\nSkipping cutoff {cutoff}: "
            "not enough observations."
        )

        continue


    history = pre_failure.iloc[
        :cutoff_index
    ].copy()


    sequence = history.tail(
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
    # Check actual V5 state before preprocessing
    # --------------------------------------------------------

    raw_state_change = None

    if previous_processed is not None:

        current_values = sequence[
            predictor.base_features +
            predictor.temporal_features
        ].copy()

        previous_values = previous_sequence[
            predictor.base_features +
            predictor.temporal_features
        ].copy()

        current_values = (
            current_values
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
        )

        previous_values = (
            previous_values
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
        )

        raw_state_change = (
            current_values
            .to_numpy()
            -
            previous_values
            .to_numpy()
        )

        raw_state_change = abs(
            raw_state_change
        ).mean()


    # --------------------------------------------------------
    # Process through exact predictor preprocessing
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
    # Processed input change
    # --------------------------------------------------------

    processed_change = None

    if previous_processed is not None:

        processed_change = abs(
            processed -
            previous_processed
        ).mean()


    previous_processed = processed.copy()
    previous_sequence = sequence.copy()


    results.append(
        {
            "cutoff_observations_before_failure":
                cutoff,

            "sequence_start_date":
                sequence["date"].iloc[0],

            "sequence_end_date":
                sequence_end,

            "actual_rul_days":
                actual_rul,

            "predicted_rul_days":
                predicted_rul,

            "prediction_error_days":
                predicted_rul - actual_rul,

            "absolute_error_days":
                abs(predicted_rul - actual_rul),

            "raw_v5_state_change":
                raw_state_change,

            "processed_input_change":
                processed_change,

            "health_features_available":
                sequence[
                    "health_features_available"
                ].iloc[-1],
        }
    )


# ============================================================
# Results
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


print("\n" + "=" * 75)
print("REAL V5 DIGITAL TWIN RESULTS")
print("=" * 75)

print(
    results_df.to_string(
        index=False,
        formatters={
            "predicted_rul_days":
                "{:.4f}".format,

            "prediction_error_days":
                "{:.4f}".format,

            "absolute_error_days":
                "{:.4f}".format,

            "raw_v5_state_change":
                lambda x:
                    "N/A"
                    if pd.isna(x)
                    else f"{x:.6f}",

            "processed_input_change":
                lambda x:
                    "N/A"
                    if pd.isna(x)
                    else f"{x:.6f}",
        }
    )
)


# ============================================================
# Inspect final V5 states
# ============================================================

print("\n" + "=" * 75)
print("LATEST V5 STATE AT EACH CUT-OFF")
print("=" * 75)


display_features = [
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


for cutoff in CUTOFFS:

    cutoff_index = (
        len(pre_failure) - cutoff
    )

    if cutoff_index < 30:
        continue


    history = pre_failure.iloc[
        :cutoff_index
    ]

    latest = history.tail(
        30
    ).iloc[-1]


    actual_rul = (
        failure_date -
        latest["date"]
    ).days


    print(
        f"\nActual RUL: {actual_rul} days"
        f" | State date: "
        f"{latest['date'].date()}"
    )


    for feature in display_features:

        value = latest[feature]

        print(
            f"  {feature:<50} {value}"
        )


# ============================================================
# Save
# ============================================================

OUTPUT_DIR = r".\results\digital_twin_rul"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    f"{SERIAL_NUMBER}_v5_rul_trajectory.csv"
)


results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 75)

print(
    "Saved:"
)

print(
    OUTPUT_FILE
)

print("=" * 75)