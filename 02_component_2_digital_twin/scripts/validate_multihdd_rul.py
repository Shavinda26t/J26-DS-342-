import os
import random
import numpy as np
import pandas as pd

from digital_twin_rul_predictor import DigitalTwinRULPredictor


# ============================================================
# Configuration
# ============================================================

V5_FILE = r".\results\digital_twin_state_v5.csv"

OUTPUT_DIR = r".\results\digital_twin_rul"

RANDOM_SEED = 42

NUMBER_OF_HDDS = 50

# Number of observations BEFORE the failure observation.
# A 30-observation model input is required.
CHECKPOINTS = [30, 14, 7, 1]

CHUNKSIZE = 500_000


# ============================================================
# V5 columns required
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
# Step 1: Find failed HDDs
# ============================================================

print("=" * 80)
print("MULTI-HDD DIGITAL TWIN RUL VALIDATION")
print("=" * 80)

print("\nStep 1: Finding failed HDDs...")

failure_records = []

for chunk in pd.read_csv(
    V5_FILE,
    usecols=[
        "date",
        "serial_number",
        "failure",
    ],
    chunksize=CHUNKSIZE
):

    failed = chunk[
        chunk["failure"] == 1
    ]

    if not failed.empty:

        failure_records.append(
            failed
        )


if not failure_records:

    raise RuntimeError(
        "No failed HDDs found in V5."
    )


failure_df = pd.concat(
    failure_records,
    ignore_index=True
)

failure_df["date"] = pd.to_datetime(
    failure_df["date"]
)


# One failure date per HDD

failure_df = (
    failure_df
    .sort_values("date")
    .drop_duplicates(
        subset=["serial_number"],
        keep="first"
    )
    .reset_index(drop=True)
)


print(
    "Total failed HDDs found:",
    len(failure_df)
)


# ============================================================
# Step 2: Select HDD sample
# ============================================================

random.seed(
    RANDOM_SEED
)

available_serials = (
    failure_df["serial_number"]
    .astype(str)
    .unique()
    .tolist()
)

sample_size = min(
    NUMBER_OF_HDDS,
    len(available_serials)
)

selected_serials = random.sample(
    available_serials,
    sample_size
)

selected_serials = set(
    selected_serials
)

print(
    "Selected HDDs:",
    len(selected_serials)
)


# ============================================================
# Step 3: Load V5 rows for selected HDDs
# ============================================================

print(
    "\nStep 2: Loading V5 states for selected HDDs..."
)

records = []

for chunk in pd.read_csv(
    V5_FILE,
    usecols=USECOLS,
    chunksize=CHUNKSIZE
):

    chunk["serial_number"] = (
        chunk["serial_number"]
        .astype(str)
    )

    match = chunk[
        chunk["serial_number"]
        .isin(selected_serials)
    ]

    if not match.empty:

        records.append(
            match
        )


if not records:

    raise RuntimeError(
        "Selected HDDs were not found in V5."
    )


data = pd.concat(
    records,
    ignore_index=True
)

data["date"] = pd.to_datetime(
    data["date"]
)


data = data.sort_values(
    [
        "serial_number",
        "date",
    ]
).reset_index(
    drop=True
)


print(
    "Loaded observations:",
    len(data)
)


# ============================================================
# Step 4: Load predictor
# ============================================================

print(
    "\nStep 3: Loading trained BiLSTM..."
)

predictor = DigitalTwinRULPredictor()


# ============================================================
# Step 5: Evaluate each HDD
# ============================================================

print(
    "\nStep 4: Running RUL predictions..."
)

results = []

processed_hdds = 0

for serial, hdd in data.groupby(
    "serial_number"
):

    hdd = hdd.sort_values(
        "date"
    ).reset_index(
        drop=True
    )


    # --------------------------------------------------------
    # Find failure
    # --------------------------------------------------------

    failure_rows = hdd[
        hdd["failure"] == 1
    ]

    if failure_rows.empty:

        continue


    failure_date = failure_rows[
        "date"
    ].iloc[0]


    # Only observations before failure

    pre_failure = hdd[
        hdd["date"] < failure_date
    ].copy()

    pre_failure = (
        pre_failure
        .sort_values("date")
        .reset_index(drop=True)
    )


    if len(pre_failure) < 30:

        continue


    processed_hdds += 1


    # --------------------------------------------------------
    # Test each checkpoint
    # --------------------------------------------------------

    for checkpoint in CHECKPOINTS:

        cutoff_index = (
            len(pre_failure)
            - checkpoint
        )


        # Need at least 30 observations
        # to create the model input.

        if cutoff_index < 30:

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


        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

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


        smart187_available = int(
            sequence[
                "smart_187_available"
            ].iloc[-1]
        )


        health_features_available = int(
            sequence[
                "health_features_available"
            ].iloc[-1]
        )


        results.append(
            {
                "serial_number":
                    serial,

                "checkpoint_observations":
                    checkpoint,

                "sequence_start_date":
                    sequence[
                        "date"
                    ].iloc[0],

                "sequence_end_date":
                    sequence_end,

                "failure_date":
                    failure_date,

                "actual_rul_days":
                    actual_rul,

                "predicted_rul_days":
                    predicted_rul,

                "error_days":
                    error,

                "absolute_error_days":
                    absolute_error,

                "smart_187_available":
                    smart187_available,

                "health_features_available":
                    health_features_available,
            }
        )


    if processed_hdds % 10 == 0:

        print(
            f"Processed HDDs: "
            f"{processed_hdds}"
        )


# ============================================================
# Results dataframe
# ============================================================

results_df = pd.DataFrame(
    results
)


if results_df.empty:

    raise RuntimeError(
        "No validation predictions were generated."
    )


# ============================================================
# Overall metrics
# ============================================================

def calculate_metrics(df):

    errors = df[
        "error_days"
    ].to_numpy(
        dtype=float
    )

    absolute_errors = np.abs(
        errors
    )


    mae = np.mean(
        absolute_errors
    )

    rmse = np.sqrt(
        np.mean(
            errors ** 2
        )
    )

    bias = np.mean(
        errors
    )


    return {
        "n": len(df),
        "MAE_days": mae,
        "RMSE_days": rmse,
        "Mean_bias_days": bias,
    }


# ============================================================
# Metrics by checkpoint
# ============================================================

checkpoint_metrics = []

for checkpoint, group in results_df.groupby(
    "checkpoint_observations"
):

    metrics = calculate_metrics(
        group
    )

    metrics[
        "checkpoint_observations"
    ] = checkpoint

    checkpoint_metrics.append(
        metrics
    )


checkpoint_metrics_df = pd.DataFrame(
    checkpoint_metrics
).sort_values(
    "checkpoint_observations"
)


# ============================================================
# Metrics by SMART187 availability
# ============================================================

smart187_metrics = []

for availability, group in results_df.groupby(
    "smart_187_available"
):

    metrics = calculate_metrics(
        group
    )

    metrics[
        "smart_187_available"
    ] = availability

    smart187_metrics.append(
        metrics
    )


smart187_metrics_df = pd.DataFrame(
    smart187_metrics
).sort_values(
    "smart_187_available"
)


# ============================================================
# RUL range metrics
# ============================================================

def rul_range(actual_rul):

    if actual_rul <= 7:
        return "1-7"

    elif actual_rul <= 14:
        return "8-14"

    elif actual_rul <= 30:
        return "15-30"

    elif actual_rul <= 60:
        return "31-60"

    else:
        return ">60"


results_df[
    "actual_rul_range"
] = results_df[
    "actual_rul_days"
].apply(
    rul_range
)


rul_range_metrics = []

for rul_range_name, group in results_df.groupby(
    "actual_rul_range"
):

    metrics = calculate_metrics(
        group
    )

    metrics[
        "actual_rul_range"
    ] = rul_range_name

    rul_range_metrics.append(
        metrics
    )


rul_range_metrics_df = pd.DataFrame(
    rul_range_metrics
)


# ============================================================
# Print results
# ============================================================

print("\n" + "=" * 80)
print("VALIDATION COMPLETE")
print("=" * 80)

print(
    "\nUnique HDDs evaluated:",
    results_df[
        "serial_number"
    ].nunique()
)

print(
    "Total predictions:",
    len(results_df)
)


print("\n" + "-" * 80)
print("METRICS BY CHECKPOINT")
print("-" * 80)

print(
    checkpoint_metrics_df.to_string(
        index=False,
        formatters={
            "MAE_days":
                "{:.3f}".format,

            "RMSE_days":
                "{:.3f}".format,

            "Mean_bias_days":
                "{:.3f}".format,
        }
    )
)


print("\n" + "-" * 80)
print("METRICS BY SMART187 AVAILABILITY")
print("-" * 80)

print(
    smart187_metrics_df.to_string(
        index=False,
        formatters={
            "MAE_days":
                "{:.3f}".format,

            "RMSE_days":
                "{:.3f}".format,

            "Mean_bias_days":
                "{:.3f}".format,
        }
    )
)


print("\n" + "-" * 80)
print("METRICS BY ACTUAL RUL RANGE")
print("-" * 80)

print(
    rul_range_metrics_df.to_string(
        index=False,
        formatters={
            "MAE_days":
                "{:.3f}".format,

            "RMSE_days":
                "{:.3f}".format,

            "Mean_bias_days":
                "{:.3f}".format,
        }
    )
)


# ============================================================
# Save outputs
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


detailed_file = os.path.join(
    OUTPUT_DIR,
    "multihdd_rul_validation_predictions.csv"
)

checkpoint_file = os.path.join(
    OUTPUT_DIR,
    "multihdd_rul_checkpoint_metrics.csv"
)

smart187_file = os.path.join(
    OUTPUT_DIR,
    "multihdd_rul_smart187_metrics.csv"
)

rul_range_file = os.path.join(
    OUTPUT_DIR,
    "multihdd_rul_range_metrics.csv"
)


results_df.to_csv(
    detailed_file,
    index=False
)

checkpoint_metrics_df.to_csv(
    checkpoint_file,
    index=False
)

smart187_metrics_df.to_csv(
    smart187_file,
    index=False
)

rul_range_metrics_df.to_csv(
    rul_range_file,
    index=False
)


print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)

print(detailed_file)
print(checkpoint_file)
print(smart187_file)
print(rul_range_file)

print("=" * 80)