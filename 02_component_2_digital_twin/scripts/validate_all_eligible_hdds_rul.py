import os
import numpy as np
import pandas as pd

from digital_twin_rul_predictor import DigitalTwinRULPredictor


# ============================================================
# Configuration
# ============================================================

V5_FILE = r".\results\digital_twin_state_v5.csv"

OUTPUT_DIR = r".\results\digital_twin_rul"

CHECKPOINTS = [30, 14, 7, 1]

CHUNKSIZE = 500_000


# ============================================================
# V5 columns
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
# Helper
# ============================================================

def calculate_metrics(df):

    errors = df[
        "error_days"
    ].to_numpy(dtype=float)

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


def rul_range(value):

    if value <= 7:
        return "1-7"

    if value <= 14:
        return "8-14"

    if value <= 30:
        return "15-30"

    if value <= 60:
        return "31-60"

    return ">60"


# ============================================================
# Start
# ============================================================

print("=" * 80)
print("FULL ELIGIBLE-HDD DIGITAL TWIN RUL VALIDATION")
print("=" * 80)


# ============================================================
# STEP 1
# Find failure date and observation count for every HDD
# ============================================================

print(
    "\nStep 1: Scanning V5 to identify eligible failed HDDs..."
)

hdd_summary = {}

for chunk in pd.read_csv(
    V5_FILE,
    usecols=[
        "date",
        "serial_number",
        "failure",
    ],
    chunksize=CHUNKSIZE
):

    chunk["serial_number"] = (
        chunk["serial_number"]
        .astype(str)
    )

    chunk["date"] = pd.to_datetime(
        chunk["date"]
    )

    for serial, group in chunk.groupby(
        "serial_number"
    ):

        if serial not in hdd_summary:

            hdd_summary[serial] = {
                "total_observations": 0,
                "pre_failure_observations": 0,
                "failure_date": None,
            }


        summary = hdd_summary[serial]

        summary[
            "total_observations"
        ] += len(group)


        failed_rows = group[
            group["failure"] == 1
        ]

        if not failed_rows.empty:

            failure_date = (
                failed_rows["date"]
                .min()
            )

            if (
                summary["failure_date"]
                is None
                or
                failure_date
                <
                summary["failure_date"]
            ):

                summary[
                    "failure_date"
                ] = failure_date


# ------------------------------------------------------------
# Calculate pre-failure observations
# ------------------------------------------------------------

# We need another scan because chunks can contain
# observations both before and after failure.

print(
    "\nCalculating pre-failure observation counts..."
)

for serial in hdd_summary:

    hdd_summary[serial][
        "pre_failure_observations"
    ] = 0


failure_dates = {
    serial: info["failure_date"]
    for serial, info
    in hdd_summary.items()
    if info["failure_date"] is not None
}


for chunk in pd.read_csv(
    V5_FILE,
    usecols=[
        "date",
        "serial_number",
    ],
    chunksize=CHUNKSIZE
):

    chunk["serial_number"] = (
        chunk["serial_number"]
        .astype(str)
    )

    chunk["date"] = pd.to_datetime(
        chunk["date"]
    )

    for serial, failure_date in failure_dates.items():

        mask = (
            chunk["serial_number"]
            == serial
        )

        if not mask.any():
            continue

        serial_dates = chunk.loc[
            mask,
            "date"
        ]

        count = (
            serial_dates
            < failure_date
        ).sum()
    # The above loop is intentionally replaced
    # by the efficient vectorised calculation below.


# ============================================================
# Rebuild summary efficiently
# ============================================================

print(
    "\nStep 1b: Building exact failed-HDD eligibility..."
)

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

    chunk["serial_number"] = (
        chunk["serial_number"]
        .astype(str)
    )

    chunk["date"] = pd.to_datetime(
        chunk["date"]
    )

    failed = chunk[
        chunk["failure"] == 1
    ]

    if not failed.empty:

        failure_records.append(
            failed[
                [
                    "serial_number",
                    "date",
                ]
            ]
        )


failure_df = pd.concat(
    failure_records,
    ignore_index=True
)

failure_df = (
    failure_df
    .sort_values("date")
    .drop_duplicates(
        subset=["serial_number"],
        keep="first"
    )
    .reset_index(drop=True)
)

failure_df["serial_number"] = (
    failure_df["serial_number"]
    .astype(str)
)

failure_df["date"] = pd.to_datetime(
    failure_df["date"]
)


failure_date_map = dict(
    zip(
        failure_df["serial_number"],
        failure_df["date"]
    )
)


# ============================================================
# Count pre-failure observations exactly
# ============================================================

pre_failure_counts = {
    serial: 0
    for serial in failure_date_map
}


for chunk in pd.read_csv(
    V5_FILE,
    usecols=[
        "date",
        "serial_number",
    ],
    chunksize=CHUNKSIZE
):

    chunk["serial_number"] = (
        chunk["serial_number"]
        .astype(str)
    )

    chunk["date"] = pd.to_datetime(
        chunk["date"]
    )

    chunk = chunk[
        chunk["serial_number"]
        .isin(
            failure_date_map.keys()
        )
    ]

    if chunk.empty:
        continue


    for serial, group in chunk.groupby(
        "serial_number"
    ):

        failure_date = (
            failure_date_map[serial]
        )

        pre_failure_counts[serial] += (
            group["date"]
            < failure_date
        ).sum()


# ============================================================
# Eligibility
# ============================================================

eligible_serials = [
    serial
    for serial, count
    in pre_failure_counts.items()
    if count >= 30
]


print(
    "\nTotal failed HDDs:",
    len(failure_date_map)
)

print(
    "HDDs with >=30 pre-failure observations:",
    len(eligible_serials)
)


if not eligible_serials:

    raise RuntimeError(
        "No eligible HDDs found."
    )


# Save eligibility information

eligibility_df = pd.DataFrame(
    {
        "serial_number":
            list(failure_date_map.keys()),

        "failure_date":
            [
                failure_date_map[s]
                for s in failure_date_map
            ],

        "pre_failure_observations":
            [
                pre_failure_counts[s]
                for s in failure_date_map
            ],
    }
)

eligibility_df[
    "eligible_for_30_step"
] = (
    eligibility_df[
        "pre_failure_observations"
    ] >= 30
)


# ============================================================
# STEP 2
# Load only eligible HDDs
# ============================================================

print(
    "\nStep 2: Loading V5 states for all eligible HDDs..."
)

eligible_set = set(
    eligible_serials
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
        .isin(eligible_set)
    ]

    if not match.empty:

        records.append(
            match
        )


if not records:

    raise RuntimeError(
        "Eligible HDD states could not be loaded."
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
# STEP 3
# Load model
# ============================================================

print(
    "\nStep 3: Loading trained BiLSTM..."
)

predictor = DigitalTwinRULPredictor()


# ============================================================
# STEP 4
# Evaluate ALL eligible HDDs
# ============================================================

print(
    "\nStep 4: Evaluating all eligible HDDs..."
)

results = []

processed_hdds = 0


for serial, hdd in data.groupby(
    "serial_number"
):

    hdd = (
        hdd
        .sort_values("date")
        .reset_index(drop=True)
    )


    failure_rows = hdd[
        hdd["failure"] == 1
    ]

    if failure_rows.empty:
        continue


    failure_date = (
        failure_rows["date"]
        .iloc[0]
    )


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


    for checkpoint in CHECKPOINTS:

        cutoff_index = (
            len(pre_failure)
            - checkpoint
        )


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


        predicted_rul = (
            predictor.predict(
                sequence
            )
        )


        error = (
            predicted_rul -
            actual_rul
        )


        smart187_available = int(
            sequence[
                "smart_187_available"
            ].iloc[-1]
        )


        health_available = int(
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
                    abs(error),

                "smart_187_available":
                    smart187_available,

                "health_features_available":
                    health_available,
            }
        )


    if processed_hdds % 50 == 0:

        print(
            "Processed HDDs:",
            processed_hdds
        )


# ============================================================
# Results
# ============================================================

results_df = pd.DataFrame(
    results
)


if results_df.empty:

    raise RuntimeError(
        "No predictions were generated."
    )


results_df[
    "actual_rul_range"
] = results_df[
    "actual_rul_days"
].apply(
    rul_range
)


# ============================================================
# Overall metrics
# ============================================================

overall_metrics = pd.DataFrame(
    [
        calculate_metrics(
            results_df
        )
    ]
)


# ============================================================
# Checkpoint metrics
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
# SMART187 metrics
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
# RUL-range metrics
# ============================================================

rul_range_metrics = []

for range_name, group in results_df.groupby(
    "actual_rul_range"
):

    metrics = calculate_metrics(
        group
    )

    metrics[
        "actual_rul_range"
    ] = range_name

    rul_range_metrics.append(
        metrics
    )


rul_range_metrics_df = pd.DataFrame(
    rul_range_metrics
)


# ============================================================
# HDD-level coverage
# ============================================================

hdd_level = (
    results_df
    .groupby("serial_number")
    .agg(
        checkpoints_evaluated=(
            "checkpoint_observations",
            "count"
        ),
        mean_absolute_error_days=(
            "absolute_error_days",
            "mean"
        ),
    )
    .reset_index()
)


# ============================================================
# Print
# ============================================================

print("\n" + "=" * 80)
print("FULL VALIDATION COMPLETE")
print("=" * 80)

print(
    "\nTotal failed HDDs:",
    len(failure_date_map)
)

print(
    "Eligible HDDs (>=30 pre-failure observations):",
    len(eligible_serials)
)

print(
    "HDDs actually evaluated:",
    results_df[
        "serial_number"
    ].nunique()
)

print(
    "Total predictions:",
    len(results_df)
)


print("\n" + "-" * 80)
print("OVERALL METRICS")
print("-" * 80)

print(
    overall_metrics.to_string(
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
# Save results
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


files_to_save = {

    "all_eligible_hdd_predictions.csv":
        results_df,

    "all_eligible_hdd_checkpoint_metrics.csv":
        checkpoint_metrics_df,

    "all_eligible_hdd_smart187_metrics.csv":
        smart187_metrics_df,

    "all_eligible_hdd_rul_range_metrics.csv":
        rul_range_metrics_df,

    "all_eligible_hdd_eligibility.csv":
        eligibility_df,

    "all_eligible_hdd_summary.csv":
        hdd_level,
}


print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)


for filename, dataframe in files_to_save.items():

    path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    dataframe.to_csv(
        path,
        index=False
    )

    print(path)


print("\n" + "=" * 80)
print("DONE")
print("=" * 80)