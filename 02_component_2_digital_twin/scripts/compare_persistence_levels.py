import os
import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

V5_FILE = r".\results\digital_twin_state_v5.csv"

OUTPUT_DIR = r".\results\adaptive_digital_twin"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "persistence_levels_comparison.csv"
)

CHUNKSIZE = 500_000

SMART1 = "smart_1_normalized_change_7d"
SMART187 = "smart_187_normalized_change_7d"

PERSISTENCE_LEVELS = [3, 5, 7]


# ============================================================
# SAME EXPERIMENTAL SEVERITY THRESHOLDS
# ============================================================

def smart1_severity(x):

    if pd.isna(x):
        return np.nan

    if x >= 0:
        return 0
    elif x >= -2:
        return 1
    elif x >= -7:
        return 2
    elif x >= -14:
        return 3
    else:
        return 4


def smart187_severity(x):

    if pd.isna(x):
        return np.nan

    if x >= 0:
        return 0
    elif x >= -2:
        return 1
    elif x >= -8:
        return 2
    elif x >= -22:
        return 3
    else:
        return 4


# ============================================================
# COMBINE SMART1 + SMART187
# ============================================================

def add_combined_severity(df):

    s1 = df["smart1_severity"].to_numpy(
        dtype=float
    )

    s187 = df["smart187_severity"].to_numpy(
        dtype=float
    )

    combined = np.full(
        len(df),
        np.nan
    )

    valid1 = ~np.isnan(s1)
    valid187 = ~np.isnan(s187)

    both = valid1 & valid187
    only1 = valid1 & ~valid187
    only187 = ~valid1 & valid187

    combined[both] = np.maximum(
        s1[both],
        s187[both]
    )

    combined[only1] = s1[only1]
    combined[only187] = s187[only187]

    df["combined_severity"] = combined

    return df


# ============================================================
# FIND FIRST PERSISTENT EVENT
# ============================================================

def find_first_persistent_event(group, persistence):

    group = group.sort_values("date")

    values = group[
        "combined_severity"
    ].to_numpy(dtype=float)

    dates = group[
        "date"
    ].to_numpy()

    consecutive = 0

    for i, value in enumerate(values):

        if pd.isna(value) or value < 2:

            consecutive = 0
            continue

        consecutive += 1

        if consecutive >= persistence:

            start_index = (
                i - persistence + 1
            )

            return pd.Timestamp(
                dates[start_index]
            )

    return pd.NaT


# ============================================================
# START
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

print("=" * 70)
print("PERSISTENCE LEVEL COMPARISON")
print("=" * 70)

print(
    "\nTesting persistence levels:",
    PERSISTENCE_LEVELS
)

# ============================================================
# PASS 1 — FIND FAILURE DATES
# ============================================================

print("\nPASS 1: Finding failure dates...")

failure_dates = {}

chunk_number = 0

for chunk in pd.read_csv(
    V5_FILE,
    usecols=[
        "date",
        "serial_number",
        "failure"
    ],
    chunksize=CHUNKSIZE
):

    chunk_number += 1

    print(
        f"Processing failure chunk "
        f"{chunk_number}..."
    )

    chunk["date"] = pd.to_datetime(
        chunk["date"]
    )

    failed = chunk[
        chunk["failure"] == 1
    ]

    for row in failed.itertuples(
        index=False
    ):

        serial = row.serial_number
        date = row.date

        if (
            serial not in failure_dates
            or date < failure_dates[serial]
        ):

            failure_dates[serial] = date


print(
    f"\nFailed HDDs: "
    f"{len(failure_dates):,}"
)

# ============================================================
# PASS 2 — FAILED HDD TRAJECTORIES
# ============================================================

print(
    "\nPASS 2: Building failed-HDD trajectories..."
)

usecols = [
    "date",
    "serial_number",
    "failure",
    SMART1,
    SMART187
]

failed_serials = set(
    failure_dates.keys()
)

records = []

chunk_number = 0

for chunk in pd.read_csv(
    V5_FILE,
    usecols=usecols,
    chunksize=CHUNKSIZE
):

    chunk_number += 1

    print(
        f"Processing trajectory chunk "
        f"{chunk_number}..."
    )

    chunk = chunk[
        chunk["serial_number"].isin(
            failed_serials
        )
    ].copy()

    if chunk.empty:
        continue

    chunk["date"] = pd.to_datetime(
        chunk["date"]
    )

    chunk["failure_date"] = (
        chunk["serial_number"]
        .map(failure_dates)
    )

    chunk["days_to_failure"] = (
        chunk["failure_date"]
        - chunk["date"]
    ).dt.days

    # Use up to 60 days before failure
    chunk = chunk[
        (chunk["days_to_failure"] >= 0)
        &
        (chunk["days_to_failure"] <= 60)
    ].copy()

    if chunk.empty:
        continue

    chunk["smart1_severity"] = (
        chunk[SMART1]
        .apply(smart1_severity)
    )

    chunk["smart187_severity"] = (
        chunk[SMART187]
        .apply(smart187_severity)
    )

    chunk = add_combined_severity(
        chunk
    )

    records.append(
        chunk[
            [
                "date",
                "serial_number",
                "failure_date",
                "days_to_failure",
                "combined_severity"
            ]
        ]
    )


print("\nCombining trajectories...")

df = pd.concat(
    records,
    ignore_index=True
)

df = df.sort_values(
    [
        "serial_number",
        "date"
    ]
)

# ============================================================
# TEST EACH PERSISTENCE LEVEL
# ============================================================

all_results = []
summary_rows = []

for persistence in PERSISTENCE_LEVELS:

    print("\n" + "=" * 70)

    print(
        f"TESTING {persistence} CONSECUTIVE "
        f"MODERATE+ OBSERVATIONS"
    )

    print("=" * 70)

    results = []

    for serial, group in df.groupby(
        "serial_number",
        sort=False
    ):

        failure_date = group[
            "failure_date"
        ].iloc[0]

        first_event = (
            find_first_persistent_event(
                group,
                persistence
            )
        )

        if pd.isna(first_event):

            lead_time = np.nan

        else:

            lead_time = (
                failure_date
                - first_event
            ).days

        results.append(
            {
                "serial_number": serial,
                "persistence":
                    persistence,
                "failure_date":
                    failure_date,
                "first_event_date":
                    first_event,
                "lead_time_days":
                    lead_time
            }
        )

    result_df = pd.DataFrame(
        results
    )

    all_results.append(
        result_df
    )

    detected = result_df[
        result_df["lead_time_days"].notna()
    ]

    total = len(result_df)
    detected_count = len(detected)

    detection_rate = (
        100.0
        * detected_count
        / total
    )

    if detected_count > 0:

        mean_lead = (
            detected["lead_time_days"]
            .mean()
        )

        median_lead = (
            detected["lead_time_days"]
            .median()
        )

        minimum_lead = (
            detected["lead_time_days"]
            .min()
        )

        maximum_lead = (
            detected["lead_time_days"]
            .max()
        )

    else:

        mean_lead = np.nan
        median_lead = np.nan
        minimum_lead = np.nan
        maximum_lead = np.nan

    print(
        f"\nDetection rate: "
        f"{detection_rate:.2f}%"
    )

    print(
        f"Detected HDDs: "
        f"{detected_count:,} / {total:,}"
    )

    print(
        f"Mean lead time: "
        f"{mean_lead:.2f} days"
    )

    print(
        f"Median lead time: "
        f"{median_lead:.2f} days"
    )

    print(
        f"Minimum lead time: "
        f"{minimum_lead:.0f} days"
    )

    print(
        f"Maximum lead time: "
        f"{maximum_lead:.0f} days"
    )

    # --------------------------------------------------------
    # Lead-time ranges
    # --------------------------------------------------------

    if detected_count > 0:

        bins = [
            -1,
            3,
            7,
            14,
            30,
            60
        ]

        labels = [
            "0-3 days",
            "4-7 days",
            "8-14 days",
            "15-30 days",
            "31-60 days"
        ]

        categories = pd.cut(
            detected["lead_time_days"],
            bins=bins,
            labels=labels
        )

        distribution = (
            categories
            .value_counts()
            .sort_index()
        )

        print("\nLead-time distribution:")

        for label, count in distribution.items():

            pct = (
                100.0
                * count
                / detected_count
            )

            print(
                f"{str(label):>10}: "
                f"{count:5d} "
                f"({pct:6.2f}%)"
            )

    summary_rows.append(
        {
            "persistence_observations":
                persistence,

            "total_failed_hdds":
                total,

            "detected_failed_hdds":
                detected_count,

            "detection_rate_pct":
                detection_rate,

            "mean_lead_time_days":
                mean_lead,

            "median_lead_time_days":
                median_lead,

            "minimum_lead_time_days":
                minimum_lead,

            "maximum_lead_time_days":
                maximum_lead
        }
    )


# ============================================================
# SAVE RESULTS
# ============================================================

all_results_df = pd.concat(
    all_results,
    ignore_index=True
)

all_results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

summary = pd.DataFrame(
    summary_rows
)

SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "persistence_levels_summary.csv"
)

summary.to_csv(
    SUMMARY_FILE,
    index=False
)

# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("FINAL PERSISTENCE COMPARISON")
print("=" * 70)

print(
    summary.to_string(
        index=False
    )
)

print("\nSaved detailed results:")
print(OUTPUT_FILE)

print("\nSaved summary:")
print(SUMMARY_FILE)

print("\n" + "=" * 70)
print("ANALYSIS COMPLETED")
print("=" * 70)