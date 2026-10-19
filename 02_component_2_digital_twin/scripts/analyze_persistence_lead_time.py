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
    "persistence_lead_time_analysis.csv"
)

SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "persistence_lead_time_summary.csv"
)

CHUNKSIZE = 500_000

# Candidate persistence rule from previous experiment
PERSISTENCE_LENGTH = 7

SMART1 = "smart_1_normalized_change_7d"
SMART187 = "smart_187_normalized_change_7d"


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
# FIND FIRST PERSISTENT MODERATE+ EVENT
# ============================================================

def find_first_persistent_event(group):

    group = group.sort_values("date").copy()

    values = group[
        "combined_severity"
    ].to_numpy(dtype=float)

    dates = group[
        "date"
    ].to_numpy()

    consecutive = 0

    for i, value in enumerate(values):

        # Missing value breaks persistence
        if pd.isna(value) or value < 2:

            consecutive = 0
            continue

        consecutive += 1

        if consecutive >= PERSISTENCE_LENGTH:

            # Date of the FIRST observation in
            # the persistent 7-observation sequence
            start_index = (
                i - PERSISTENCE_LENGTH + 1
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
print("PERSISTENCE LEAD-TIME ANALYSIS")
print("=" * 70)

print(
    f"\nPersistence rule: "
    f"{PERSISTENCE_LENGTH} consecutive Moderate+ observations"
)

# ============================================================
# PASS 1
# Identify failed HDDs and failure dates
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
    f"\nFailed HDDs identified: "
    f"{len(failure_dates):,}"
)

# ============================================================
# PASS 2
# Read failed HDD trajectories
# ============================================================

print(
    "\nPASS 2: Building failed-HDD "
    "degradation trajectories..."
)

usecols = [
    "date",
    "serial_number",
    "failure",
    SMART1,
    SMART187
]

failed_records = []

chunk_number = 0

failed_serials = set(
    failure_dates.keys()
)

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

    # Keep observations from 60 days before
    # failure through failure.
    chunk["failure_date"] = (
        chunk["serial_number"]
        .map(failure_dates)
    )

    chunk["days_to_failure"] = (
        chunk["failure_date"]
        - chunk["date"]
    ).dt.days

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

    failed_records.append(
        chunk[
            [
                "date",
                "serial_number",
                "failure_date",
                "days_to_failure",
                "smart1_severity",
                "smart187_severity",
                "combined_severity"
            ]
        ]
    )


print("\nCombining trajectories...")

df = pd.concat(
    failed_records,
    ignore_index=True
)

df = df.sort_values(
    [
        "serial_number",
        "date"
    ]
)

# ============================================================
# FIND FIRST PERSISTENT EVENT
# ============================================================

print(
    "\nFinding first persistent "
    "degradation event for each failed HDD..."
)

lead_time_results = []

for serial, group in df.groupby(
    "serial_number",
    sort=False
):

    failure_date = group[
        "failure_date"
    ].iloc[0]

    first_event_date = (
        find_first_persistent_event(
            group
        )
    )

    if pd.isna(first_event_date):

        lead_time = np.nan

    else:

        lead_time = (
            failure_date
            - first_event_date
        ).days

    lead_time_results.append(
        {
            "serial_number": serial,
            "failure_date": failure_date,
            "first_persistent_event_date":
                first_event_date,
            "lead_time_days":
                lead_time
        }
    )


lead_df = pd.DataFrame(
    lead_time_results
)

# ============================================================
# SAVE HDD-LEVEL RESULTS
# ============================================================

lead_df.to_csv(
    OUTPUT_FILE,
    index=False
)

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PERSISTENCE LEAD-TIME SUMMARY")
print("=" * 70)

total_failed = len(lead_df)

detected = lead_df[
    lead_df["lead_time_days"].notna()
].copy()

detected_count = len(detected)

detection_rate = (
    100.0
    * detected_count
    / total_failed
)

print(
    f"\nTotal failed HDDs: "
    f"{total_failed:,}"
)

print(
    f"Detected by persistent rule: "
    f"{detected_count:,}"
)

print(
    f"Detection rate: "
    f"{detection_rate:.2f}%"
)

if detected_count > 0:

    print(
        f"\nLead-time statistics:"
    )

    print(
        f"Mean:   "
        f"{detected['lead_time_days'].mean():.2f} days"
    )

    print(
        f"Median: "
        f"{detected['lead_time_days'].median():.2f} days"
    )

    print(
        f"Minimum:"
        f" {detected['lead_time_days'].min():.0f} days"
    )

    print(
        f"Maximum:"
        f" {detected['lead_time_days'].max():.0f} days"
    )

    print("\nLead-time distribution:")

    bins = [
        0,
        1,
        3,
        7,
        14,
        30,
        60
    ]

    labels = [
        "0-1 days",
        "2-3 days",
        "4-7 days",
        "8-14 days",
        "15-30 days",
        "31-60 days"
    ]

    categories = pd.cut(
        detected["lead_time_days"],
        bins=[
            -1,
            1,
            3,
            7,
            14,
            30,
            60
        ],
        labels=labels
    )

    distribution = (
        categories
        .value_counts()
        .sort_index()
    )

    for label, count in distribution.items():

        pct = (
            100.0
            * count
            / detected_count
        )

        print(
            f"{label:>10}: "
            f"{count:5d} "
            f"({pct:6.2f}%)"
        )


# ============================================================
# SAVE SUMMARY TABLE
# ============================================================

summary_rows = [
    {
        "persistence_observations":
            PERSISTENCE_LENGTH,
        "total_failed_hdds":
            total_failed,
        "detected_failed_hdds":
            detected_count,
        "detection_rate_pct":
            detection_rate,
        "mean_lead_time_days":
            (
                detected["lead_time_days"].mean()
                if detected_count > 0
                else np.nan
            ),
        "median_lead_time_days":
            (
                detected["lead_time_days"].median()
                if detected_count > 0
                else np.nan
            ),
        "minimum_lead_time_days":
            (
                detected["lead_time_days"].min()
                if detected_count > 0
                else np.nan
            ),
        "maximum_lead_time_days":
            (
                detected["lead_time_days"].max()
                if detected_count > 0
                else np.nan
            )
    }
]

summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)

# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS COMPLETED")
print("=" * 70)

print(
    "\nHDD-level results:"
)

print(OUTPUT_FILE)

print(
    "\nSummary:"
)

print(SUMMARY_FILE)