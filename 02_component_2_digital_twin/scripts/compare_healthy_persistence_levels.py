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
    "healthy_persistence_levels_comparison.csv"
)

SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "healthy_persistence_levels_summary.csv"
)

CHUNKSIZE = 500_000

SMART1 = "smart_1_normalized_change_7d"
SMART187 = "smart_187_normalized_change_7d"

PERSISTENCE_LEVELS = [3, 5, 7]


# ============================================================
# SAME THRESHOLDS USED THROUGHOUT THE EXPERIMENTS
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
# FIND LONGEST MODERATE+ RUN
# ============================================================

def longest_moderate_plus_run(values):

    longest = 0
    current = 0

    for value in values:

        if pd.notna(value) and value >= 2:

            current += 1

            if current > longest:
                longest = current

        else:

            current = 0

    return longest


# ============================================================
# START
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

print("=" * 70)
print("HEALTHY HDD PERSISTENCE LEVEL COMPARISON")
print("=" * 70)

print(
    "\nTesting persistence levels:",
    PERSISTENCE_LEVELS
)

# ============================================================
# PASS 1
# IDENTIFY NO-FAILURE HDDs
# ============================================================

print(
    "\nPASS 1: Identifying HDD failure status..."
)

failure_status = {}

chunk_number = 0

for chunk in pd.read_csv(
    V5_FILE,
    usecols=[
        "serial_number",
        "failure"
    ],
    chunksize=CHUNKSIZE
):

    chunk_number += 1

    print(
        f"Processing failure-status chunk "
        f"{chunk_number}..."
    )

    grouped = (
        chunk
        .groupby("serial_number")["failure"]
        .max()
    )

    for serial, failure in grouped.items():

        if serial not in failure_status:
            failure_status[serial] = 0

        if failure == 1:
            failure_status[serial] = 1


no_failure_hdds = {
    serial
    for serial, failed in failure_status.items()
    if failed == 0
}

failed_hdds = {
    serial
    for serial, failed in failure_status.items()
    if failed == 1
}

print("\nPopulation:")
print(
    f"Total HDDs:       {len(failure_status):,}"
)

print(
    f"Failed HDDs:      {len(failed_hdds):,}"
)

print(
    f"No-failure HDDs:  {len(no_failure_hdds):,}"
)

# ============================================================
# PASS 2
# BUILD NO-FAILURE HDD TRAJECTORIES
# ============================================================

print(
    "\nPASS 2: Building no-failure HDD trajectories..."
)

usecols = [
    "date",
    "serial_number",
    "failure",
    SMART1,
    SMART187
]

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
            no_failure_hdds
        )
    ].copy()

    if chunk.empty:
        continue

    chunk["date"] = pd.to_datetime(
        chunk["date"]
    )

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
# CALCULATE LONGEST RUN FOR EACH HDD
# ============================================================

print(
    "\nCalculating longest Moderate+ "
    "run for each no-failure HDD..."
)

hdd_results = []

for serial, group in df.groupby(
    "serial_number",
    sort=False
):

    values = group[
        "combined_severity"
    ].to_numpy(dtype=float)

    longest_run = (
        longest_moderate_plus_run(
            values
        )
    )

    hdd_results.append(
        {
            "serial_number": serial,
            "longest_moderate_plus_run":
                longest_run
        }
    )


hdd_df = pd.DataFrame(
    hdd_results
)

# ============================================================
# COMPARE PERSISTENCE LEVELS
# ============================================================

print("\n" + "=" * 70)
print("HEALTHY HDD PERSISTENCE COMPARISON")
print("=" * 70)

summary_rows = []

total_hdds = len(hdd_df)

for persistence in PERSISTENCE_LEVELS:

    count = (
        hdd_df[
            "longest_moderate_plus_run"
        ] >= persistence
    ).sum()

    percentage = (
        100.0
        * count
        / total_hdds
    )

    summary_rows.append(
        {
            "persistence_observations":
                persistence,

            "no_failure_hdds":
                total_hdds,

            "hdds_reaching_persistence":
                count,

            "false_positive_rate_pct":
                percentage
        }
    )

    print(
        f"\n{persistence} consecutive observations:"
    )

    print(
        f"  HDDs: "
        f"{count:,} / {total_hdds:,}"
    )

    print(
        f"  Rate: "
        f"{percentage:.4f}%"
    )


# ============================================================
# SAVE
# ============================================================

summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    SUMMARY_FILE,
    index=False
)

hdd_df.to_csv(
    OUTPUT_FILE,
    index=False
)

# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("FINAL HEALTHY HDD COMPARISON")
print("=" * 70)

print(
    summary.to_string(
        index=False
    )
)

print("\nDetailed HDD results:")
print(OUTPUT_FILE)

print("\nSummary:")
print(SUMMARY_FILE)

print("\n" + "=" * 70)
print("ANALYSIS COMPLETED")
print("=" * 70)