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
    "degradation_persistence_comparison.csv"
)

CHUNKSIZE = 500_000

SMART1 = "smart_1_normalized_change_7d"
SMART187 = "smart_187_normalized_change_7d"

# Same candidate thresholds already tested
# ------------------------------------------------------------
# SMART1:
# 0 = normal
# 1 = mild
# 2 = moderate
# 3 = strong
# 4 = severe
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# SMART187
# ------------------------------------------------------------

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
# START
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("DEGRADATION PERSISTENCE COMPARISON")
print("=" * 70)

# ============================================================
# PASS 1
# Identify failed / non-failed HDDs
# ============================================================

print("\nPASS 1: Identifying HDD failure status...")

failure_status = {}

chunk_number = 0

for chunk in pd.read_csv(
    V5_FILE,
    usecols=["serial_number", "failure"],
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


failed_hdds = {
    serial
    for serial, failed in failure_status.items()
    if failed == 1
}

nonfailed_hdds = {
    serial
    for serial, failed in failure_status.items()
    if failed == 0
}

print("\nPopulation:")
print(f"Total HDDs:       {len(failure_status):,}")
print(f"Failed HDDs:      {len(failed_hdds):,}")
print(f"No-failure HDDs:  {len(nonfailed_hdds):,}")


# ============================================================
# PASS 2
# Build severity sequences
# ============================================================

print("\nPASS 2: Building degradation sequences...")

usecols = [
    "date",
    "serial_number",
    "failure",
    SMART1,
    SMART187
]

failed_records = []
nonfailed_records = []

chunk_number = 0

for chunk in pd.read_csv(
    V5_FILE,
    usecols=usecols,
    chunksize=CHUNKSIZE
):

    chunk_number += 1

    print(
        f"Processing data chunk "
        f"{chunk_number}..."
    )

    # --------------------------------------------------------
    # Keep only required HDDs
    # --------------------------------------------------------

    failed_part = chunk[
        chunk["serial_number"].isin(failed_hdds)
    ].copy()

    nonfailed_part = chunk[
        chunk["serial_number"].isin(nonfailed_hdds)
    ].copy()

    # --------------------------------------------------------
    # Calculate individual severities
    # --------------------------------------------------------

    if not failed_part.empty:

        failed_part["smart1_severity"] = (
            failed_part[SMART1]
            .apply(smart1_severity)
        )

        failed_part["smart187_severity"] = (
            failed_part[SMART187]
            .apply(smart187_severity)
        )

        failed_records.append(
            failed_part[
                [
                    "date",
                    "serial_number",
                    "failure",
                    "smart1_severity",
                    "smart187_severity"
                ]
            ]
        )

    if not nonfailed_part.empty:

        nonfailed_part["smart1_severity"] = (
            nonfailed_part[SMART1]
            .apply(smart1_severity)
        )

        nonfailed_part["smart187_severity"] = (
            nonfailed_part[SMART187]
            .apply(smart187_severity)
        )

        nonfailed_records.append(
            nonfailed_part[
                [
                    "date",
                    "serial_number",
                    "failure",
                    "smart1_severity",
                    "smart187_severity"
                ]
            ]
        )


print("\nCombining records...")

failed_df = pd.concat(
    failed_records,
    ignore_index=True
)

nonfailed_df = pd.concat(
    nonfailed_records,
    ignore_index=True
)

# ============================================================
# COMBINED SEVERITY
# ============================================================

print("Calculating combined severity...")

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


failed_df = add_combined_severity(failed_df)
nonfailed_df = add_combined_severity(nonfailed_df)

# ============================================================
# SORT
# ============================================================

failed_df["date"] = pd.to_datetime(
    failed_df["date"]
)

nonfailed_df["date"] = pd.to_datetime(
    nonfailed_df["date"]
)

failed_df = failed_df.sort_values(
    ["serial_number", "date"]
)

nonfailed_df = nonfailed_df.sort_values(
    ["serial_number", "date"]
)

# ============================================================
# PERSISTENCE FUNCTION
# ============================================================

def longest_consecutive_run(values):
    """
    Find longest consecutive run where severity >= 2.

    Missing values break the run.
    """

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


def calculate_hdd_persistence(df):

    results = []

    for serial, group in df.groupby(
        "serial_number",
        sort=False
    ):

        values = group[
            "combined_severity"
        ].to_numpy(dtype=float)

        longest = longest_consecutive_run(
            values
        )

        results.append(
            {
                "serial_number": serial,
                "longest_moderate_plus_run": longest
            }
        )

    return pd.DataFrame(results)


print("\nCalculating HDD-level persistence...")

failed_persistence = calculate_hdd_persistence(
    failed_df
)

nonfailed_persistence = calculate_hdd_persistence(
    nonfailed_df
)

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PERSISTENCE COMPARISON")
print("=" * 70)

thresholds = [1, 3, 5, 7, 14]

summary_rows = []

for threshold in thresholds:

    failed_count = (
        failed_persistence[
            "longest_moderate_plus_run"
        ] >= threshold
    ).sum()

    nonfailed_count = (
        nonfailed_persistence[
            "longest_moderate_plus_run"
        ] >= threshold
    ).sum()

    failed_total = len(failed_persistence)
    nonfailed_total = len(nonfailed_persistence)

    failed_pct = (
        100.0 * failed_count / failed_total
    )

    nonfailed_pct = (
        100.0 * nonfailed_count / nonfailed_total
    )

    # Ratio of prevalence
    prevalence_ratio = (
        failed_pct / nonfailed_pct
        if nonfailed_pct > 0
        else np.inf
    )

    summary_rows.append(
        {
            "minimum_consecutive_observations":
                threshold,

            "failed_hdds":
                failed_count,

            "failed_pct":
                failed_pct,

            "nonfailed_hdds":
                nonfailed_count,

            "nonfailed_pct":
                nonfailed_pct,

            "failed_to_nonfailed_ratio":
                prevalence_ratio
        }
    )


summary = pd.DataFrame(summary_rows)

print(
    summary.to_string(
        index=False
    )
)

# ============================================================
# MEAN / MEDIAN RUN LENGTH
# ============================================================

print("\n" + "=" * 70)
print("LONGEST MODERATE+ RUN")
print("=" * 70)

print(
    f"Failed HDDs:"
)

print(
    f"  Mean:   "
    f"{failed_persistence['longest_moderate_plus_run'].mean():.2f}"
)

print(
    f"  Median: "
    f"{failed_persistence['longest_moderate_plus_run'].median():.2f}"
)

print(
    f"  Max:    "
    f"{failed_persistence['longest_moderate_plus_run'].max():.0f}"
)

print(
    f"\nNo-failure HDDs:"
)

print(
    f"  Mean:   "
    f"{nonfailed_persistence['longest_moderate_plus_run'].mean():.2f}"
)

print(
    f"  Median: "
    f"{nonfailed_persistence['longest_moderate_plus_run'].median():.2f}"
)

print(
    f"  Max:    "
    f"{nonfailed_persistence['longest_moderate_plus_run'].max():.0f}"
)

# ============================================================
# SAVE
# ============================================================

summary.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("ANALYSIS COMPLETED")
print("=" * 70)

print(
    f"\nSaved summary:"
)

print(OUTPUT_FILE)