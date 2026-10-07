import os
import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

V5_FILE = r".\results\digital_twin_state_v5.csv"

OUTPUT_DIR = r".\results\adaptive_digital_twin"

TRAJECTORY_FILE = os.path.join(
    OUTPUT_DIR,
    "healthy_degradation_validation.csv"
)

SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "healthy_degradation_summary.csv"
)

CHUNKSIZE = 500_000

SMART1 = "smart_1_normalized_change_7d"
SMART187 = "smart_187_normalized_change_7d"

SMART1_AVAILABLE = "smart_1_available"
SMART187_AVAILABLE = "smart_187_available"

# ============================================================
# SAME EXPERIMENTAL THRESHOLDS USED FOR FAILED HDD VALIDATION
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


def calculate_severity(series, detector):
    return series.apply(detector).astype(float)


# ============================================================
# START
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("HEALTHY HDD FALSE-POSITIVE VALIDATION")
print("=" * 70)

print("\nScanning V5 Digital Twin state...")

# ------------------------------------------------------------
# PASS 1:
# Identify HDDs with NO recorded failure in Q1 2023
# ------------------------------------------------------------

failure_status = {}

chunk_number = 0

for chunk in pd.read_csv(
    V5_FILE,
    usecols=["serial_number", "failure"],
    chunksize=CHUNKSIZE
):

    chunk_number += 1

    print(f"Processing failure-status chunk {chunk_number}...")

    grouped = chunk.groupby("serial_number")["failure"].max()

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

healthy_hdds = {
    serial
    for serial, failed in failure_status.items()
    if failed == 0
}

print("\n" + "=" * 70)
print("HDD POPULATION")
print("=" * 70)

print(f"Total HDDs:    {len(failure_status):,}")
print(f"Failed HDDs:   {len(failed_hdds):,}")
print(f"Healthy HDDs:  {len(healthy_hdds):,}")

# ------------------------------------------------------------
# PASS 2:
# Analyse only healthy HDDs
# ------------------------------------------------------------

print("\nScanning healthy HDD observations...")

results = []

chunk_number = 0

usecols = [
    "date",
    "serial_number",
    SMART1,
    SMART187,
    SMART1_AVAILABLE,
    SMART187_AVAILABLE
]

for chunk in pd.read_csv(
    V5_FILE,
    usecols=usecols,
    chunksize=CHUNKSIZE
):

    chunk_number += 1

    print(f"Processing healthy-data chunk {chunk_number}...")

    # Keep only HDDs with no recorded Q1 failure
    chunk = chunk[
        chunk["serial_number"].isin(healthy_hdds)
    ].copy()

    if chunk.empty:
        continue

    # --------------------------------------------------------
    # Calculate severity
    # --------------------------------------------------------

    chunk["smart1_severity"] = calculate_severity(
        chunk[SMART1],
        smart1_severity
    )

    chunk["smart187_severity"] = calculate_severity(
        chunk[SMART187],
        smart187_severity
    )

    # --------------------------------------------------------
    # Combine evidence
    #
    # MAX is used only when at least one signal is available.
    # Missing SMART187 does NOT reduce SMART1 evidence.
    # --------------------------------------------------------

    s1 = chunk["smart1_severity"].to_numpy(dtype=float)
    s187 = chunk["smart187_severity"].to_numpy(dtype=float)

    combined = np.full(len(chunk), np.nan)

    valid_s1 = ~np.isnan(s1)
    valid_s187 = ~np.isnan(s187)

    both = valid_s1 & valid_s187
    only_s1 = valid_s1 & ~valid_s187
    only_s187 = ~valid_s1 & valid_s187

    combined[both] = np.maximum(
        s1[both],
        s187[both]
    )

    combined[only_s1] = s1[only_s1]
    combined[only_s187] = s187[only_s187]

    chunk["combined_severity"] = combined

    # --------------------------------------------------------
    # Store only necessary columns
    # --------------------------------------------------------

    results.append(
        chunk[
            [
                "date",
                "serial_number",
                SMART1,
                SMART187,
                SMART1_AVAILABLE,
                SMART187_AVAILABLE,
                "smart1_severity",
                "smart187_severity",
                "combined_severity"
            ]
        ]
    )

# ------------------------------------------------------------
# Combine results
# ------------------------------------------------------------

print("\nCombining healthy HDD observations...")

if not results:
    raise RuntimeError("No healthy HDD observations were found.")

healthy_df = pd.concat(
    results,
    ignore_index=True
)

# ------------------------------------------------------------
# Save trajectory-level results
# ------------------------------------------------------------

healthy_df.to_csv(
    TRAJECTORY_FILE,
    index=False
)

print(f"\nHealthy trajectory file:")
print(TRAJECTORY_FILE)

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("HEALTHY HDD DEGRADATION SUMMARY")
print("=" * 70)

summary_rows = []

def percentage_at_least(series, threshold):
    valid = series.dropna()

    if len(valid) == 0:
        return np.nan

    return 100.0 * (valid >= threshold).mean()


for feature_name, series in [
    ("smart1", healthy_df["smart1_severity"]),
    ("smart187", healthy_df["smart187_severity"]),
    ("combined", healthy_df["combined_severity"])
]:

    valid = series.dropna()

    summary_rows.append({
        "feature": feature_name,
        "observations": len(valid),
        "mean_severity": valid.mean(),
        "moderate_or_higher_pct":
            percentage_at_least(series, 2),
        "strong_or_higher_pct":
            percentage_at_least(series, 3),
        "severe_pct":
            percentage_at_least(series, 4)
    })


summary = pd.DataFrame(summary_rows)

summary.to_csv(
    SUMMARY_FILE,
    index=False
)

print(summary.to_string(index=False))

# ============================================================
# HDD-LEVEL PERSISTENCE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("HEALTHY HDD PERSISTENCE ANALYSIS")
print("=" * 70)

# For every healthy HDD calculate maximum observed severity.
hdd_max = (
    healthy_df
    .groupby("serial_number")["combined_severity"]
    .max()
    .dropna()
)

total_healthy = len(hdd_max)

for threshold, label in [
    (1, "Mild+"),
    (2, "Moderate+"),
    (3, "Strong+"),
    (4, "Severe")
]:

    count = int((hdd_max >= threshold).sum())

    percentage = (
        100.0 * count / total_healthy
        if total_healthy > 0
        else np.nan
    )

    print(
        f"{label}: "
        f"{count:,} / {total_healthy:,} HDDs "
        f"({percentage:.2f}%)"
    )

# ------------------------------------------------------------
# Persistent degradation:
# Count HDDs with >=3 consecutive observations
# at Moderate+ severity.
# ------------------------------------------------------------

print("\nChecking persistent Moderate+ degradation...")

healthy_df["date"] = pd.to_datetime(
    healthy_df["date"]
)

healthy_df = healthy_df.sort_values(
    ["serial_number", "date"]
)

healthy_df["moderate_plus"] = (
    healthy_df["combined_severity"] >= 2
)

persistent_hdds = set()

for serial, group in healthy_df.groupby(
    "serial_number",
    sort=False
):

    values = group["moderate_plus"].fillna(False).to_numpy()

    consecutive = 0

    for value in values:

        if value:
            consecutive += 1

            if consecutive >= 3:
                persistent_hdds.add(serial)
                break

        else:
            consecutive = 0


persistent_count = len(persistent_hdds)

persistent_percentage = (
    100.0 * persistent_count / total_healthy
    if total_healthy > 0
    else np.nan
)

print(
    f"Healthy HDDs with >=3 consecutive "
    f"Moderate+ observations: "
    f"{persistent_count:,} / {total_healthy:,} "
    f"({persistent_percentage:.2f}%)"
)

# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION COMPLETED")
print("=" * 70)

print("\nSummary file:")
print(SUMMARY_FILE)

print("\nTrajectory file:")
print(TRAJECTORY_FILE)