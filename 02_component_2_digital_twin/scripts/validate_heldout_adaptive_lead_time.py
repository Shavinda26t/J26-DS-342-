
"""
Validate adaptive Digital Twin degradation detection lead time
on the original held-out HDD test set.

Method:
1. Load the original held-out test HDD IDs.
2. Read the compact Digital Twin state CSV in chunks.
3. Identify each HDD's recorded failure date.
4. Evaluate observations before the failure date.
5. Apply the same SMART1 / SMART187 severity thresholds used by
   adaptive_digital_twin.py.
6. Define detection when Moderate-or-higher degradation persists
   for 7 consecutive observations.
7. Count detections confirmed within 0-60 calendar days before failure.
8. Save per-HDD results, summary metrics, and lead-time distribution.

Important:
- Persistence counts consecutive observations, not necessarily calendar days.
- Lead time is measured in calendar days.
- This evaluates degradation detection, not RUL prediction accuracy.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. Paths and configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_HDDS_PATH = (
    PROJECT_ROOT / "results" / "rul_sequences" / "test_hdds.csv"
)

STATE_PATH = PROJECT_ROOT / "results" / "digital_twin_state_v5.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "digital_twin_rul"

RESULTS_PATH = OUTPUT_DIR / "heldout_adaptive_lead_time_results.csv"
SUMMARY_PATH = OUTPUT_DIR / "heldout_adaptive_lead_time_summary.csv"
DISTRIBUTION_PATH = (
    OUTPUT_DIR / "heldout_adaptive_lead_time_distribution.csv"
)

CHUNK_SIZE = 250_000
PERSISTENCE_OBSERVATIONS = 7
MAX_LEAD_TIME_DAYS = 60

SMART1_COLUMN = "smart_1_normalized_change_7d"
SMART187_COLUMN = "smart_187_normalized_change_7d"

REQUIRED_STATE_COLUMNS = [
    "date",
    "serial_number",
    "failure",
    SMART1_COLUMN,
    SMART187_COLUMN,
]


# ============================================================
# 2. Severity thresholds
#    These match adaptive_digital_twin.py
# ============================================================

def severity_smart1(value):
    """Return SMART 1 degradation severity from 0 to 4."""

    if pd.isna(value):
        return 0

    if value <= -15:
        return 4
    elif value <= -8:
        return 3
    elif value <= -3:
        return 2
    elif value <= -1:
        return 1

    return 0


def severity_smart187(value):
    """Return SMART 187 degradation severity from 0 to 4."""

    if pd.isna(value):
        return 0

    if value <= -23:
        return 4
    elif value <= -9:
        return 3
    elif value <= -3:
        return 2
    elif value <= -1:
        return 1

    return 0


# ============================================================
# 3. Load held-out HDD identifiers
# ============================================================

def load_test_hdd_ids(path):
    """Load held-out HDD serial numbers from the test HDD CSV."""

    if not path.exists():
        raise FileNotFoundError(
            f"Test HDD file not found: {path}"
        )

    test_df = pd.read_csv(path, dtype=str)

    if test_df.empty:
        raise ValueError(
            f"The held-out HDD file is empty: {path}"
        )

    # Identify the serial-number column.
    possible_names = [
        "serial_number",
        "serial",
        "serialnumber",
        "hdd_serial_number",
    ]

    normalized_names = {
        str(column).strip().lower().replace(" ", "_"): column
        for column in test_df.columns
    }

    serial_column = None

    for name in possible_names:
        if name in normalized_names:
            serial_column = normalized_names[name]
            break

    # Some split files contain only a single serial-number column.
    if serial_column is None and len(test_df.columns) == 1:
        serial_column = test_df.columns[0]

    if serial_column is None:
        raise ValueError(
            "Could not identify the serial-number column in "
            f"{path}.\nAvailable columns: {list(test_df.columns)}"
        )

    serials = (
        test_df[serial_column]
        .dropna()
        .astype(str)
        .str.strip()
    )

    serials = serials[serials != ""].drop_duplicates().tolist()

    if not serials:
        raise ValueError(
            "No valid HDD serial numbers were found in the test file."
        )

    return serials


# ============================================================
# 4. Load only held-out HDD rows from the large state CSV
# ============================================================

def load_heldout_state(test_hdd_ids, state_path):
    """
    Read the large state CSV in chunks and retain only the
    original held-out HDDs.
    """

    if not state_path.exists():
        raise FileNotFoundError(
            f"Digital Twin state file not found: {state_path}"
        )

    test_hdd_set = set(test_hdd_ids)
    collected_chunks = []

    print("\nReading Digital Twin state in chunks...")
    print(f"State file: {state_path}")
    print(f"Chunk size: {CHUNK_SIZE:,}")

    reader = pd.read_csv(
        state_path,
        usecols=lambda column: column in REQUIRED_STATE_COLUMNS,
        dtype={
            "serial_number": str,
        },
        chunksize=CHUNK_SIZE,
        low_memory=True,
    )

    columns_found = None
    total_rows_read = 0

    for chunk_number, chunk in enumerate(reader, start=1):
        total_rows_read += len(chunk)

        if columns_found is None:
            columns_found = set(chunk.columns)

            missing_columns = (
                set(REQUIRED_STATE_COLUMNS) - columns_found
            )

            if missing_columns:
                raise ValueError(
                    "The state CSV is missing required columns: "
                    f"{sorted(missing_columns)}"
                )

        chunk["serial_number"] = (
            chunk["serial_number"].astype(str).str.strip()
        )

        selected = chunk[
            chunk["serial_number"].isin(test_hdd_set)
        ].copy()

        if not selected.empty:
            collected_chunks.append(selected)

        if chunk_number % 10 == 0:
            print(
                f"Chunks read: {chunk_number:,} | "
                f"Rows scanned: {total_rows_read:,} | "
                f"Held-out rows collected: "
                f"{sum(len(x) for x in collected_chunks):,}"
            )

    if not collected_chunks:
        raise ValueError(
            "No rows for the held-out HDDs were found in the "
            "Digital Twin state CSV. Check serial-number formats."
        )

    heldout_df = pd.concat(
        collected_chunks,
        ignore_index=True,
    )

    heldout_df["date"] = pd.to_datetime(
        heldout_df["date"],
        errors="coerce",
    )

    heldout_df["failure"] = pd.to_numeric(
        heldout_df["failure"],
        errors="coerce",
    )

    heldout_df[SMART1_COLUMN] = pd.to_numeric(
        heldout_df[SMART1_COLUMN],
        errors="coerce",
    )

    heldout_df[SMART187_COLUMN] = pd.to_numeric(
        heldout_df[SMART187_COLUMN],
        errors="coerce",
    )

    heldout_df = heldout_df.dropna(
        subset=["date", "serial_number"]
    )

    heldout_df = heldout_df.sort_values(
        ["serial_number", "date"]
    ).reset_index(drop=True)

    found_hdds = heldout_df["serial_number"].nunique()

    print("\nState loading complete.")
    print(f"Total state rows scanned: {total_rows_read:,}")
    print(f"Held-out HDDs requested: {len(test_hdd_ids):,}")
    print(f"Held-out HDDs found: {found_hdds:,}")
    print(f"Held-out observations collected: {len(heldout_df):,}")

    return heldout_df


# ============================================================
# 5. Calculate degradation severity and persistence
# ============================================================

def add_adaptive_degradation_columns(pre_failure):
    """
    Apply the production severity thresholds and calculate
    consecutive Moderate-or-higher observations.

    If both SMART change features are missing, the row does not
    count as degradation evidence and breaks the persistence run.
    If one feature is available, its severity can still contribute.
    """

    result = pre_failure.copy()

    smart1_values = result[SMART1_COLUMN]
    smart187_values = result[SMART187_COLUMN]

    # At least one of the two degradation features must be available.
    evidence_available = (
        smart1_values.notna() | smart187_values.notna()
    ).to_numpy()

    smart1_severity = (
        smart1_values.apply(severity_smart1).to_numpy(dtype=int)
    )

    smart187_severity = (
        smart187_values.apply(severity_smart187).to_numpy(dtype=int)
    )

    combined_severity = np.maximum(
        smart1_severity,
        smart187_severity,
    )

    moderate_or_higher = (
        evidence_available & (combined_severity >= 2)
    )

    # Count consecutive qualifying observations.
    persistence = np.zeros(
        len(result),
        dtype=int,
    )

    current_run = 0

    for index, is_qualifying in enumerate(moderate_or_higher):
        if is_qualifying:
            current_run += 1
        else:
            current_run = 0

        persistence[index] = current_run

    result["smart1_severity"] = smart1_severity
    result["smart187_severity"] = smart187_severity
    result["combined_severity"] = combined_severity
    result["degradation_evidence_available"] = evidence_available
    result["moderate_or_higher"] = moderate_or_higher
    result["persistence_observations"] = persistence

    return result


# ============================================================
# 6. Evaluate one HDD
# ============================================================

def evaluate_one_hdd(serial_number, hdd_rows):
    """
    Evaluate one HDD.

    A detection is confirmed on the observation that completes
    seven consecutive Moderate-or-higher observations.

    Only detections confirmed 0-60 calendar days before the
    recorded failure are counted as detections within the window.
    """

    hdd_rows = hdd_rows.sort_values("date").copy()

    failure_rows = hdd_rows[
        hdd_rows["failure"].eq(1)
    ]

    if failure_rows.empty:
        return {
            "serial_number": serial_number,
            "failure_date": pd.NaT,
            "observations_total": len(hdd_rows),
            "pre_failure_observations": 0,
            "first_detection_date": pd.NaT,
            "lead_time_days": np.nan,
            "detected_within_60_days": False,
            "longest_persistence": 0,
            "status": "no_recorded_failure",
        }

    failure_date = failure_rows["date"].min()

    # Exclude the failure day and any observations after it.
    pre_failure = hdd_rows[
        (hdd_rows["date"] < failure_date)
        & (hdd_rows["failure"] != 1)
    ].copy()

    if pre_failure.empty:
        return {
            "serial_number": serial_number,
            "failure_date": failure_date,
            "observations_total": len(hdd_rows),
            "pre_failure_observations": 0,
            "first_detection_date": pd.NaT,
            "lead_time_days": np.nan,
            "detected_within_60_days": False,
            "longest_persistence": 0,
            "status": "insufficient_pre_failure_observations",
        }

    pre_failure = add_adaptive_degradation_columns(pre_failure)

    longest_persistence = int(
        pre_failure["persistence_observations"].max()
    )

    # Candidate confirmation points: the first observation at which
    # persistence reaches the required seven observations.
    
    # Find the first confirmation of seven consecutive
    # Moderate-or-higher observations, regardless of lead time.
    confirmation_points = pre_failure[
        pre_failure["persistence_observations"]
        == PERSISTENCE_OBSERVATIONS
    ].copy()

    if confirmation_points.empty:
        return {
            "serial_number": serial_number,
            "failure_date": failure_date,
            "observations_total": len(hdd_rows),
            "pre_failure_observations": len(pre_failure),
            "first_detection_date": pd.NaT,
            "lead_time_days": np.nan,
            "detected_within_60_days": False,
            "longest_persistence": longest_persistence,
            "status": "not_detected",
        }

    # The first confirmation is the detection event.
    first_detection = confirmation_points.sort_values(
        "date"
    ).iloc[0]

    first_lead_time = int(
        (failure_date - first_detection["date"]).days
    )

    detected_within_window = (
        0 <= first_lead_time <= MAX_LEAD_TIME_DAYS
    )

    return {
        "serial_number": serial_number,
        "failure_date": failure_date,
        "observations_total": len(hdd_rows),
        "pre_failure_observations": len(pre_failure),
        "first_detection_date": first_detection["date"],
        "lead_time_days": first_lead_time,
        "detected_within_60_days": detected_within_window,
        "longest_persistence": longest_persistence,
        "status": (
            "detected_within_60_days"
            if detected_within_window
            else "detection_outside_60_day_window"
        ),
    }
    return {
        "serial_number": serial_number,
        "failure_date": failure_date,
        "observations_total": len(hdd_rows),
        "pre_failure_observations": len(pre_failure),
        "first_detection_date": first_detection["date"],
        "lead_time_days": int(first_detection["lead_time_days"]),
        "detected_within_60_days": True,
        "longest_persistence": longest_persistence,
        "status": "detected_within_60_days",
    }


# ============================================================
# 7. Create summary and lead-time distribution
# ============================================================

def save_outputs(results_df):
    """Save per-HDD results, summary metrics, and lead-time bands."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False,
    )

    # Use HDDs with an observed failure date as the evaluation
    # denominator. The expected original test set has 117 HDDs.
    failure_labelled = results_df[
        results_df["failure_date"].notna()
    ].copy()

    total_failure_labelled = len(failure_labelled)

    detected = failure_labelled[
        failure_labelled["detected_within_60_days"]
    ].copy()

    detected_count = len(detected)

    detection_rate = (
        detected_count / total_failure_labelled * 100
        if total_failure_labelled > 0
        else np.nan
    )

    if detected_count > 0:
        mean_lead = detected["lead_time_days"].mean()
        median_lead = detected["lead_time_days"].median()
        minimum_lead = detected["lead_time_days"].min()
        maximum_lead = detected["lead_time_days"].max()
    else:
        mean_lead = np.nan
        median_lead = np.nan
        minimum_lead = np.nan
        maximum_lead = np.nan

    summary = pd.DataFrame(
        [
            {
                "test_hdds_in_results": len(results_df),
                "hdds_with_recorded_failure": total_failure_labelled,
                "detected_within_60_days": detected_count,
                "not_detected_within_60_days": (
                    total_failure_labelled - detected_count
                ),
                "detection_rate_percent": detection_rate,
                "persistence_observations": PERSISTENCE_OBSERVATIONS,
                "max_lead_time_window_days": MAX_LEAD_TIME_DAYS,
                "mean_lead_time_days": mean_lead,
                "median_lead_time_days": median_lead,
                "minimum_lead_time_days": minimum_lead,
                "maximum_lead_time_days": maximum_lead,
            }
        ]
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    # Distribution is calculated only among detections within 60 days.
    bins = [0, 3, 7, 14, 30, 60]
    labels = [
        "0-3 days",
        "4-7 days",
        "8-14 days",
        "15-30 days",
        "31-60 days",
    ]

    if detected_count > 0:
        detected["lead_time_band"] = pd.cut(
            detected["lead_time_days"],
            bins=bins,
            labels=labels,
            include_lowest=True,
            right=True,
        )

        distribution = (
            detected["lead_time_band"]
            .value_counts(sort=False)
            .reindex(labels, fill_value=0)
            .rename_axis("lead_time_band")
            .reset_index(name="hdd_count")
        )

        distribution["percentage_of_detections"] = (
            distribution["hdd_count"] / detected_count * 100
        )
    else:
        distribution = pd.DataFrame(
            {
                "lead_time_band": labels,
                "hdd_count": [0] * len(labels),
                "percentage_of_detections": [0.0] * len(labels),
            }
        )

    distribution.to_csv(
        DISTRIBUTION_PATH,
        index=False,
    )

    return summary, distribution


# ============================================================
# 8. Main execution
# ============================================================

def main():
    print("=" * 72)
    print("HELD-OUT ADAPTIVE DIGITAL TWIN LEAD-TIME VALIDATION")
    print("=" * 72)

    print(f"\nProject root: {PROJECT_ROOT}")
    print(f"Test HDD file: {TEST_HDDS_PATH}")
    print(f"State file: {STATE_PATH}")

    test_hdd_ids = load_test_hdd_ids(TEST_HDDS_PATH)

    print(f"\nOriginal held-out HDD IDs: {len(test_hdd_ids)}")

    heldout_state = load_heldout_state(
        test_hdd_ids,
        STATE_PATH,
    )

    available_hdds = set(
        heldout_state["serial_number"].unique()
    )

    missing_hdds = sorted(
        set(test_hdd_ids) - available_hdds
    )

    if missing_hdds:
        print(
            f"\nWarning: {len(missing_hdds)} held-out HDDs "
            "were not found in the state CSV."
        )

    results = []

    grouped = {
        serial: group
        for serial, group in heldout_state.groupby(
            "serial_number",
            sort=False,
        )
    }

    print("\nEvaluating held-out HDDs...")

    for index, serial_number in enumerate(test_hdd_ids, start=1):
        hdd_rows = grouped.get(serial_number)

        if hdd_rows is None:
            results.append(
                {
                    "serial_number": serial_number,
                    "failure_date": pd.NaT,
                    "observations_total": 0,
                    "pre_failure_observations": 0,
                    "first_detection_date": pd.NaT,
                    "lead_time_days": np.nan,
                    "detected_within_60_days": False,
                    "longest_persistence": 0,
                    "status": "hdd_not_found_in_state",
                }
            )
        else:
            results.append(
                evaluate_one_hdd(
                    serial_number,
                    hdd_rows,
                )
            )

        if index % 20 == 0 or index == len(test_hdd_ids):
            print(
                f"Evaluated {index}/{len(test_hdd_ids)} HDDs"
            )

    results_df = pd.DataFrame(results)

    # Save dates in a consistent format.
    for column in ["failure_date", "first_detection_date"]:
        results_df[column] = pd.to_datetime(
            results_df[column],
            errors="coerce",
        ).dt.strftime("%Y-%m-%d")

    summary, distribution = save_outputs(results_df)

    print("\n" + "=" * 72)
    print("VALIDATION SUMMARY")
    print("=" * 72)

    print(summary.to_string(index=False))

    print("\nLead-time distribution among detections:")
    print(distribution.to_string(index=False))

    print("\nPer-HDD results:")
    print(RESULTS_PATH)

    print("\nSummary:")
    print(SUMMARY_PATH)

    print("\nDistribution:")
    print(DISTRIBUTION_PATH)

    print("\nValidation completed.")


if __name__ == "__main__":
    main()
