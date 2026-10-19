"""
Tune the adaptive degradation detector without touching the test set.

Detector configurations are selected on the TRAIN + VALIDATION HDD
split (results/rul_sequences/train_hdds.csv + validation_hdds.csv),
then the winning configuration is evaluated exactly ONCE on the
untouched held-out test set (test_hdds.csv).

Healthy (never-failed) control HDDs provide the false-alarm rate.
Two DISJOINT control samples are drawn - one for tuning, one for the
final test evaluation - so neither the configuration nor its reported
false-alarm rate is selected on the test data.

Configurations vary:
  - signal set: 7-day SMART changes only, or 7d + 14d changes
  - threshold table: original (production), or relaxed (half as strict)
  - severity required: Mild+ (>=1) or Moderate+ (>=2)
  - persistence: 3, 4, 5, 6 or 7 consecutive observations

NOTE: the change features are observation diffs, not calendar-day
diffs (see degradation_detector.py).

Outputs (results/digital_twin_rul/):
  - threshold_sweep_results.csv                per-config tuning metrics
  - threshold_sweep_best.csv                   best config on the tuning set
  - threshold_sweep_tuning_healthy_controls.csv tuning control HDDs
  - threshold_sweep_test_healthy_controls.csv   test control HDDs
  - threshold_sweep_final_test.csv             chosen config evaluated once
                                               on the held-out test set

Run from the component root:
  python .\\scripts\\tune_detection_thresholds.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

import degradation_detector as detector
from degradation_detector import (
    SMART1_7D,
    SMART187_7D,
    SMART1_14D,
    SMART187_14D,
)


# ============================================================
# 1. Paths and configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPLIT_DIR = PROJECT_ROOT / "results" / "rul_sequences"

# Tuning pool: failed HDDs the detector may be selected on.
TUNING_HDDS_PATHS = [
    SPLIT_DIR / "train_hdds.csv",
    SPLIT_DIR / "validation_hdds.csv",
]

# Held-out test HDDs: never used for selection, evaluated once.
TEST_HDDS_PATH = SPLIT_DIR / "test_hdds.csv"

STATE_PATH = PROJECT_ROOT / "results" / "digital_twin_state_v5.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "digital_twin_rul"

SWEEP_RESULTS_PATH = OUTPUT_DIR / "threshold_sweep_results.csv"

BEST_PATH = OUTPUT_DIR / "threshold_sweep_best.csv"

FINAL_TEST_PATH = OUTPUT_DIR / "threshold_sweep_final_test.csv"

TUNING_CONTROLS_PATH = (
    OUTPUT_DIR / "threshold_sweep_tuning_healthy_controls.csv"
)

TEST_CONTROLS_PATH = (
    OUTPUT_DIR / "threshold_sweep_test_healthy_controls.csv"
)

CHUNK_SIZE = 250_000

HEALTHY_CONTROL_SAMPLE = 400

RANDOM_SEED = 42

MAX_LEAD_TIME_DAYS = 60

MAX_HEALTHY_FALSE_ALARM_RATE = 20.0

REQUIRED_STATE_COLUMNS = [
    "date",
    "serial_number",
    "failure",
    SMART1_7D,
    SMART187_7D,
    SMART1_14D,
    SMART187_14D,
]


# ============================================================
# 2. Threshold tables
#    The original (production) tables come from
#    degradation_detector.py so they cannot diverge.
#    Relaxed tables are exactly half as strict.
# ============================================================

def make_severity_function(thresholds):
    """
    Build a severity function from an ordered list of
    (bound, severity) pairs evaluated from strict to mild.
    Missing values always return severity 0.
    """

    def severity(value):

        return detector.severity_from_thresholds(
            value,
            thresholds,
            missing=0,
        )

    return severity


THRESHOLD_TABLES = {
    "original": {
        SMART1_7D: list(detector.SMART1_THRESHOLDS),
        SMART187_7D: list(detector.SMART187_THRESHOLDS),
    },
    "relaxed": {
        SMART1_7D: [(-7.5, 4), (-4, 3), (-1.5, 2), (-0.5, 1)],
        SMART187_7D: [(-11.5, 4), (-4.5, 3), (-1.5, 2), (-0.5, 1)],
    },
}

# 14-day tables use double the 7-day bounds so that both windows
# represent the same degradation rate per day.
THRESHOLD_TABLES_14D = {
    key: {
        SMART1_14D: [(bound * 2, level) for bound, level in table[SMART1_7D]],
        SMART187_14D: [
            (bound * 2, level) for bound, level in table[SMART187_7D]
        ],
    }
    for key, table in THRESHOLD_TABLES.items()
}


# ============================================================
# 3. Detector configurations
# ============================================================

CONFIGS = []

for table_name in ["original", "relaxed"]:
    for use_14d in [False, True]:
        for min_severity in [1, 2]:
            for persistence in [3, 4, 5, 6, 7]:
                CONFIGS.append({
                    "name": (
                        f"{table_name}"
                        f"{'_7d_14d' if use_14d else '_7d'}"
                        f"_sev{min_severity}"
                        f"_p{persistence}"
                    ),
                    "threshold_table": table_name,
                    "use_14d": use_14d,
                    "min_severity": min_severity,
                    "persistence": persistence,
                    # Flag the configuration currently deployed in
                    # degradation_detector.py.
                    "matches_production": (
                        table_name == "original"
                        and not use_14d
                        and min_severity == detector.MIN_SEVERITY
                        and persistence == detector.PERSISTENCE
                    ),
                })


# ============================================================
# 4. Load HDD identifiers
# ============================================================

def load_hdd_ids(path, label):
    """Load serial numbers from an HDD split file."""

    if not path.exists():
        raise FileNotFoundError(f"{label} HDD file not found: {path}")

    test_df = pd.read_csv(path, dtype=str)

    serial_column = next(
        (
            column
            for column in test_df.columns
            if column.strip().lower() in {"serial_number", "serial"}
        ),
        test_df.columns[0],
    )

    serials = (
        test_df[serial_column]
        .dropna()
        .astype(str)
        .str.strip()
    )

    return sorted(set(serials[serials != ""]))


def load_tuning_hdd_ids():
    """Failed HDDs the detector is allowed to be selected on."""

    tuning_ids = set()

    for path in TUNING_HDDS_PATHS:
        tuning_ids.update(load_hdd_ids(path, f"{path.stem}"))

    return sorted(tuning_ids)


# ============================================================
# 5. Read the twin state rows for a set of HDDs
# ============================================================


def load_state(wanted_ids, state_path):
    """
    Read the twin state in chunks, keep requested HDDs only.

    A single pass over the large state CSV; callers split the
    returned rows into tuning and test subsets.
    """

    heldout_set = set(wanted_ids)

    collected = []

    print("\nReading Digital Twin state in chunks...")
    print(f"State file: {state_path}")

    reader = pd.read_csv(
        state_path,
        usecols=lambda column: column in REQUIRED_STATE_COLUMNS,
        dtype={"serial_number": str},
        chunksize=CHUNK_SIZE,
        low_memory=True,
    )

    total_rows_read = 0

    for chunk_number, chunk in enumerate(reader, start=1):
        total_rows_read += len(chunk)

        chunk["serial_number"] = (
            chunk["serial_number"].astype(str).str.strip()
        )

        selected = chunk[
            chunk["serial_number"].isin(heldout_set)
        ].copy()

        if not selected.empty:
            collected.append(selected)

        if chunk_number % 10 == 0:
            collected_rows = sum(len(frame) for frame in collected)
            print(
                f"Chunks read: {chunk_number:,} | "
                f"Rows scanned: {total_rows_read:,} | "
                f"Rows collected: {collected_rows:,}"
            )

    if not collected:
        raise ValueError(
            "No rows were found for the requested HDDs. "
            f"Requested: {len(heldout_set):,} HDDs."
        )

    heldout = pd.concat(collected, ignore_index=True)

    heldout["date"] = pd.to_datetime(
        heldout["date"],
        errors="coerce",
    )

    heldout["failure"] = pd.to_numeric(
        heldout["failure"],
        errors="coerce",
    )

    for column in [SMART1_7D, SMART187_7D, SMART1_14D, SMART187_14D]:
        heldout[column] = pd.to_numeric(
            heldout[column],
            errors="coerce",
        )

    heldout = heldout.dropna(subset=["date", "serial_number"])

    heldout = heldout.sort_values(
        ["serial_number", "date"]
    ).reset_index(drop=True)

    print("\nState loading complete.")
    print(f"Total state rows scanned: {total_rows_read:,}")
    print(f"Observations collected: {len(heldout):,}")
    print(f"HDDs found: {heldout['serial_number'].nunique():,}")

    return heldout


# ============================================================
# 5b. Healthy control HDDs
# ============================================================

def select_healthy_controls(
    state_path,
    sample_size,
    seed=RANDOM_SEED,
    exclude_ids=(),
    output_path=None,
):
    """
    Light pass over serial_number + failure to find HDDs that
    never recorded a failure, then sample a random subset.

    These serve as the false-alarm (specificity) control group,
    since the HDD split files contain only failed drives.

    `exclude_ids` keeps failed HDDs and any controls drawn for the
    other evaluation set out of the sample, so the tuning and test
    control groups stay disjoint. The selected ids are written to
    `output_path` so any run can be audited afterwards.
    """

    excluded = {
        str(serial).strip()
        for serial in exclude_ids
    }

    print("\nSelecting healthy control HDDs (light pass)...")

    reader = pd.read_csv(
        state_path,
        usecols=["serial_number", "failure"],
        dtype={"serial_number": str},
        chunksize=CHUNK_SIZE,
        low_memory=True,
    )

    ever_seen = set()
    ever_failed = set()

    for chunk in reader:
        serials = (
            chunk["serial_number"].astype(str).str.strip()
        )

        ever_seen.update(serials.unique())

        failed_mask = pd.to_numeric(
            chunk["failure"], errors="coerce"
        ).fillna(0).eq(1)

        ever_failed.update(serials[failed_mask].unique())

    healthy = sorted(ever_seen - ever_failed - excluded)

    print(f"Drives observed: {len(ever_seen):,}")
    print(f"Drives that failed: {len(ever_failed):,}")
    print(f"Healthy (never-failed) drives: {len(healthy):,}")
    print(f"Excluded ids (failed/other controls): {len(excluded):,}")

    if not healthy:
        raise ValueError(
            "No never-failed drives were found in the state CSV."
        )

    rng = np.random.default_rng(seed)

    chosen = sorted(
        rng.choice(
            healthy,
            size=min(sample_size, len(healthy)),
            replace=False,
        ).tolist()
    )

    print(f"Sampled healthy controls: {len(chosen):,}")

    if output_path is not None:
        pd.DataFrame({"serial_number": chosen}).to_csv(
            output_path,
            index=False,
        )
        print(f"Control ids written to: {output_path}")

    return chosen


# ============================================================
# 6. Per-configuration evaluation
# ============================================================

def add_severity_columns(rows, config):
    """Add combined severity and persistence columns for one config."""

    table_name = config["threshold_table"]

    severity_functions = {
        column: make_severity_function(bounds)
        for column, bounds in THRESHOLD_TABLES[table_name].items()
    }

    signal_columns = [SMART1_7D, SMART187_7D]

    if config["use_14d"]:
        severity_functions.update({
            column: make_severity_function(bounds)
            for column, bounds in THRESHOLD_TABLES_14D[table_name].items()
        })
        signal_columns += [SMART1_14D, SMART187_14D]

    # A row counts as evidence when at least one signal is present.
    evidence_available = rows[signal_columns].notna().any(axis=1).to_numpy()

    severities = np.column_stack([
        rows[column].apply(severity_functions[column]).to_numpy(dtype=int)
        for column in signal_columns
    ])

    combined_severity = severities.max(axis=1)

    qualifying = (
        evidence_available
        & (combined_severity >= config["min_severity"])
    )

    persistence = np.zeros(len(rows), dtype=int)

    current_run = 0

    for index, is_qualifying in enumerate(qualifying):
        current_run = current_run + 1 if is_qualifying else 0
        persistence[index] = current_run

    return persistence


def evaluate_config(heldout_state, config):
    """Evaluate one detector config over failed and healthy HDDs."""

    detection_rates = []
    lead_times = []
    healthy_alarm_flags = []

    for serial_number, hdd_rows in heldout_state.groupby(
        "serial_number",
        sort=False,
    ):
        hdd_rows = hdd_rows.sort_values("date")

        failure_date = hdd_rows.loc[
            hdd_rows["failure"].eq(1),
            "date",
        ].min()

        if pd.isna(failure_date):
            # Healthy drive: any completed persistence run is a
            # false alarm.
            persistence = add_severity_columns(hdd_rows, config)

            healthy_alarm_flags.append(
                int(persistence.max() >= config["persistence"])
            )

            continue

        pre_failure = hdd_rows[
            (hdd_rows["date"] < failure_date)
            & (hdd_rows["failure"] != 1)
        ].copy()

        if pre_failure.empty:
            detection_rates.append(False)
            continue

        persistence = add_severity_columns(pre_failure, config)

        confirmation_points = pre_failure[
            persistence == config["persistence"]
        ]

        if confirmation_points.empty:
            detection_rates.append(False)
            continue

        first_detection_date = confirmation_points["date"].min()

        lead_time = int(
            (failure_date - first_detection_date).days
        )

        detected_within_window = (
            0 <= lead_time <= MAX_LEAD_TIME_DAYS
        )

        detection_rates.append(detected_within_window)

        if detected_within_window:
            lead_times.append(lead_time)

    failed_count = len(detection_rates)
    healthy_count = len(healthy_alarm_flags)

    detection_rate = (
        np.mean(detection_rates) * 100 if failed_count else np.nan
    )

    healthy_false_alarm_rate = (
        np.mean(healthy_alarm_flags) * 100 if healthy_count else np.nan
    )

    return {
        "config": config["name"],
        "threshold_table": config["threshold_table"],
        "use_14d": config["use_14d"],
        "min_severity": config["min_severity"],
        "persistence": config["persistence"],
        "matches_production": bool(
            config.get("matches_production", False)
        ),
        "failed_hdds": failed_count,
        "detected_within_60_days": int(np.sum(detection_rates)),
        "detection_rate_percent": detection_rate,
        "median_lead_time_days": (
            float(np.median(lead_times)) if lead_times else np.nan
        ),
        "mean_lead_time_days": (
            float(np.mean(lead_times)) if lead_times else np.nan
        ),
        "healthy_hdds": healthy_count,
        "healthy_false_alarms": int(np.sum(healthy_alarm_flags)),
        "healthy_false_alarm_rate_percent": healthy_false_alarm_rate,
    }


# ============================================================
# 7. Main execution
# ============================================================

def main():
    print("=" * 72)
    print("DETECTOR THRESHOLD SWEEP (tune on train+validation, test once)")
    print("=" * 72)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------
    # Split the failed HDDs: tuning pool vs untouched test
    # ----------------------------------------------------

    tuning_failed_ids = load_tuning_hdd_ids()

    test_failed_ids = load_hdd_ids(TEST_HDDS_PATH, "test")

    overlap = set(tuning_failed_ids) & set(test_failed_ids)

    if overlap:
        raise ValueError(
            f"Tuning and test splits share {len(overlap)} HDDs."
        )

    failed_ids = set(tuning_failed_ids) | set(test_failed_ids)

    print(f"\nTuning (failed) HDDs: {len(tuning_failed_ids)}")
    print(f"Held-out test (failed) HDDs: {len(test_failed_ids)}")

    # ----------------------------------------------------
    # Two disjoint healthy control samples
    # ----------------------------------------------------

    tuning_control_ids = select_healthy_controls(
        STATE_PATH,
        HEALTHY_CONTROL_SAMPLE,
        seed=RANDOM_SEED,
        exclude_ids=failed_ids,
        output_path=TUNING_CONTROLS_PATH,
    )

    test_control_ids = select_healthy_controls(
        STATE_PATH,
        HEALTHY_CONTROL_SAMPLE,
        seed=RANDOM_SEED + 1,
        exclude_ids=failed_ids | set(tuning_control_ids),
        output_path=TEST_CONTROLS_PATH,
    )

    assert not set(tuning_control_ids) & set(test_control_ids)
    assert not set(tuning_control_ids) & failed_ids
    assert not set(test_control_ids) & failed_ids

    # ----------------------------------------------------
    # One pass over the state CSV for both subsets
    # ----------------------------------------------------

    tuning_ids = set(tuning_failed_ids) | set(tuning_control_ids)
    test_ids = set(test_failed_ids) | set(test_control_ids)

    state = load_state(sorted(tuning_ids | test_ids), STATE_PATH)

    tuning_state = state[state["serial_number"].isin(tuning_ids)]
    test_state = state[state["serial_number"].isin(test_ids)]

    print(f"\nTuning observations: {len(tuning_state):,}")
    print(f"Test observations: {len(test_state):,}")

    # ----------------------------------------------------
    # Sweep every configuration on the tuning pool only
    # ----------------------------------------------------

    print(f"\nSweeping {len(CONFIGS)} configurations on the tuning set...\n")

    rows = []

    for index, config in enumerate(CONFIGS, start=1):
        result = evaluate_config(tuning_state, config)
        result["evaluation_set"] = "tuning (train+validation)"
        rows.append(result)

        print(
            f"[{index:02d}/{len(CONFIGS)}] {config['name']:<34} "
            f"detection={result['detection_rate_percent']:5.1f}%  "
            f"healthy_fp={result['healthy_false_alarm_rate_percent']:5.1f}%  "
            f"median_lead={result['median_lead_time_days']}"
        )

    sweep = pd.DataFrame(rows).sort_values(
        ["detection_rate_percent", "healthy_false_alarm_rate_percent"],
        ascending=[False, True],
    ).reset_index(drop=True)

    sweep.to_csv(SWEEP_RESULTS_PATH, index=False)

    eligible = sweep[
        sweep["healthy_false_alarm_rate_percent"]
        <= MAX_HEALTHY_FALSE_ALARM_RATE
    ]

    best = (
        eligible.iloc[0]
        if not eligible.empty
        else sweep.iloc[0]
    )

    pd.DataFrame([best]).to_csv(BEST_PATH, index=False)

    print("\n" + "=" * 72)
    print("TOP 10 CONFIGURATIONS BY DETECTION RATE (TUNING SET)")
    print("=" * 72)

    print(
        sweep.head(10)[
            [
                "config",
                "detection_rate_percent",
                "median_lead_time_days",
                "healthy_false_alarm_rate_percent",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 72)
    print(
        f"SELECTED CONFIG (healthy false-alarm rate "
        f"<= {MAX_HEALTHY_FALSE_ALARM_RATE}%)"
    )
    print("=" * 72)

    print(
        pd.DataFrame([best])[
            [
                "config",
                "detection_rate_percent",
                "median_lead_time_days",
                "healthy_false_alarm_rate_percent",
            ]
        ].to_string(index=False)
    )

    # ----------------------------------------------------
    # Report the untouched held-out test set results.
    # The configuration was chosen on the tuning set only;
    # this block is reporting, not selection.
    # ----------------------------------------------------

    selected_config = {
        "name": best["config"],
        "threshold_table": best["threshold_table"],
        "use_14d": bool(best["use_14d"]),
        "min_severity": int(best["min_severity"]),
        "persistence": int(best["persistence"]),
        "matches_production": bool(best["matches_production"]),
    }

    deployed_config = {
        "name": "deployed_in_degradation_detector",
        "threshold_table": "original",
        "use_14d": False,
        "min_severity": detector.MIN_SEVERITY,
        "persistence": detector.PERSISTENCE,
        "matches_production": True,
    }

    candidates = [("selected_on_tuning", selected_config)]

    if not selected_config["matches_production"]:
        candidates.append(("currently_deployed", deployed_config))

    final_rows = []

    for role, config in candidates:
        result = evaluate_config(test_state, config)
        result["role"] = role
        result["evaluation_set"] = "held-out test (reported once)"
        final_rows.append(result)

    pd.DataFrame(final_rows).to_csv(FINAL_TEST_PATH, index=False)

    print("\n" + "=" * 72)
    print("HELD-OUT TEST RESULTS (reporting only - selection was made above)")
    print("=" * 72)

    print(
        pd.DataFrame(final_rows)[
            [
                "role",
                "config",
                "failed_hdds",
                "detected_within_60_days",
                "detection_rate_percent",
                "median_lead_time_days",
                "mean_lead_time_days",
                "healthy_hdds",
                "healthy_false_alarms",
                "healthy_false_alarm_rate_percent",
            ]
        ].to_string(index=False)
    )

    if not selected_config["matches_production"]:
        print(
            "\nNOTE: the tuning set selected a different configuration\n"
            "than the one currently deployed in\n"
            "degradation_detector.py. Compare the two rows above and,\n"
            "if adopting the selection, update MIN_SEVERITY /\n"
            "PERSISTENCE there, then re-run\n"
            "validate_heldout_adaptive_lead_time.py, the state-manager\n"
            "tests and the validation report before shipping."
        )

    print("\nFull tuning sweep:")
    print(SWEEP_RESULTS_PATH)

    print("\nSelected configuration (tuning set):")
    print(BEST_PATH)

    print("\nHeld-out test result:")
    print(FINAL_TEST_PATH)


if __name__ == "__main__":
    main()
