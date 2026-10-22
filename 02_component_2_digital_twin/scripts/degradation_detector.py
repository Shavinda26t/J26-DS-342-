"""
Single source of truth for the tuned production degradation detector.

Every detector in this component imports its constants from here so the
configuration cannot silently diverge between scripts:

  - adaptive_digital_twin.py            (batch detector + RUL model)
  - digital_twin_state_manager.py       (streaming detector)
  - validate_heldout_adaptive_lead_time.py (held-out validation)
  - tune_detection_thresholds.py        (threshold sweep)

Detector configuration
----------------------
Selected by scripts/tune_detection_thresholds.py on the TRAIN+VALIDATION
HDD split only, then evaluated once on the untouched held-out test set
(see threshold_sweep_results.csv and threshold_sweep_final_test.csv).

  MIN_SEVERITY = 1   mild-or-higher severity counts as degradation
  PERSISTENCE  = 7   consecutive qualifying observations to confirm

Change-feature semantics
------------------------
The "7d" / "14d" change features are 7- and 14-OBSERVATION diffs, not
calendar-day diffs (build_temporal_features.py uses groupby().diff(n)).
Sampling gaps are common in this dataset (see
results/hdd_observation_gap_analysis.csv), so a persistence run of 7
observations does NOT imply 7 calendar days. The thresholds below are
expressed in those observation-diff units.
"""

import pandas as pd


# ============================================================
# Tuned detector configuration
# ============================================================

MIN_SEVERITY = 1

PERSISTENCE = 7

# Per-attribute change thresholds, from strict to mild.
# Each entry maps a lower bound (change <= bound) to a severity
# level: 0 = none, 1 = mild, 2 = moderate, 3 = strong, 4 = severe.
SMART1_THRESHOLDS = [
    (-15, 4),
    (-8, 3),
    (-3, 2),
    (-1, 1),
]

SMART187_THRESHOLDS = [
    (-23, 4),
    (-9, 3),
    (-3, 2),
    (-1, 1),
]

# Signal column names used by the detectors.
SMART1_7D = "smart_1_normalized_change_7d"
SMART187_7D = "smart_187_normalized_change_7d"
SMART1_14D = "smart_1_normalized_change_14d"
SMART187_14D = "smart_187_normalized_change_14d"

# Adaptive state labels, shared by every detector and by the
# analysis scripts that group results by adaptive_state.
ADAPTIVE_STATE_NAMES = {
    0: "Normal / Monitor",
    1: "Early Warning",
    2: "Elevated Degradation",
    3: "Strong Persistent Degradation",
}


# ============================================================
# Severity
# ============================================================

def severity_from_thresholds(value, thresholds, missing=0):
    """
    Return the severity level (0-4) for a change value, given an
    ordered list of (bound, level) pairs evaluated strict to mild.

    `missing` is the value returned for absent evidence. The batch
    detector passes 0 (absent evidence never qualifies); the
    streaming state manager passes np.nan so absence stays
    distinguishable from an observed healthy value.
    """

    if pd.isna(value):
        return missing

    for bound, level in thresholds:

        if value <= bound:
            return level

    return 0


def smart1_severity(value, missing=0):
    """Severity level (0-4) for the SMART 1 change value."""

    return severity_from_thresholds(
        value,
        SMART1_THRESHOLDS,
        missing=missing,
    )


def smart187_severity(value, missing=0):
    """Severity level (0-4) for the SMART 187 change value."""

    return severity_from_thresholds(
        value,
        SMART187_THRESHOLDS,
        missing=missing,
    )


def combined_severity(smart1_value, smart187_value, missing=0):
    """
    Combined severity of both degradation signals (their maximum).

    A signal without evidence contributes nothing; when neither
    signal has evidence the result is `missing`.
    """

    severities = [
        value
        for value in (
            smart1_severity(smart1_value, missing=missing),
            smart187_severity(smart187_value, missing=missing),
        )
        if not pd.isna(value)
    ]

    if not severities:
        return missing

    return max(severities)


def is_qualifying(severity, min_severity=None):
    """
    True when a severity counts towards the persistence streak.

    Missing evidence (np.nan) never qualifies: it neither extends
    nor is mistaken for observed degradation. Callers decide
    whether a non-qualifying row breaks the run (batch detector)
    or is flagged as missing evidence (streaming manager).
    """

    if min_severity is None:
        min_severity = MIN_SEVERITY

    if pd.isna(severity):
        return False

    return bool(severity >= min_severity)


def scaled_thresholds(thresholds, factor):
    """
    Scale threshold bounds by a factor (< 1 relaxes them).
    Returns a new list; the shared tables stay untouched.
    """

    if factor is None:
        factor = 1.0

    return [
        (bound * factor, level)
        for bound, level in thresholds
    ]


# ============================================================
# Escalation
# ============================================================

def escalation_level(streak, persistence=None):
    """
    Map a consecutive qualifying-observation streak to the
    adaptive state level 0-3.

        3 = Strong Persistent Degradation  (streak >= p + 2)
        2 = Elevated Degradation           (streak >= p)
        1 = Early Warning                  (streak >= max(1, p-2))
        0 = Normal / Monitor
    """

    if persistence is None:
        persistence = PERSISTENCE

    if streak >= persistence + 2:
        return 3

    if streak >= persistence:
        return 2

    if streak >= max(1, persistence - 2):
        return 1

    return 0


def adaptive_state_name(level):
    """Readable label for an adaptive state level."""

    return ADAPTIVE_STATE_NAMES.get(level, "Unknown")
