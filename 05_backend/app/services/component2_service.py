"""Component 2 (Adaptive Digital Twin) service.

Reads the validated CSV artifacts produced by
``02_component_2_digital_twin`` so the API and the React dashboard report
exactly the same numbers as the component's own held-out validation
(single source of truth: no metrics are re-computed here).

Files consumed (all in ``results/digital_twin_rul``):
- ``heldout_adaptive_lead_time_summary.csv``      -> headline metrics
- ``heldout_adaptive_lead_time_results.csv``      -> per-HDD detections
- ``heldout_adaptive_lead_time_distribution.csv`` -> lead-time bands
- ``threshold_sweep_results.csv``                 -> tuning trade-off curve
- ``threshold_sweep_final_test.csv``              -> selected vs deployed
- ``heldout_rul_uncertainty_by_state.csv``        -> RUL error by state
- ``heldout_rul_trajectory_metrics.csv``          -> RUL error by checkpoint
- ``heldout_test_adaptive_digital_twin_results.csv`` -> per-HDD twin detail
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

# 05_backend/app/services/<this file> -> repo root is parents[3]
REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "02_component_2_digital_twin" / "results" / "digital_twin_rul"

SUMMARY_CSV = "heldout_adaptive_lead_time_summary.csv"
DETECTIONS_CSV = "heldout_adaptive_lead_time_results.csv"
DISTRIBUTION_CSV = "heldout_adaptive_lead_time_distribution.csv"
SWEEP_CSV = "threshold_sweep_results.csv"
FINAL_TEST_CSV = "threshold_sweep_final_test.csv"
RUL_STATE_CSV = "heldout_rul_uncertainty_by_state.csv"
RUL_TRAJECTORY_CSV = "heldout_rul_trajectory_metrics.csv"
TWIN_DETAIL_CSV = "heldout_test_adaptive_digital_twin_results.csv"


def _read_rows(filename: str) -> List[Dict[str, str]]:
    path = RESULTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Component 2 result file not found: {path}. "
            "Run the component's validation scripts first "
            "(validate_heldout_adaptive_lead_time.py / tune_detection_thresholds.py)."
        )
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(value: Optional[str], default: Optional[float] = None) -> Optional[float]:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _int(value: Optional[str], default: int = 0) -> int:
    number = _float(value)
    return int(number) if number is not None else default


def _bool(value: Optional[str]) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes")


@lru_cache(maxsize=1)
def get_overview() -> Dict:
    summary = _read_rows(SUMMARY_CSV)[0]

    deployed = None
    for row in _read_rows(FINAL_TEST_CSV):
        if row.get("role") == "currently_deployed":
            deployed = row
            break
    if deployed is None:  # fall back to the production-flagged sweep row
        for row in _read_rows(SWEEP_CSV):
            if _bool(row.get("matches_production")):
                deployed = row
                break
    if deployed is None:
        raise FileNotFoundError(
            "No deployed-configuration row found in threshold_sweep_final_test.csv "
            "or threshold_sweep_results.csv."
        )

    return {
        "summary": {
            "test_hdds": _int(summary.get("test_hdds_in_results")),
            "detected": _int(summary.get("detected_within_60_days")),
            "not_detected": _int(summary.get("not_detected_within_60_days")),
            "detection_rate_percent": _float(summary.get("detection_rate_percent"), 0.0),
            "persistence_observations": _int(summary.get("persistence_observations")),
            "max_lead_time_window_days": _int(summary.get("max_lead_time_window_days")),
            "mean_lead_time_days": _float(summary.get("mean_lead_time_days"), 0.0),
            "median_lead_time_days": _float(summary.get("median_lead_time_days"), 0.0),
            "minimum_lead_time_days": _float(summary.get("minimum_lead_time_days"), 0.0),
            "maximum_lead_time_days": _float(summary.get("maximum_lead_time_days"), 0.0),
            "median_persistence_span_days": _float(
                summary.get("median_persistence_span_days"), 0.0
            ),
            "maximum_persistence_span_days": _float(
                summary.get("maximum_persistence_span_days"), 0.0
            ),
        },
        "false_alarms": {
            "healthy_hdds": _int(deployed.get("healthy_hdds")),
            "healthy_false_alarms": _int(deployed.get("healthy_false_alarms")),
            "healthy_false_alarm_rate_percent": _float(
                deployed.get("healthy_false_alarm_rate_percent"), 0.0
            ),
        },
        "config": {
            "config": deployed.get("config", ""),
            "threshold_table": deployed.get("threshold_table", ""),
            "min_severity": _int(deployed.get("min_severity"), 1),
            "persistence": _int(deployed.get("persistence"), 7),
            "matches_production": _bool(deployed.get("matches_production")),
        },
    }


@lru_cache(maxsize=1)
def get_detections() -> List[Dict]:
    records = []
    for row in _read_rows(DETECTIONS_CSV):
        records.append(
            {
                "serial_number": row.get("serial_number", ""),
                "failure_date": row.get("failure_date", ""),
                "pre_failure_observations": _int(row.get("pre_failure_observations")),
                "first_detection_date": row.get("first_detection_date") or None,
                "lead_time_days": _float(row.get("lead_time_days")),
                "persistence_span_days": _float(row.get("persistence_span_days")),
                "longest_persistence": _int(row.get("longest_persistence")),
                "detected_within_60_days": _bool(row.get("detected_within_60_days")),
                "status": row.get("status", ""),
            }
        )
    # detected HDDs first, longest lead time on top
    records.sort(
        key=lambda r: (r["detected_within_60_days"], r["lead_time_days"] or -1),
        reverse=True,
    )
    return records


@lru_cache(maxsize=1)
def get_lead_time_distribution() -> List[Dict]:
    return [
        {
            "lead_time_band": row.get("lead_time_band", ""),
            "hdd_count": _int(row.get("hdd_count")),
            "percentage_of_detections": _float(
                row.get("percentage_of_detections"), 0.0
            ),
        }
        for row in _read_rows(DISTRIBUTION_CSV)
    ]


@lru_cache(maxsize=1)
def get_sweep() -> List[Dict]:
    return [
        {
            "config": row.get("config", ""),
            "threshold_table": row.get("threshold_table", ""),
            "use_14d": _bool(row.get("use_14d")),
            "min_severity": _int(row.get("min_severity"), 1),
            "persistence": _int(row.get("persistence"), 7),
            "matches_production": _bool(row.get("matches_production")),
            "detection_rate_percent": _float(row.get("detection_rate_percent"), 0.0),
            "healthy_false_alarm_rate_percent": _float(
                row.get("healthy_false_alarm_rate_percent"), 0.0
            ),
            "median_lead_time_days": _float(row.get("median_lead_time_days"), 0.0),
            "evaluation_set": row.get("evaluation_set", ""),
        }
        for row in _read_rows(SWEEP_CSV)
    ]


@lru_cache(maxsize=1)
def get_config_comparison() -> List[Dict]:
    return [
        {
            "role": row.get("role", ""),
            "config": row.get("config", ""),
            "threshold_table": row.get("threshold_table", ""),
            "use_14d": _bool(row.get("use_14d")),
            "min_severity": _int(row.get("min_severity"), 1),
            "persistence": _int(row.get("persistence"), 7),
            "matches_production": _bool(row.get("matches_production")),
            "failed_hdds": _int(row.get("failed_hdds")),
            "detected_within_60_days": _int(row.get("detected_within_60_days")),
            "detection_rate_percent": _float(row.get("detection_rate_percent"), 0.0),
            "median_lead_time_days": _float(row.get("median_lead_time_days"), 0.0),
            "healthy_hdds": _int(row.get("healthy_hdds")),
            "healthy_false_alarms": _int(row.get("healthy_false_alarms")),
            "healthy_false_alarm_rate_percent": _float(
                row.get("healthy_false_alarm_rate_percent"), 0.0
            ),
            "evaluation_set": row.get("evaluation_set", ""),
        }
        for row in _read_rows(FINAL_TEST_CSV)
    ]


@lru_cache(maxsize=1)
def get_rul_metrics() -> Dict:
    by_state = [
        {
            "adaptive_state": row.get("adaptive_state", ""),
            "n_predictions": _int(row.get("n_predictions")),
            "n_hdds": _int(row.get("n_hdds")),
            "mae_days": _float(row.get("mae_days"), 0.0),
            "rmse_days": _float(row.get("rmse_days"), 0.0),
            "bias_days": _float(row.get("bias_days"), 0.0),
            "median_absolute_error_days": _float(
                row.get("median_absolute_error_days"), 0.0
            ),
            "mean_actual_rul_days": _float(row.get("mean_actual_rul_days"), 0.0),
            "mean_predicted_rul_days": _float(row.get("mean_predicted_rul_days"), 0.0),
        }
        for row in _read_rows(RUL_STATE_CSV)
    ]
    trajectory = [
        {
            "checkpoint_observations": _int(row.get("checkpoint_observations")),
            "n_predictions": _int(row.get("n_predictions")),
            "n_hdds": _int(row.get("n_hdds")),
            "mae": _float(row.get("MAE"), 0.0),
            "rmse": _float(row.get("RMSE"), 0.0),
            "bias": _float(row.get("bias"), 0.0),
        }
        for row in _read_rows(RUL_TRAJECTORY_CSV)
    ]
    trajectory.sort(key=lambda p: p["checkpoint_observations"])
    return {"by_state": by_state, "trajectory": trajectory}


@lru_cache(maxsize=1)
def _twin_detail_index() -> Dict[str, Dict[str, str]]:
    return {row.get("serial_number", ""): row for row in _read_rows(TWIN_DETAIL_CSV)}


@lru_cache(maxsize=1)
def _detection_index() -> Dict[str, Dict]:
    return {rec["serial_number"]: rec for rec in get_detections()}


def get_twin_detail(serial_number: str) -> Optional[Dict]:
    row = _twin_detail_index().get(serial_number)
    if row is None:
        return None
    detection = _detection_index().get(serial_number, {})

    failure_date = row.get("failure_date") or detection.get("failure_date")
    detected = bool(detection.get("detected_within_60_days"))

    return {
        "serial_number": serial_number,
        "failure_date": failure_date or None,
        "adaptive_state": row.get("adaptive_state", ""),
        "actual_rul_days": _float(row.get("actual_rul_days")),
        "predicted_rul_days": _float(row.get("predicted_rul_days")),
        "absolute_error_days": _float(row.get("absolute_error_days")),
        "current_degradation_persistence": _float(
            row.get("current_degradation_persistence"), 0.0
        ),
        "longest_degradation_persistence": _float(
            row.get("longest_degradation_persistence"), 0.0
        ),
        "health_features_available": _float(row.get("health_features_available")),
        "detected": detected,
        "first_detection_date": detection.get("first_detection_date"),
        "lead_time_days": detection.get("lead_time_days"),
        "persistence_span_days": detection.get("persistence_span_days"),
        "longest_persistence_run": detection.get("longest_persistence", 0),
    }
