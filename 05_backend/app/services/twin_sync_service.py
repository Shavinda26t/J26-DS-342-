"""Live Digital Twin synchronization service (Feature 5 + integrator).

This is the runtime heart of the enhanced Component 2 twin. It holds a live,
per-HDD twin that:

* **Synchronizes** new SMART observations as they stream in
  (``DigitalTwinStateManager`` from Component 2 = single source of truth).
* Runs **adaptive, per-drive thresholds** (Feature 3).
* Produces a **RUL point estimate** from the frozen BiLSTM, corrected by a
  **genuine online learner** (Feature 1).
* Wraps it in a **quality-gated conformal interval + confidence** (Feature 2).
* Projects a **multi-horizon health forecast** (Feature 4).

Observation histories are persisted to disk so twin state survives restarts and
can be re-synchronised.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from app.ml._component2_paths import (
    component2_results_dir,
    get_detector,
    get_state_manager_class,
)
from app.ml.adaptive_thresholds import AdaptiveThresholds
from app.ml.conformal import ConformalCalibrator, data_quality_score
from app.ml.forecaster import STATE_NAMES, forecast
from app.ml.online_adapter import DualRulAdapter

# Base SMART features the RUL model expects (normalized values).
CORE_SMART = [
    "smart_1_normalized",
    "smart_5_normalized",
    "smart_187_normalized",
    "smart_197_normalized",
    "smart_198_normalized",
]

AVAILABILITY_FLAGS = [
    "smart_1_available",
    "smart_5_available",
    "smart_187_available",
    "smart_197_available",
    "smart_198_available",
]

TEMPORAL_FEATURES = [
    "smart_1_normalized_change_7d",
    "smart_1_normalized_change_14d",
    "smart_1_normalized_rolling_std_7",
    "smart_187_normalized_change_7d",
    "smart_187_normalized_change_14d",
    "smart_187_normalized_rolling_std_7",
]

SEQUENCE_LENGTH = 30


def _missing(value) -> bool:
    """True for None / NaN / pandas-NaN-like values."""
    if value is None:
        return True
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return False

# Persistence for observation histories (survives restarts).
_RUNTIME_DIR = Path(__file__).resolve().parents[1] / "runtime"
_HISTORY_FILE = _RUNTIME_DIR / "twin_histories.json"

_detector = None
_state_manager_cls = None
_rul_predictor = None
_predictor_error: Optional[str] = None

_conformal = ConformalCalibrator(alpha=0.1)
_adapter = DualRulAdapter()
_conformal_seeded = False

_lock = threading.RLock()
_managers: Dict[str, object] = {}
_thresholds: Dict[str, AdaptiveThresholds] = {}
_last_predicted: Dict[str, Optional[float]] = {}
_histories: Dict[str, List[dict]] = {}


# ----------------------------------------------------------------------
# Lazy shared components
# ----------------------------------------------------------------------
def _detector_module():
    global _detector
    if _detector is None:
        _detector = get_detector()
    return _detector


def _state_manager_class():
    global _state_manager_cls
    if _state_manager_cls is None:
        _state_manager_cls = get_state_manager_class()
    return _state_manager_cls


def _get_predictor():
    """Load the frozen BiLSTM once; surface a clear error if unavailable."""
    global _rul_predictor, _predictor_error
    if _rul_predictor is None and _predictor_error is None:
        try:
            from app.ml.component2_rul_predictor import (  # noqa: PLC0415
                Component2RULPredictor,
            )

            _rul_predictor = Component2RULPredictor()
        except Exception as exc:  # pragma: no cover - env dependent
            _predictor_error = str(exc)
    return _rul_predictor


# ----------------------------------------------------------------------
# History persistence
# ----------------------------------------------------------------------
def _load_histories() -> None:
    if not _HISTORY_FILE.exists():
        return
    try:
        with _HISTORY_FILE.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            _histories.update(data)
    except Exception:
        pass


def _save_histories() -> None:
    try:
        _RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        tmp = _HISTORY_FILE.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(_histories, handle, default=str)
        tmp.replace(_HISTORY_FILE)
    except Exception:
        pass


# ----------------------------------------------------------------------
# Observation normalisation
# ----------------------------------------------------------------------
def _normalize_observation(observation: dict) -> dict:
    """Ensure availability flags exist (derived from the SMART values)."""
    obs = dict(observation)
    for smart, flag in zip(CORE_SMART, AVAILABILITY_FLAGS):
        if flag not in obs:
            value = obs.get(smart)
            available = value is not None and not (
                isinstance(value, float) and np.isnan(value)
            )
            obs[flag] = 1 if available else 0
    return obs


def _ensure_twin(serial: str):
    """Get (or lazily create/rehydrate) the state manager for a serial."""
    if serial not in _managers:
        cls = _state_manager_class()
        manager = cls(serial_number=serial, history_size=SEQUENCE_LENGTH)
        for obs in _histories.get(serial, []):
            manager.add_observation(obs)
        _managers[serial] = manager
    if serial not in _thresholds:
        _thresholds[serial] = AdaptiveThresholds(serial_number=serial)
    return _managers[serial], _thresholds[serial]


# ----------------------------------------------------------------------
# Calibration seeding (from Component 2 held-out residuals)
# ----------------------------------------------------------------------
def _seed_calibration() -> None:
    global _conformal_seeded
    if _conformal_seeded:
        return
    _conformal_seeded = True
    path = (
        component2_results_dir()
        / "digital_twin_rul"
        / "heldout_rul_trajectory_results.csv"
    )
    if not path.exists():
        return
    try:
        import csv  # noqa: PLC0415

        pairs = []
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                pred = row.get("predicted_rul_days")
                act = row.get("actual_rul_days")
                if pred in (None, "") or act in (None, ""):
                    continue
                pairs.append((float(pred), float(act)))
        _conformal.seed_from_pairs(pairs)
        # Population-level prior for the online adapter (global source).
        for pred, act in pairs:
            _adapter.global_adapter.update(pred, act)
    except Exception:
        pass


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def reset_service() -> None:
    """Clear all runtime state (used by tests)."""
    global _conformal_seeded, _predictor_error, _rul_predictor
    with _lock:
        _managers.clear()
        _thresholds.clear()
        _histories.clear()
        _last_predicted.clear()
        _conformal._residuals.clear()
        _adapter.serial_adapters.clear()
        _adapter.global_adapter = type(_adapter.global_adapter)()
        _conformal_seeded = False
        _rul_predictor = None
        _predictor_error = None
        _save_histories()


def sync_observation(serial_number: str, observation: dict) -> dict:
    """Ingest one observation and return the full live twin state."""
    if not serial_number:
        raise ValueError("serial_number is required")
    if not isinstance(observation, dict) or not observation:
        raise ValueError("observation object is required")

    _seed_calibration()

    with _lock:
        obs = _normalize_observation(observation)
        manager, thresholds = _ensure_twin(serial_number)
        manager.add_observation(obs)

        # Persist history.
        _histories[serial_number] = list(manager.history)
        _save_histories()

        return _build_state(serial_number, manager, thresholds)


def get_state(serial_number: str) -> Optional[dict]:
    """Read the current live twin state without ingesting data."""
    _seed_calibration()
    with _lock:
        if serial_number not in _managers and serial_number not in _histories:
            return None
        manager, thresholds = _ensure_twin(serial_number)
        return _build_state(serial_number, manager, thresholds)


def list_serials() -> List[str]:
    with _lock:
        return sorted(set(_managers) | set(_histories))


def reset_serial(serial_number: str) -> bool:
    with _lock:
        existed = serial_number in _managers or serial_number in _histories
        _managers.pop(serial_number, None)
        _thresholds.pop(serial_number, None)
        _histories.pop(serial_number, None)
        _last_predicted.pop(serial_number, None)
        _save_histories()
        return existed


def learn_outcome(serial_number: str, actual_rul_days: float) -> dict:
    """
    Feed a realised RUL back to the twin: this is where **genuine online
    learning** happens (updates conformal calibration + RLS adapter).
    """
    if actual_rul_days is None:
        raise ValueError("actual_rul_days is required")
    with _lock:
        predicted = _last_predicted.get(serial_number)
        if predicted is None:
            raise ValueError(
                f"No predicted RUL on record for '{serial_number}'. "
                "Sync at least one observation first."
            )
        _conformal.add_residual(predicted, actual_rul_days)
        _adapter.update(serial_number, predicted, actual_rul_days)
        manager, thresholds = _ensure_twin(serial_number)
        return _build_state(serial_number, manager, thresholds)


def calibration_snapshot() -> dict:
    _seed_calibration()
    return {
        "conformal": _conformal.snapshot(),
        "online_adapter": _adapter.snapshot(),
    }


# ----------------------------------------------------------------------
# State assembly
# ----------------------------------------------------------------------
def _build_state(serial: str, manager, thresholds) -> dict:
    detector = _detector_module()

    # --- Component 2 streaming state (source of truth) -----------------
    component_state = manager.get_current_state()

    # --- Temporal features (for thresholds + RUL input) ----------------
    temporal = manager.calculate_temporal_features()
    change_1 = change_187 = None
    temporal_valid_fraction = None
    if temporal is not None and not temporal.empty:
        latest = temporal.iloc[-1]
        change_1 = latest.get("smart_1_normalized_change_7d")
        change_187 = latest.get("smart_187_normalized_change_7d")
        valid = sum(
            1
            for f in TEMPORAL_FEATURES
            if latest.get(f) is not None and not (
                isinstance(latest.get(f), float) and np.isnan(latest.get(f))
            )
        )
        temporal_valid_fraction = valid / len(TEMPORAL_FEATURES)

    # --- Feature 3: adaptive thresholds -------------------------------
    deg = manager.calculate_degradation_state()
    global_sev = deg["combined_severity"] if deg else np.nan
    qualifying = (not _missing(global_sev)) and (
        global_sev >= detector.MIN_SEVERITY
    )
    thresholds.observe(change_1, change_187, qualifying)
    adaptive_severity = thresholds.severity(change_1, change_187)
    acceleration_z = thresholds.acceleration_z(change_1, change_187)
    adaptive_persistence = thresholds.suggested_persistence()
    threshold_snap = thresholds.snapshot()

    # --- RUL prediction (frozen BiLSTM) --------------------------------
    history_length = manager.get_history_length()
    raw_rul: Optional[float] = None
    rul_error: Optional[str] = None
    predictor = _get_predictor()
    if history_length >= SEQUENCE_LENGTH and temporal is not None:
        if predictor is not None:
            try:
                raw_rul = float(predictor.predict(temporal))
            except Exception as exc:
                rul_error = str(exc)
        else:
            rul_error = _predictor_error or "RUL model unavailable"
    else:
        rul_error = (
            f"Need {SEQUENCE_LENGTH} observations for RUL, have {history_length}"
        )

    # --- Data quality + conformal + online adaptation -------------------
    health_available = None
    if temporal is not None and not temporal.empty:
        row = temporal.iloc[-1]
        health_available = sum(
            1 for f in AVAILABILITY_FLAGS if row.get(f)
        )

    quality = data_quality_score(
        health_features_available=health_available,
        history_length=history_length,
        history_target=SEQUENCE_LENGTH,
        temporal_valid_fraction=temporal_valid_fraction,
    )

    adapted: Optional[dict] = None
    final_rul: Optional[float] = None
    if raw_rul is not None:
        adapted = _adapter.adapt(serial, raw_rul)
        final_rul = adapted["adapted"]
        _last_predicted[serial] = final_rul

    interval = None
    confidence = None
    if final_rul is not None:
        interval = _conformal.interval(final_rul, quality)
        confidence = _conformal.confidence(final_rul, quality, interval)

    # --- Feature 4: forecast -------------------------------------------
    base_half = interval["base_half_width"] if interval else 0.0
    forecast_payload = forecast(
        predicted_rul=final_rul,
        current_state=int(component_state.get("adaptive_state", 0)),
        current_streak=int(component_state.get("moderate_plus_streak", 0)),
        persistence=int(adaptive_persistence),
        base_half_width=float(base_half or 0.0),
    )

    # --- Health label ---------------------------------------------------
    state_level = int(component_state.get("adaptive_state", 0))

    return {
        "serial_number": serial,
        "last_update": component_state.get("last_update"),
        "history_length": history_length,
        # Feature 5 state (component source of truth)
        "adaptive_state": state_level,
        "adaptive_state_name": component_state.get("adaptive_state_name"),
        "streak": component_state.get("moderate_plus_streak"),
        "degradation_evidence_available": component_state.get(
            "degradation_evidence_available"
        ),
        "global_severity": (
            None if _missing(global_sev) else int(global_sev)
        ),
        "smart_1_severity": (
            None if _missing(component_state.get("smart_1_severity"))
            else int(component_state["smart_1_severity"])
        ),
        "smart_187_severity": (
            None if _missing(component_state.get("smart_187_severity"))
            else int(component_state["smart_187_severity"])
        ),
        # Feature 3
        "adaptive_severity": adaptive_severity,
        "acceleration_z": acceleration_z,
        "adaptive_thresholds": threshold_snap,
        "detector": {
            "global_persistence": int(detector.PERSISTENCE),
            "adaptive_persistence": int(adaptive_persistence),
            "min_severity": int(detector.MIN_SEVERITY),
        },
        # Feature 1 + 2
        "predicted_rul_days": (
            round(final_rul, 3) if final_rul is not None else None
        ),
        "raw_predicted_rul_days": (
            round(raw_rul, 3) if raw_rul is not None else None
        ),
        "online_adaptation": adapted,
        "interval": interval,
        "confidence": confidence,
        "data_quality": round(quality, 3),
        "health_features_available": health_available,
        "temporal_valid_fraction": (
            round(temporal_valid_fraction, 3)
            if temporal_valid_fraction is not None
            else None
        ),
        "rul_error": rul_error,
        # Feature 4
        "forecast": forecast_payload,
    }


# Load persisted histories at import.
_load_histories()
