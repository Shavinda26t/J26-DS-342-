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
import logging
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from app.ml import calibration_data
from app.ml._component2_paths import (
    get_detector,
    get_state_manager_class,
)
from app.ml.adaptive_thresholds import AdaptiveThresholds
from app.ml.conformal import ConformalCalibrator, data_quality_score
from app.ml.forecaster import STATE_NAMES, forecast
from app.ml.online_adapter import DualRulAdapter
from app.ml.warning_decision import (
    PRESERVE_GLOBAL_WARNINGS,
    AdaptiveWarningTracker,
    combine_warnings,
)

logger = logging.getLogger(__name__)


class TwinPersistenceError(RuntimeError):
    """Raised when the twin state could not be durably persisted.

    Carries no client-facing detail: the API maps it to a stable 503 message,
    while the technical cause (and traceback) is logged server-side via
    ``logger.exception``.  The original exception is preserved as
    ``__cause__``.
    """

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

# Persistence for twin state (survives restarts).
#
# The store is a single JSON document written atomically (temp file + replace)
# under a lock.  It holds, per serial:
#   * ``history``            - the accepted observations (capped at SEQUENCE_LENGTH)
#   * ``thresholds``         - the adaptive-threshold baseline/learned state
#   * ``warning_tracker``    - the adaptive persistence streak state
#   * ``last_predicted_raw`` - the RAW BiLSTM prediction of the last sync
#   * ``last_observation``   - metadata of the last accepted observation
# and a top-level ``version``/``deleted`` tombstone list so a serial that was
# deleted cannot be resurrected from a stale in-memory cache.
#
# The observation history alone is NOT sufficient to restore a twin: without
# the threshold baseline and warning streak the rebuilt twin reported
# ``baseline_samples == 0`` and re-learned from scratch after every restart.
_RUNTIME_DIR = Path(__file__).resolve().parents[1] / "runtime"
_LEGACY_HISTORY_FILE = _RUNTIME_DIR / "twin_histories.json"
_STATE_FILE = _RUNTIME_DIR / "twin_state.json"
_STATE_VERSION = 2

_detector = None
_state_manager_cls = None
_rul_predictor = None
_predictor_error: Optional[str] = None

_conformal = ConformalCalibrator(alpha=0.1)
_adapter = DualRulAdapter()
_conformal_seeded = False
# Why calibration seeding failed, if it did (surfaced by the diagnostics API).
_conformal_error: Optional[str] = None

_lock = threading.RLock()
_managers: Dict[str, object] = {}
_thresholds: Dict[str, AdaptiveThresholds] = {}
# Per-drive adaptive persistence trackers (Feature 3 warning path).  Fed once
# per ingested observation so the final decision layer can compare the adaptive
# warning with the global one.  See app/ml/warning_decision.py.
_warning_trackers: Dict[str, AdaptiveWarningTracker] = {}
# Raw BiLSTM prediction of the most recent sync, per serial.  Deliberately the
# RAW value: learn_outcome() uses it as the reference for conformal residuals
# and for the RLS update, so it must never be the adapted prediction.
_last_predicted: Dict[str, Optional[float]] = {}
_histories: Dict[str, List[dict]] = {}
# Serialized threshold / warning-tracker state and last-observation metadata,
# restored on startup alongside the histories.
_threshold_states: Dict[str, dict] = {}
_tracker_states: Dict[str, dict] = {}
_last_observation: Dict[str, dict] = {}
# Tombstones: serials explicitly deleted.  Prevents a restart (or a stale
# history file) from resurrecting deleted state.
_deleted: set = set()
# Serial -> last accepted observation identity (dedup key), for the documented
# repeated-observation policy.
_last_identity: Dict[str, str] = {}


# ----------------------------------------------------------------------
# Lazy shared components
# ----------------------------------------------------------------------
def _detector_module():
    global _detector
    if _detector is None:
        _detector = get_detector()
    return _detector


def _observation_gaps_days(manager) -> list:
    """Consecutive calendar-day gaps between the twin's stored observations.

    The forecast horizons are in *days*, but the persistence streak counts
    *observations*; the two are only interchangeable when the stream is daily.
    These gaps let the forecaster validate that instead of assuming it.  Rows
    without a parseable date are skipped rather than imputed.
    """
    import pandas as pd  # local import: this module otherwise only needs numpy

    try:
        history = list(getattr(manager, "history", []) or [])
    except Exception:  # pragma: no cover - defensive
        return []
    stamps = []
    for obs in history:
        raw = obs.get("date") if isinstance(obs, dict) else None
        if raw is None:
            continue
        ts = pd.to_datetime(raw, errors="coerce")
        if ts is not None and not pd.isna(ts):
            stamps.append(ts)
    if len(stamps) < 2:
        return []
    stamps.sort()
    gaps = [
        (stamps[i] - stamps[i - 1]).total_seconds() / 86400.0
        for i in range(1, len(stamps))
    ]
    return [g for g in gaps if g > 0]


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
# State persistence
# ----------------------------------------------------------------------
def _load_state() -> None:
    """Restore persisted twin state on import (application startup).

    Reads the versioned state document.  A legacy history-only file (written by
    the pre-repair service) is migrated: its observations are kept, but the
    threshold baseline is unknown, so those twins start with an empty baseline -
    the same behaviour as before the repair, without inventing statistics.
    """
    if _STATE_FILE.exists():
        try:
            with _STATE_FILE.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            return
        if not isinstance(data, dict):
            return
        serials = data.get("serials")
        if isinstance(serials, dict):
            for serial, record in serials.items():
                if not isinstance(record, dict):
                    continue
                history = record.get("history")
                if isinstance(history, list):
                    _histories[str(serial)] = [h for h in history if isinstance(h, dict)]
                thresholds = record.get("thresholds")
                if isinstance(thresholds, dict):
                    _threshold_states[str(serial)] = thresholds
                tracker = record.get("warning_tracker")
                if isinstance(tracker, dict):
                    _tracker_states[str(serial)] = tracker
                meta = record.get("last_observation")
                if isinstance(meta, dict):
                    _last_observation[str(serial)] = meta
                raw = record.get("last_predicted_raw")
                if isinstance(raw, (int, float)):
                    _last_predicted[str(serial)] = float(raw)
        deleted = data.get("deleted")
        if isinstance(deleted, list):
            _deleted.update(str(s) for s in deleted)
        # A deleted serial must never keep state from a stale record.
        for serial in _deleted:
            _histories.pop(serial, None)
            _threshold_states.pop(serial, None)
            _tracker_states.pop(serial, None)
            _last_observation.pop(serial, None)
            _last_predicted.pop(serial, None)
        return

    # Legacy history-only file (pre-repair).  Migrate read-only.
    if _LEGACY_HISTORY_FILE.exists():
        try:
            with _LEGACY_HISTORY_FILE.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            return
        if isinstance(data, dict):
            for serial, history in data.items():
                if isinstance(history, list):
                    _histories[str(serial)] = [
                        h for h in history if isinstance(h, dict)
                    ]


def _persist_state_locked() -> None:
    """Atomically write the full twin state.  Caller must hold ``_lock``.

    Atomic state *replacement*: the document is written to a temp file, flushed
    and ``os.replace``d over the target, so a crash mid-write cannot leave a
    partially written store and the last known-good file is never replaced by an
    incomplete one.

    A persistence failure is **not** swallowed: the temp file is removed (so no
    partial artifact is left behind), the technical cause is logged with
    ``logger.exception`` and a :class:`TwinPersistenceError` (with the original
    exception as ``__cause__``) is raised.  This lets the caller roll back and
    never report a durable success when nothing was durable.

    Serials in ``_deleted`` are omitted entirely and recorded as tombstones so
    deletion survives a restart.
    """
    tmp = _STATE_FILE.with_suffix(".json.tmp")
    try:
        _RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        serials: Dict[str, dict] = {}
        for serial in sorted(set(_histories) | set(_managers)):
            if serial in _deleted:
                continue
            record: Dict[str, object] = {
                "history": _history_for(serial),
            }
            # Prefer the *live* object when it exists, so the persisted copy can
            # never lag an in-memory mutation; fall back to the cached state
            # (restored from disk, not yet materialised) otherwise.
            if serial in _thresholds:
                record["thresholds"] = _thresholds[serial].to_state()
            elif serial in _threshold_states:
                record["thresholds"] = _threshold_states[serial]
            if serial in _warning_trackers:
                record["warning_tracker"] = _warning_trackers[serial].to_state()
            elif serial in _tracker_states:
                record["warning_tracker"] = _tracker_states[serial]
            if serial in _last_observation:
                record["last_observation"] = _last_observation[serial]
            if serial in _last_predicted and _last_predicted[serial] is not None:
                record["last_predicted_raw"] = float(_last_predicted[serial])
            serials[serial] = record

        document = {
            "version": _STATE_VERSION,
            "deleted": sorted(_deleted),
            "serials": serials,
        }
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(document, handle, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        # Atomic replacement: the target is either the old file or the fully
        # written new one, never a partial document.
        os.replace(tmp, _STATE_FILE)
    except Exception as exc:
        # Clean up the partial temp file so a failed write leaves no artifact
        # and cannot be mistaken for committed state on the next startup.
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:  # pragma: no cover - cleanup is best-effort
            logger.warning(
                "twin state persistence: could not remove temp file %s",
                tmp,
                exc_info=True,
            )
        logger.exception("twin state persistence failed")
        raise TwinPersistenceError(
            "Twin state could not be persisted"
        ) from exc


def _history_for(serial: str) -> List[dict]:
    if serial in _managers:
        return [dict(o) for o in getattr(_managers[serial], "history", [])]
    return list(_histories.get(serial, []))


def _save_state() -> None:
    """Public wrapper: persist under the lock."""
    with _lock:
        _persist_state_locked()


# Backwards-compatible aliases (older tests/tools referenced the history file).
def _load_histories() -> None:
    _load_state()


def _save_histories() -> None:
    _save_state()


# ----------------------------------------------------------------------
# Observation validation / normalisation
# ----------------------------------------------------------------------
# Required SMART attributes: the RUL model's base features.  An observation
# without any of these cannot be scored, so the request is rejected *before*
# any mutation (previously this raised KeyError -> HTTP 500 *after* the
# observation had already been appended to the history).
REQUIRED_SMART = (
    "smart_1_normalized",
    "smart_187_normalized",
)

# Optional SMART attributes: used when present, but the twin degrades
# gracefully (availability flag = 0) when they are absent.  An absent optional
# attribute is *never* silently coerced to 0.0, because 0 is a real measurement
# for several of these counters.
OPTIONAL_SMART = (
    "smart_5_normalized",
    "smart_197_normalized",
    "smart_198_normalized",
)


class ObservationValidationError(ValueError):
    """Raised for a malformed observation payload (mapped to HTTP 4xx)."""


def _is_finite_number(value) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        try:
            return bool(np.isfinite(float(value)))
        except (TypeError, ValueError):
            return False
    return False


def validate_observation(observation: dict) -> dict:
    """Validate and normalise an observation payload.

    Returns a clean, JSON-safe observation dict on success.  Raises
    :class:`ObservationValidationError` (a ``ValueError``) on any malformed
    input, so the API can answer 4xx *before* the twin is mutated:

    * the payload must be a non-empty object,
    * a parseable ``date`` is required (the twin is time-ordered),
    * every required SMART attribute must be a finite number,
    * optional SMART attributes, when present, must be finite numbers or
      ``null``; ``null``/absent means "not measured" (availability 0).
    """
    if not isinstance(observation, dict) or not observation:
        raise ObservationValidationError("observation object is required")

    raw_date = observation.get("date")
    if raw_date is None or (isinstance(raw_date, str) and not raw_date.strip()):
        raise ObservationValidationError("observation.date is required")
    try:
        import pandas as pd  # local import

        parsed = pd.to_datetime(raw_date, errors="coerce")
    except Exception as exc:  # pragma: no cover - defensive
        raise ObservationValidationError(f"observation.date is invalid: {exc}")
    if parsed is None or pd.isna(parsed):
        raise ObservationValidationError(
            f"observation.date is not a parseable timestamp: {raw_date!r}"
        )

    cleaned: Dict[str, object] = {"date": str(raw_date)}

    for smart in REQUIRED_SMART:
        if smart not in observation:
            raise ObservationValidationError(
                f"observation.{smart} is required"
            )
        value = observation.get(smart)
        if not _is_finite_number(value):
            raise ObservationValidationError(
                f"observation.{smart} must be a finite number, got {value!r}"
            )
        cleaned[smart] = float(value)

    for smart in OPTIONAL_SMART:
        if smart not in observation:
            continue
        value = observation.get(smart)
        if value is None:
            continue
        if not _is_finite_number(value):
            raise ObservationValidationError(
                f"observation.{smart} must be a finite number or null, "
                f"got {value!r}"
            )
        cleaned[smart] = float(value)

    # Pass through any other scalar fields (dates, pre-normalised features) so
    # existing valid request formats keep working, but reject obviously
    # unserialisable nested payloads.
    for key, value in observation.items():
        if key in cleaned or key in REQUIRED_SMART or key in OPTIONAL_SMART:
            continue
        if isinstance(value, (dict, list, tuple, set)):
            continue
        if isinstance(value, float) and not np.isfinite(value):
            raise ObservationValidationError(
                f"observation.{key} must be finite, got {value!r}"
            )
        cleaned[key] = value

    return cleaned


def _normalize_observation(observation: dict) -> dict:
    """Ensure availability flags exist (derived from the SMART values)."""
    obs = dict(observation)
    for smart, flag in zip(CORE_SMART, AVAILABILITY_FLAGS):
        if flag not in obs:
            value = obs.get(smart)
            available = _is_finite_number(value)
            obs[flag] = 1 if available else 0
    return obs


def observation_identity(observation: dict) -> str:
    """Stable identity of an observation, used for the dedup policy.

    The public API supplies no unique observation id, so identity is the
    ``date`` (the twin is a time series keyed by observation timestamp).
    Re-sending an observation whose timestamp is already present in the
    retained history is a duplicate and is **not** re-ingested (see
    ``sync_observation``), which makes replaying a stream idempotent.
    """
    return str(observation.get("date"))


def _ensure_twin(serial: str):
    """Get (or lazily create/rehydrate) the state manager for a serial.

    Rehydration restores the *full* learned state: observation history, the
    adaptive-threshold baseline windows and the warning streak.  Restoring only
    the history (the pre-repair behaviour) reset ``baseline_samples`` to 0 on
    every restart, so a drive silently re-learned its noise from scratch.
    """
    if serial not in _managers:
        cls = _state_manager_class()
        manager = cls(serial_number=serial, history_size=SEQUENCE_LENGTH)
        for obs in _histories.get(serial, []):
            manager.add_observation(obs)
        _managers[serial] = manager
    if serial not in _thresholds:
        thresholds = AdaptiveThresholds(serial_number=serial)
        state = _threshold_states.get(serial)
        if state:
            thresholds.load_state(state)
        _thresholds[serial] = thresholds
    return _managers[serial], _thresholds[serial]


def _warning_tracker(serial: str) -> AdaptiveWarningTracker:
    """Get (or lazily create/rehydrate) the adaptive persistence tracker."""
    tracker = _warning_trackers.get(serial)
    if tracker is None:
        tracker = AdaptiveWarningTracker(drive=serial, window=SEQUENCE_LENGTH)
        state = _tracker_states.get(serial)
        if state:
            tracker.load_state(state)
        _warning_trackers[serial] = tracker
    return tracker


# ----------------------------------------------------------------------
# Calibration seeding (leakage-free calibration split)
# ----------------------------------------------------------------------
def _seed_calibration() -> None:
    """Seed the conformal calibrator from a leakage-free calibration split.

    The population is built by :mod:`app.ml.calibration_data` from the seq_30
    **validation** split (89 drives / 2922 observations), produced by the
    *deployed* checkpoint and its preprocessing parameters.  The drives used
    for the project's final evaluation (the seq_30 **test** split) are
    excluded and :class:`~app.ml.calibration_data.CalibrationLeakageError` is
    raised if that ever stops being true.

    This replaces the previous seed, which read
    ``digital_twin_rul/heldout_rul_trajectory_results.csv``: all 90 of those
    drives are test-split drives (301/303 rows reproduce the committed test
    predictions) and the file mixes sequence lengths 1/7/14/30, so the old
    intervals were calibrated in-sample.

    If the calibration source cannot be trusted the calibrator is left empty:
    the twin then reports ``calibrated: false`` and uses a documented
    conservative band instead of silently falling back to in-sample residuals.
    """
    global _conformal_seeded, _conformal_error
    if _conformal_seeded:
        return
    _conformal_seeded = True
    try:
        data = calibration_data.load_calibration()
    except Exception as exc:  # missing files / leakage / bad provenance
        _conformal_error = f"{type(exc).__name__}: {exc}"
        return

    try:
        accepted = _conformal.seed_from_pairs(data.pairs)
        diagnostics = dict(data.diagnostics)
        diagnostics.update(
            {
                "seeded_residuals": int(accepted),
                "retained_residuals": _conformal.n_calibration,
                "rejected_residuals": _conformal.n_rejected,
                "seed_truncated": bool(accepted != len(data.pairs)),
            }
        )
        _conformal.source = diagnostics
        # Population-level prior for the online adapter (global source).
        # Drive-level fit + out-of-sample gate: the correction is only
        # activated if it beats the raw prediction on held-back drives.
        # NOTE: the pairs are now drawn from the calibration split (never the
        # final evaluation split); the adapter's own gate decides whether the
        # correction may switch on at all.
        _adapter.seed_population(data.pairs, data.serials)
    except Exception as exc:  # pragma: no cover - defensive
        _conformal_error = f"{type(exc).__name__}: {exc}"


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def reset_service() -> None:
    """Clear all runtime state (used by tests).

    Removes the persisted store rather than writing an empty one, so a test that
    only resets the service leaves no stray runtime artifact behind.  A
    subsequent ingest recreates the store.
    """
    global _conformal_seeded, _conformal_error, _predictor_error, _rul_predictor
    with _lock:
        _managers.clear()
        _thresholds.clear()
        _warning_trackers.clear()
        _histories.clear()
        _last_predicted.clear()
        _threshold_states.clear()
        _tracker_states.clear()
        _last_observation.clear()
        _last_identity.clear()
        _deleted.clear()
        _conformal.reset()
        _adapter.reset()
        _conformal_seeded = False
        _conformal_error = None
        _rul_predictor = None
        _predictor_error = None
        # Best-effort by design: ``reset_service`` is a test/reset utility that
        # clears the in-memory twin set.  Removing the persisted store so a
        # subsequent ingest starts clean is *not* a durability guarantee, so a
        # failure here is logged rather than raised (and never leaves a partial
        # document behind - ``_persist_state_locked`` is not used at all).
        try:
            if _STATE_FILE.exists():
                _STATE_FILE.unlink()
        except OSError:
            logger.warning(
                "reset_service could not remove the state file", exc_info=True
            )


def compute_state(serial_number: str) -> dict:
    """**Pure** state assembly for ``serial``.  No mutation, no persistence.

    Every function called here is read-only on the stored twin: the adaptive
    thresholds are *not* fed (that is ingestion-only, see
    :func:`sync_observation`), the online adapter is not trained, and no
    prediction state is written.  Repeated reads therefore return identical
    results (test: ``TestReadsArePure``).
    """
    manager, thresholds = _ensure_twin(serial_number)
    return _build_state(serial_number, manager, thresholds)


def sync_observation(serial_number: str, observation: dict) -> dict:
    """Ingest one validated observation and return the resulting twin state.

    Lifecycle (each accepted observation processed exactly once):

    1. **Validate** the complete request (serial + payload).  A malformed
       payload raises :class:`~app.services.twin_sync_service.ObservationValidationError`
       *before* anything is mutated, so a rejected request leaves the twin
       untouched (no HTTP 500, no partial write).
    2. **Normalise & dedup**: identity is the observation ``date``.  Re-sending
       an already-stored timestamp is a duplicate and is ignored (no repeated
       baseline training).
    3. **Compute** the RUL/detector results for the new observation.
    4. **Apply** the threshold/baseline update exactly once and **commit**
       atomically; the whole update runs under ``_lock`` and on a working copy,
       so a failure rolls back to the last committed state.

    Returns the committed state plus a ``sync`` block describing what happened
    (``accepted`` / ``duplicate`` / ``reason``).
    """
    if not isinstance(serial_number, str) or not serial_number.strip():
        raise ObservationValidationError("serial_number is required")

    observation = validate_observation(observation)
    _seed_calibration()

    with _lock:
        # Capture the pre-sync deletion state *before* mutating ``_deleted``.
        # Re-using a deleted serial recreates it cleanly, but if the recreation
        # fails to persist the rollback must restore the original tombstone so
        # the serial is not accidentally resurrected.
        was_deleted = serial_number in _deleted

        # Was the twin materialised before this call?  Used to remove a phantom
        # empty twin if the very first sync of a new serial fails to persist.
        # Captured before any state mutation so the rollback can undo creation.
        had_twin = (
            serial_number in _managers
            or serial_number in _histories
            or serial_number in _thresholds
            or serial_number in _warning_trackers
        )

        # Only after both snapshots are taken may the tombstone be discarded to
        # permit recreation.  ``_ensure_twin`` assumes the caller holds the lock.
        if was_deleted:
            _deleted.discard(serial_number)

        manager, thresholds = _ensure_twin(serial_number)
        identity = observation_identity(observation)
        normalized = _normalize_observation(observation)

        # Dedup policy (documented API semantics): the twin is a time series
        # keyed by observation date and the API supplies no unique observation
        # id, so an observation whose timestamp is *already present in the
        # retained history* is a duplicate and is not re-ingested.  This makes
        # re-sending a whole stream idempotent instead of re-training the
        # baseline once per replay.
        stored = _last_identity.get(serial_number)
        if stored is None and manager.history:
            stored = observation_identity(manager.history[-1])
        existing = {
            observation_identity(o) for o in manager.history
        }
        duplicate = (stored is not None and identity == stored) or (
            identity in existing
        )

        if not duplicate:
            # --- Snapshot everything a commit could touch ----------------
            # A failure anywhere below (model compute *or* persistence) must
            # leave the in-memory twin exactly as it was, and must never report
            # a durable success.
            previous_history = list(manager.history)
            previous_last_update = manager.last_update
            previous_threshold_state = thresholds.to_state()
            previous_tracker_state = _warning_tracker(serial_number).to_state()
            previous_last_predicted = _last_predicted.get(serial_number)
            previous_last_observation = _last_observation.get(serial_number)
            previous_last_identity = stored
            had_history = serial_number in _histories
            previous_history_global = _histories.get(serial_number)
            had_threshold_state = serial_number in _threshold_states
            previous_threshold_state_global = _threshold_states.get(serial_number)
            had_tracker_state = serial_number in _tracker_states
            previous_tracker_state_global = _tracker_states.get(serial_number)

            try:
                manager.add_observation(normalized)
                # The threshold baseline is updated here - once per accepted
                # observation - and nowhere else.  It is never updated by a GET.
                temporal = manager.calculate_temporal_features()
                change_1 = change_187 = None
                if temporal is not None and not temporal.empty:
                    latest = temporal.iloc[-1]
                    change_1 = latest.get("smart_1_normalized_change_7d")
                    change_187 = latest.get("smart_187_normalized_change_7d")
                deg = manager.calculate_degradation_state()
                detector = _detector_module()
                global_sev = deg["combined_severity"] if deg else np.nan
                qualifying = (not _missing(global_sev)) and (
                    global_sev >= detector.MIN_SEVERITY
                )
                thresholds.observe(change_1, change_187, qualifying)

                # Adaptive warning streak (fed once per accepted observation).
                adaptive_severity = thresholds.severity(change_1, change_187)
                _warning_tracker(serial_number).observe(
                    adaptive_severity,
                    observation_key=(
                        manager.last_update,
                        manager.get_history_length(),
                    ),
                )

                # Compute the committed state before committing anything.
                state = _build_state(serial_number, manager, thresholds)

                # --- Prepare the candidate in memory ----------------------
                _histories[serial_number] = list(manager.history)
                _threshold_states[serial_number] = thresholds.to_state()
                _tracker_states[serial_number] = _warning_tracker(
                    serial_number
                ).to_state()
                _last_identity[serial_number] = identity
                _last_observation[serial_number] = {
                    "identity": identity,
                    "date": manager.last_update,
                    "history_length": manager.get_history_length(),
                }
                # Store the RAW BiLSTM prediction for the online-feedback loop
                # (never the adapted value).
                raw_value = state.get("raw_predicted_rul_days")
                if raw_value is not None:
                    _last_predicted[serial_number] = float(raw_value)

                # --- Persist the candidate; only then is the sync durable --
                # If this raises TwinPersistenceError the whole commit is rolled
                # back below, so a durability failure never silently claims the
                # observation was stored.
                _persist_state_locked()
            except Exception:
                # Roll back to the last committed state.
                manager.history = previous_history
                manager.last_update = previous_last_update
                thresholds.load_state(previous_threshold_state)
                _warning_tracker(serial_number).load_state(previous_tracker_state)
                if previous_last_predicted is None:
                    _last_predicted.pop(serial_number, None)
                else:
                    _last_predicted[serial_number] = previous_last_predicted
                if previous_last_observation is None:
                    _last_observation.pop(serial_number, None)
                else:
                    _last_observation[serial_number] = previous_last_observation
                if previous_last_identity is None:
                    _last_identity.pop(serial_number, None)
                else:
                    _last_identity[serial_number] = previous_last_identity
                if had_history:
                    _histories[serial_number] = previous_history_global
                else:
                    _histories.pop(serial_number, None)
                if had_threshold_state:
                    _threshold_states[serial_number] = previous_threshold_state_global
                else:
                    _threshold_states.pop(serial_number, None)
                if had_tracker_state:
                    _tracker_states[serial_number] = previous_tracker_state_global
                else:
                    _tracker_states.pop(serial_number, None)
                # If this was the *first* sync of a brand-new serial, remove the
                # empty twin that ``_ensure_twin`` materialised so a failed
                # create does not leave a phantom serial behind.
                if not had_twin:
                    _managers.pop(serial_number, None)
                    _thresholds.pop(serial_number, None)
                    _warning_trackers.pop(serial_number, None)
                # A failed recreation of a deleted serial leaves it deleted.
                if was_deleted:
                    _deleted.add(serial_number)
                raise

            # --- Commit --------------------------------------------------
            # The state is already persisted (durable); the in-memory commit is
            # confirmed by the successful write above.
        else:
            state = _build_state(serial_number, manager, thresholds)

        state["sync"] = {
            "accepted": not duplicate,
            "duplicate": duplicate,
            "identity": identity,
            "reason": (
                "duplicate observation (same date already stored)"
                if duplicate
                else None
            ),
        }
        return state


def get_state(serial_number: str) -> Optional[dict]:
    """Read the current live twin state without ingesting data.

    Side-effect free: no observation is added, no baseline update is applied,
    no adapter is trained, no timestamp is changed.  Returns ``None`` for an
    unknown or deleted serial.
    """
    _seed_calibration()
    with _lock:
        if serial_number in _deleted:
            return None
        if serial_number not in _managers and serial_number not in _histories:
            return None
        return compute_state(serial_number)


def list_serials() -> List[str]:
    with _lock:
        return sorted(
            (set(_managers) | set(_histories)) - _deleted
        )


def reset_serial(serial_number: str) -> bool:
    """Delete a twin and all its persisted state.

    The serial is recorded as a tombstone so a restart (or a stale in-memory
    cache) cannot resurrect it.

    Deletion is transactional like ingestion: if the tombstone cannot be
    persisted the in-memory deletion is rolled back and a
    :class:`TwinPersistenceError` is raised, so a caller never believes a twin
    was durably deleted when disk still holds it.
    """
    with _lock:
        existed = (
            serial_number in _managers
            or serial_number in _histories
            or serial_number in _thresholds
        )
        # Snapshot for rollback.
        prev_managers = _managers.get(serial_number)
        prev_thresholds = _thresholds.get(serial_number)
        prev_tracker = _warning_trackers.get(serial_number)
        prev_histories = _histories.get(serial_number)
        prev_predicted = _last_predicted.get(serial_number)
        prev_threshold_state = _threshold_states.get(serial_number)
        prev_tracker_state = _tracker_states.get(serial_number)
        prev_last_obs = _last_observation.get(serial_number)
        prev_identity = _last_identity.get(serial_number)
        prev_adapter = _adapter.serial_adapters.get(serial_number)
        was_deleted = serial_number in _deleted

        _managers.pop(serial_number, None)
        _thresholds.pop(serial_number, None)
        _warning_trackers.pop(serial_number, None)
        _histories.pop(serial_number, None)
        _last_predicted.pop(serial_number, None)
        _threshold_states.pop(serial_number, None)
        _tracker_states.pop(serial_number, None)
        _last_observation.pop(serial_number, None)
        _last_identity.pop(serial_number, None)
        # Drop the drive's learned correction along with its history.
        _adapter.serial_adapters.pop(serial_number, None)
        _deleted.add(serial_number)
        try:
            _persist_state_locked()
        except TwinPersistenceError:
            # Restore the pre-delete in-memory state; disk is unchanged.
            def _restore(mapping, key, value, present):
                if present:
                    mapping[key] = value
                else:
                    mapping.pop(key, None)

            _restore(_managers, serial_number, prev_managers, prev_managers is not None)
            _restore(_thresholds, serial_number, prev_thresholds, prev_thresholds is not None)
            _restore(_warning_trackers, serial_number, prev_tracker, prev_tracker is not None)
            _restore(_histories, serial_number, prev_histories, prev_histories is not None)
            _restore(_last_predicted, serial_number, prev_predicted, prev_predicted is not None)
            _restore(_threshold_states, serial_number, prev_threshold_state, prev_threshold_state is not None)
            _restore(_tracker_states, serial_number, prev_tracker_state, prev_tracker_state is not None)
            _restore(_last_observation, serial_number, prev_last_obs, prev_last_obs is not None)
            _restore(_last_identity, serial_number, prev_identity, prev_identity is not None)
            _restore(_adapter.serial_adapters, serial_number, prev_adapter, prev_adapter is not None)
            if not was_deleted:
                _deleted.discard(serial_number)
            raise
        return existed


def is_deleted(serial_number: str) -> bool:
    """True when ``serial`` was explicitly deleted (tombstoned)."""
    with _lock:
        return serial_number in _deleted


def learn_outcome(serial_number: str, actual_rul_days: float) -> dict:
    """
    Feed a realised RUL back to the twin: this is where **genuine online
    learning** happens (updates conformal calibration + RLS adapter).

    The reference prediction is always the stored **raw** BiLSTM output, never
    the adapted value (Task 1 contract), so the adaptation never feeds its own
    correction back into itself.
    """
    if actual_rul_days is None:
        raise ValueError("actual_rul_days is required")
    if not _is_finite_number(actual_rul_days):
        raise ValueError("actual_rul_days must be a finite number")
    actual_rul_days = float(actual_rul_days)
    with _lock:
        if serial_number in _deleted:
            raise ValueError(
                f"Twin '{serial_number}' was deleted; sync an observation "
                "before learning."
            )
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
    """Conformal calibration + adapter diagnostics (auditable provenance)."""
    _seed_calibration()
    return {
        "conformal": _conformal.snapshot(),
        "online_adapter": _adapter.snapshot(),
        "calibration_error": _conformal_error,
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
    # NOTE: this is a *read*.  ``thresholds.observe`` is called exactly once
    # per accepted observation in ``sync_observation``; calling it here (as the
    # pre-repair code did) made every GET mutate the baseline.
    deg = manager.calculate_degradation_state()
    global_sev = deg["combined_severity"] if deg else np.nan
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
        # ``adapt`` is a pure read of the adapter: it forwards the raw value to
        # compute the correction and never trains.  ``_last_predicted`` is
        # written by ``sync_observation`` (the single ingest path), not here, so
        # a GET cannot change the stored feedback reference.
        adapted = _adapter.adapt(serial, raw_rul)
        final_rul = adapted["adapted"]

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
        # Cadence evidence: the persistence streak counts observations while the
        # horizons are in days.  Passing the measured gaps lets the forecaster
        # validate the conversion (and report it as unsupported when it cannot)
        # instead of silently assuming daily sampling.
        observation_gaps_days=_observation_gaps_days(manager),
        observation_count=history_length,
    )

    # --- Health label / final warning decision --------------------------
    # Global detector decision (source of truth): a completed persistence
    # episode (streak >= PERSISTENCE) is "Elevated Degradation" (state >= 2).
    global_state = int(component_state.get("adaptive_state", 0))
    global_streak = int(component_state.get("moderate_plus_streak", 0))

    # Adaptive decision path over the same observations.  The tracker is fed
    # once per accepted observation in ``sync_observation``; here it is only
    # *read*, so repeated state requests cannot advance the streak.
    tracker = _warning_tracker(serial)

    # Final decision layer: hard global-warning preservation by construction
    # (max of the two streaks), never a severity-score comparison.
    decision = combine_warnings(
        global_streak,
        tracker.streak(),
        detector.PERSISTENCE,
        preserve_global_warnings=PRESERVE_GLOBAL_WARNINGS,
    )
    state_level = decision.state

    return {
        "serial_number": serial,
        "last_update": component_state.get("last_update"),
        "history_length": history_length,
        # Feature 5 state (component source of truth)
        "adaptive_state": state_level,
        "adaptive_state_name": (
            detector.adaptive_state_name(state_level)
            if state_level == global_state
            else component_state.get("adaptive_state_name")
        ),
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
        # Final warning decision (global-warning preservation).  With the
        # production contract on, ``adaptive_state``/``state_level`` equal the
        # global decision; the adaptive-only decision is reported separately.
        "warning_decision": {
            "global_streak": decision.global_streak,
            "adaptive_streak": decision.adaptive_streak,
            "global_state": decision.global_state,
            "adaptive_warning_state": decision.adaptive_state,
            "state": decision.state,
            "preserved": decision.preserved,
            "suppressed": decision.suppressed,
            "source": decision.source,
            "preserve_global_warnings": decision.preserve_global_warnings,
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


# Restore persisted twin state at import (application startup).
_load_state()
