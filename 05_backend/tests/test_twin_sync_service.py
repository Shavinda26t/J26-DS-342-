"""Tests for twin synchronisation, persistence and API validation (Task 5).

These pin the repaired contract:

* read operations (``GET /twin/{serial}/state``, ``list_serials``) are
  **side-effect free** - no baseline update, no adapter training, no timestamp
  change;
* each accepted observation is processed **exactly once**: the threshold
  baseline is updated on ingest and only on ingest;
* duplicate observations (same ``date``) follow the documented dedup policy;
* ``POST /twin/sync`` is **atomic**: validation happens before any mutation and
  a failure rolls back to the last committed state;
* malformed / non-finite payloads return **422**, not HTTP 500;
* complete twin state (history + threshold baseline + warning streak + raw
  prediction) **survives a restart**;
* **deletion** survives a restart (tombstones);
* raw and adapted predictions stay distinct.

Storage is redirected to a temporary file per test, so the real project state is
never touched.

Run with:
    cd 05_backend
    python -m pytest tests/test_twin_sync_service.py -q
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.services import twin_sync_service as svc


def _observation(day: int, **overrides) -> dict:
    import datetime

    date = datetime.date(2024, 1, 1) + datetime.timedelta(days=day - 1)
    obs = {
        "date": date.isoformat(),
        "smart_1_normalized": 100.0 + day * 0.1,
        "smart_5_normalized": 0.0 + day * 0.001,
        "smart_187_normalized": 0.0 + day * 0.01,
        "smart_197_normalized": 1.0,
        "smart_198_normalized": 0.0 + day * 0.001,
    }
    obs.update(overrides)
    return obs


class _IsolatedStorage(unittest.TestCase):
    """Base: point the state store at a temp file and disable the predictor."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._state_file = Path(self._tmpdir.name) / "twin_state.json"
        self._legacy_file = Path(self._tmpdir.name) / "twin_histories.json"
        self._orig_state = svc._STATE_FILE
        self._orig_legacy = svc._LEGACY_HISTORY_FILE
        self._orig_runtime = svc._RUNTIME_DIR
        svc._STATE_FILE = self._state_file
        svc._LEGACY_HISTORY_FILE = self._legacy_file
        svc._RUNTIME_DIR = Path(self._tmpdir.name)

        svc.reset_service()
        # Keep the (real, on-disk) calibration seed and BiLSTM out of the way.
        svc._conformal_seeded = True
        svc._predictor_error = "disabled in test"
        svc._rul_predictor = None

    def tearDown(self) -> None:
        svc.reset_service()
        svc._STATE_FILE = self._orig_state
        svc._LEGACY_HISTORY_FILE = self._orig_legacy
        svc._RUNTIME_DIR = self._orig_runtime
        self._tmpdir.cleanup()

    def _restart(self) -> None:
        """Simulate an application restart: drop in-memory state, reload disk.

        Keeps the persistent dicts (they are repopulated by ``_load_state``),
        exactly as a new process would.
        """
        svc._managers.clear()
        svc._thresholds.clear()
        svc._warning_trackers.clear()
        svc._histories.clear()
        svc._last_predicted.clear()
        svc._threshold_states.clear()
        svc._tracker_states.clear()
        svc._last_observation.clear()
        svc._last_identity.clear()
        svc._deleted.clear()
        svc._load_state()


class TestReadsArePure(_IsolatedStorage):
    """Requirement 1: repeated GETs do not mutate stored state."""

    def test_repeated_get_does_not_change_baseline(self) -> None:
        for day in range(1, 32):
            svc.sync_observation("READ-1", _observation(day))
        before = svc.get_state("READ-1")
        for _ in range(5):
            svc.get_state("READ-1")
        after = svc.get_state("READ-1")
        self.assertEqual(
            before["adaptive_thresholds"]["baseline_samples"],
            after["adaptive_thresholds"]["baseline_samples"],
        )
        self.assertEqual(
            before["adaptive_thresholds"]["window_samples"],
            after["adaptive_thresholds"]["window_samples"],
        )

    def test_get_returns_identical_payload(self) -> None:
        for day in range(1, 32):
            svc.sync_observation("READ-2", _observation(day))
        first = svc.get_state("READ-2")
        for _ in range(5):
            svc.get_state("READ-2")
        self.assertEqual(first, svc.get_state("READ-2"))

    def test_get_does_not_change_history_or_timestamp(self) -> None:
        for day in range(1, 32):
            svc.sync_observation("READ-3", _observation(day))
        before_hist = list(svc._managers["READ-3"].history)
        before_ts = svc._managers["READ-3"].last_update
        for _ in range(5):
            svc.get_state("READ-3")
        self.assertEqual(svc._managers["READ-3"].history, before_hist)
        self.assertEqual(svc._managers["READ-3"].last_update, before_ts)

    def test_get_does_not_train_the_adapter(self) -> None:
        for day in range(1, 32):
            svc.sync_observation("READ-4", _observation(day))
        before = svc._adapter._get("READ-4").n_updates
        for _ in range(5):
            svc.get_state("READ-4")
        self.assertEqual(svc._adapter._get("READ-4").n_updates, before)

    def test_get_does_not_advance_warning_streak(self) -> None:
        for day in range(1, 32):
            svc.sync_observation("READ-5", _observation(day))
        tracker = svc._warning_trackers["READ-5"]
        snap_before = tracker.snapshot()
        for _ in range(5):
            svc.get_state("READ-5")
        self.assertEqual(tracker.snapshot(), snap_before)

    def test_get_does_not_rewrite_state_file(self) -> None:
        for day in range(1, 32):
            svc.sync_observation("READ-6", _observation(day))
        contents = self._state_file.read_text(encoding="utf-8")
        for _ in range(5):
            svc.get_state("READ-6")
        self.assertEqual(self._state_file.read_text(encoding="utf-8"), contents)


class TestExactlyOnce(_IsolatedStorage):
    """Requirement 2: one accepted observation -> one baseline update."""

    def _baseline(self, serial: str) -> int:
        return svc.get_state(serial)["adaptive_thresholds"]["baseline_samples"]

    def test_one_observation_one_update(self) -> None:
        serial = "ONCE-1"
        # Let the temporal features warm up (the first observations cannot
        # produce a change signal yet), then measure a single increment.
        for day in range(1, 10):
            svc.sync_observation(serial, _observation(day))
        first = self._baseline(serial)
        svc.sync_observation(serial, _observation(10))
        second = self._baseline(serial)
        # Exactly one observation was ingested -> at most one new sample per
        # signal (two signals are observed here), never more.
        self.assertGreaterEqual(second - first, 0)
        self.assertLessEqual(second - first, 2)
        self.assertGreater(second, first)

    def test_baseline_only_moves_on_ingest(self) -> None:
        serial = "ONCE-2"
        for day in range(1, 11):
            svc.sync_observation(serial, _observation(day))
        baseline = self._baseline(serial)
        for _ in range(10):
            svc.get_state(serial)
        self.assertEqual(self._baseline(serial), baseline)


class TestDeduplication(_IsolatedStorage):
    """Requirement 3: duplicate observations follow the documented policy."""

    def test_duplicate_observation_is_not_reingested(self) -> None:
        serial = "DUP-1"
        first = svc.sync_observation(serial, _observation(5))
        self.assertTrue(first["sync"]["accepted"])
        baseline = first["adaptive_thresholds"]["baseline_samples"]

        again = svc.sync_observation(serial, _observation(5))
        self.assertFalse(again["sync"]["accepted"])
        self.assertTrue(again["sync"]["duplicate"])
        self.assertEqual(again["history_length"], first["history_length"])
        self.assertEqual(
            again["adaptive_thresholds"]["baseline_samples"], baseline
        )

    def test_different_timestamp_is_accepted(self) -> None:
        serial = "DUP-2"
        svc.sync_observation(serial, _observation(5))
        state = svc.sync_observation(serial, _observation(6))
        self.assertTrue(state["sync"]["accepted"])
        self.assertEqual(state["history_length"], 2)

    def test_repeated_full_stream_is_idempotent(self) -> None:
        serial = "DUP-3"
        for day in range(1, 11):
            svc.sync_observation(serial, _observation(day))
        state = svc.get_state(serial)
        baseline = state["adaptive_thresholds"]["baseline_samples"]
        for day in range(1, 11):
            svc.sync_observation(serial, _observation(day))
        after = svc.get_state(serial)
        self.assertEqual(after["history_length"], state["history_length"])
        self.assertEqual(
            after["adaptive_thresholds"]["baseline_samples"], baseline
        )

    def test_duplicate_after_restart_is_detected(self) -> None:
        serial = "DUP-4"
        for day in range(1, 11):
            svc.sync_observation(serial, _observation(day))
        before = svc.get_state(serial)
        self._restart()
        again = svc.sync_observation(serial, _observation(10))
        self.assertTrue(again["sync"]["duplicate"])
        self.assertEqual(again["history_length"], before["history_length"])

    def test_duplicate_timestamp_with_new_values_is_rejected(self) -> None:
        serial = "DUP-5"
        svc.sync_observation(serial, _observation(5, smart_1_normalized=1.0))
        state = svc.sync_observation(
            serial, _observation(5, smart_1_normalized=999.0)
        )
        self.assertTrue(state["sync"]["duplicate"])
        # The originally stored value is untouched (no silent overwrite).
        stored = svc._managers[serial].history[-1]["smart_1_normalized"]
        self.assertEqual(stored, 1.0)


class TestValidation(_IsolatedStorage):
    """Requirements 4 and 5: missing/invalid fields are client errors."""

    def test_missing_required_field_rejected(self) -> None:
        with self.assertRaises(svc.ObservationValidationError):
            svc.sync_observation(
                "VAL-1", {"date": "2024-01-01", "smart_1_normalized": 1.0}
            )

    def test_rejected_request_leaves_no_state(self) -> None:
        with self.assertRaises(svc.ObservationValidationError):
            svc.sync_observation(
                "VAL-2", {"date": "2024-01-01", "smart_1_normalized": 1.0}
            )
        self.assertNotIn("VAL-2", svc._histories)
        self.assertNotIn("VAL-2", svc._managers)
        self.assertIsNone(svc.get_state("VAL-2"))

    def test_missing_date_rejected(self) -> None:
        with self.assertRaises(svc.ObservationValidationError):
            svc.sync_observation(
                "VAL-3",
                {"smart_1_normalized": 1.0, "smart_187_normalized": 2.0},
            )

    def test_unparseable_date_rejected(self) -> None:
        with self.assertRaises(svc.ObservationValidationError):
            svc.sync_observation(
                "VAL-4",
                {
                    "date": "not-a-date",
                    "smart_1_normalized": 1.0,
                    "smart_187_normalized": 2.0,
                },
            )

    def test_non_finite_values_rejected(self) -> None:
        for bad in (float("nan"), float("inf"), float("-inf"), "abc", None):
            with self.assertRaises(svc.ObservationValidationError):
                svc.sync_observation(
                    "VAL-5",
                    _observation(1, smart_1_normalized=bad),
                )

    def test_optional_fields_are_optional(self) -> None:
        # smart_5 / smart_197 / smart_198 absent -> accepted; only the two
        # required features are marked available (count, not "unavailable").
        state = svc.sync_observation(
            "VAL-6",
            {
                "date": "2024-01-01",
                "smart_1_normalized": 1.0,
                "smart_187_normalized": 2.0,
            },
        )
        self.assertTrue(state["sync"]["accepted"])
        self.assertEqual(state["health_features_available"], 2)
        # The absent optional attributes are NOT silently stored as 0.0.
        stored = svc._managers["VAL-6"].history[-1]
        self.assertNotIn("smart_5_normalized", stored)
        self.assertNotIn("smart_197_normalized", stored)
        self.assertNotIn("smart_198_normalized", stored)

    def test_optional_null_is_not_zero(self) -> None:
        state = svc.sync_observation(
            "VAL-7",
            {
                "date": "2024-01-01",
                "smart_1_normalized": 1.0,
                "smart_187_normalized": 2.0,
                "smart_5_normalized": None,
            },
        )
        # Availability flag 0 because it was not measured; the value is not 0.
        self.assertEqual(state["sync"]["accepted"], True)
        stored = svc._managers["VAL-7"].history[-1]
        self.assertNotIn("smart_5_normalized", stored)

    def test_empty_payload_rejected(self) -> None:
        with self.assertRaises(svc.ObservationValidationError):
            svc.sync_observation("VAL-8", {})

    def test_missing_serial_rejected(self) -> None:
        with self.assertRaises(svc.ObservationValidationError):
            svc.sync_observation("", _observation(1))


class TestAtomicity(_IsolatedStorage):
    """Requirement 6 and 7: rejected/failed sync commits nothing."""

    def test_failure_preserves_last_committed_state(self) -> None:
        serial = "ATOM-1"
        for day in range(1, 11):
            svc.sync_observation(serial, _observation(day))
        before = svc.get_state(serial)

        # Force a failure inside the commit path.
        manager = svc._managers[serial]
        original = manager.calculate_temporal_features

        def boom():
            raise RuntimeError("simulated failure")

        manager.calculate_temporal_features = boom
        try:
            with self.assertRaises(RuntimeError):
                svc.sync_observation(serial, _observation(11))
        finally:
            manager.calculate_temporal_features = original

        after = svc.get_state(serial)
        self.assertEqual(after["history_length"], before["history_length"])
        self.assertEqual(
            after["adaptive_thresholds"]["baseline_samples"],
            before["adaptive_thresholds"]["baseline_samples"],
        )
        self.assertEqual(after["last_update"], before["last_update"])

    def test_rejected_payload_does_not_change_existing_twin(self) -> None:
        serial = "ATOM-2"
        svc.sync_observation(serial, _observation(1))
        before = svc.get_state(serial)
        with self.assertRaises(svc.ObservationValidationError):
            svc.sync_observation(
                serial, {"date": "2024-01-02", "smart_1_normalized": 1.0}
            )
        after = svc.get_state(serial)
        self.assertEqual(after, before)


class TestPersistence(_IsolatedStorage):
    """Requirement 8: state (incl. baseline) survives a restart."""

    def test_baseline_survives_restart(self) -> None:
        serial = "PERSIST-1"
        for day in range(1, 32):
            svc.sync_observation(serial, _observation(day))
        before = svc.get_state(serial)

        self._restart()

        after = svc.get_state(serial)
        self.assertIsNotNone(after)
        self.assertEqual(after["history_length"], before["history_length"])
        self.assertEqual(
            after["adaptive_thresholds"]["baseline_samples"],
            before["adaptive_thresholds"]["baseline_samples"],
        )
        self.assertEqual(
            after["adaptive_thresholds"]["smart_1_strictness"],
            before["adaptive_thresholds"]["smart_1_strictness"],
        )
        self.assertEqual(after["adaptive_state"], before["adaptive_state"])

    def test_warning_streak_survives_restart(self) -> None:
        serial = "PERSIST-2"
        for day in range(1, 32):
            svc.sync_observation(serial, _observation(day))
        before = svc.get_state(serial)["warning_decision"]

        self._restart()

        after = svc.get_state(serial)["warning_decision"]
        self.assertEqual(after["global_streak"], before["global_streak"])
        self.assertEqual(after["adaptive_streak"], before["adaptive_streak"])

    def test_raw_prediction_survives_restart(self) -> None:
        serial = "PERSIST-3"

        class FakePredictor:
            def predict(self, temporal):
                return 42.5

        svc._rul_predictor = FakePredictor()
        svc._predictor_error = None
        for day in range(1, 32):
            svc.sync_observation(serial, _observation(day))
        self.assertEqual(svc._last_predicted[serial], 42.5)

        svc._rul_predictor = None
        svc._predictor_error = "disabled in test"
        self._restart()

        self.assertEqual(svc._last_predicted.get(serial), 42.5)
        # ... and learn-after-restart works off the restored raw value.
        state = svc.learn_outcome(serial, 40.0)
        self.assertIsNotNone(state)

    def test_state_file_is_versioned(self) -> None:
        svc.sync_observation("PERSIST-4", _observation(1))
        doc = json.loads(self._state_file.read_text(encoding="utf-8"))
        self.assertEqual(doc["version"], svc._STATE_VERSION)
        record = doc["serials"]["PERSIST-4"]
        self.assertIn("thresholds", record)
        self.assertIn("warning_tracker", record)

    def test_corrupt_state_file_is_ignored(self) -> None:
        self._state_file.write_text("{not valid json", encoding="utf-8")
        svc._managers.clear()
        svc._histories.clear()
        svc._load_state()  # must not raise
        self.assertEqual(svc.list_serials(), [])


class TestDeletion(_IsolatedStorage):
    """Requirements 9, 10, 11: deletion semantics and unknown serials."""

    def test_delete_persists_across_restart(self) -> None:
        serial = "DEL-1"
        for day in range(1, 6):
            svc.sync_observation(serial, _observation(day))
        self.assertTrue(svc.reset_serial(serial))
        self.assertIsNone(svc.get_state(serial))

        self._restart()

        self.assertIsNone(svc.get_state(serial))
        self.assertNotIn(serial, svc.list_serials())
        self.assertTrue(svc.is_deleted(serial))

    def test_unknown_serial_returns_none(self) -> None:
        self.assertIsNone(svc.get_state("NEVER-SEEN"))

    def test_learn_after_delete_is_rejected(self) -> None:
        serial = "DEL-2"
        svc.sync_observation(serial, _observation(1))
        svc.reset_serial(serial)
        with self.assertRaises(ValueError):
            svc.learn_outcome(serial, 10.0)

    def test_recreate_deleted_serial(self) -> None:
        serial = "DEL-3"
        for day in range(1, 6):
            svc.sync_observation(serial, _observation(day))
        svc.reset_serial(serial)
        state = svc.sync_observation(serial, _observation(1))
        self.assertTrue(state["sync"]["accepted"])
        self.assertEqual(state["history_length"], 1)
        self.assertFalse(svc.is_deleted(serial))

    def test_reset_returns_false_for_unknown(self) -> None:
        self.assertFalse(svc.reset_serial("NEVER-SEEN"))


class TestOutOfOrderAndCaps(_IsolatedStorage):
    """Requirement 12: out-of-order timestamps and the 30-observation cap."""

    def test_out_of_order_observations_are_sorted(self) -> None:
        serial = "ORDER-1"
        for day in (5, 3, 1, 4, 2):
            svc.sync_observation(serial, _observation(day))
        history = svc._managers[serial].history
        dates = [h["date"] for h in history]
        self.assertEqual(dates, sorted(dates))
        self.assertEqual(svc.get_state(serial)["last_update"], "2024-01-05")

    def test_history_capped_at_sequence_length(self) -> None:
        serial = "ORDER-2"
        for day in range(1, 41):
            svc.sync_observation(serial, _observation(day))
        self.assertEqual(
            svc.get_state(serial)["history_length"], svc.SEQUENCE_LENGTH
        )


class TestPredictionProvenance(_IsolatedStorage):
    """Requirement 14: raw and adapted predictions stay distinct."""

    def test_raw_stored_even_when_adaptation_engaged(self) -> None:
        serial = "PROV-1"

        class FakePredictor:
            def predict(self, temporal):
                return 30.0

        svc._rul_predictor = FakePredictor()
        svc._predictor_error = None
        pairs, serials = [], []
        for i in range(40):
            for j in range(12):
                raw = 5.0 + ((i * 12 + j) * 7) % 55
                pairs.append((float(raw), float(raw) + 6.0))
                serials.append(f"D{i // 12}")
        diag = svc._adapter.seed_population(pairs, serials)
        self.assertEqual(diag["status"], "validated")

        state = None
        for day in range(1, 32):
            state = svc.sync_observation(serial, _observation(day))

        self.assertTrue(state["online_adaptation"]["enabled"])
        self.assertNotEqual(
            state["predicted_rul_days"], state["raw_predicted_rul_days"]
        )
        # The stored feedback reference is the RAW value, not the adapted one.
        self.assertEqual(svc._last_predicted[serial], 30.0)


class TestConcurrency(_IsolatedStorage):
    """Requirement 13: concurrent updates for one serial do not corrupt state."""

    def test_threaded_syncs_are_serialised(self) -> None:
        import threading

        serial = "CONC-1"
        errors = []

        def worker(start: int) -> None:
            try:
                for day in range(start, start + 20):
                    svc.sync_observation(
                        serial,
                        {
                            "date": f"2024-{1 + (day // 28):02d}-"
                                    f"{1 + (day % 28):02d}",
                            "smart_1_normalized": 100.0 + day * 0.01,
                            "smart_5_normalized": day * 0.001,
                            "smart_187_normalized": day * 0.001,
                            "smart_197_normalized": 1.0,
                            "smart_198_normalized": day * 0.001,
                        },
                    )
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)

        threads = [
            threading.Thread(target=worker, args=(offset,))
            for offset in range(0, 100, 20)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        history = svc._managers[serial].history
        dates = [h["date"] for h in history]
        # No duplicates, chronological, capped at the window size.
        self.assertEqual(len(dates), len(set(dates)))
        self.assertEqual(dates, sorted(dates))
        self.assertLessEqual(len(dates), svc.SEQUENCE_LENGTH)


class TestApiContracts(_IsolatedStorage):
    """Requirement 15 + status codes: existing flows and client errors."""

    def setUp(self) -> None:
        super().setUp()
        self.client = TestClient(app)

    def test_valid_sync_and_state_flow(self) -> None:
        resp = self.client.post(
            "/api/v1/component2/twin/sync",
            json={"serial_number": "API-1", "observation": _observation(1)},
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["sync"]["accepted"])
        self.assertIn("forecast", body)

        state = self.client.get("/api/v1/component2/twin/API-1/state")
        self.assertEqual(state.status_code, 200)
        self.assertEqual(state.json()["history_length"], 1)

    def test_missing_required_field_is_422_not_500(self) -> None:
        resp = self.client.post(
            "/api/v1/component2/twin/sync",
            json={
                "serial_number": "API-2",
                "observation": {"date": "2024-01-01", "smart_1_normalized": 1.0},
            },
        )
        self.assertEqual(resp.status_code, 422)
        self.assertIn("smart_187_normalized", resp.json()["detail"])

    def test_non_finite_field_is_422(self) -> None:
        # JSON cannot encode NaN directly, but a string is still invalid.
        resp = self.client.post(
            "/api/v1/component2/twin/sync",
            json={
                "serial_number": "API-3",
                "observation": _observation(1, smart_1_normalized="oops"),
            },
        )
        self.assertEqual(resp.status_code, 422)

    def test_unknown_serial_state_is_404(self) -> None:
        resp = self.client.get("/api/v1/component2/twin/UNKNOWN-API/state")
        self.assertEqual(resp.status_code, 404)

    def test_learn_without_sync_is_400(self) -> None:
        resp = self.client.post(
            "/api/v1/component2/twin/API-4/learn",
            json={"actual_rul_days": 10.0},
        )
        self.assertEqual(resp.status_code, 400)

    def test_serials_and_calibration_endpoints(self) -> None:
        self.client.post(
            "/api/v1/component2/twin/sync",
            json={"serial_number": "API-5", "observation": _observation(1)},
        )
        serials = self.client.get("/api/v1/component2/twin/serials").json()
        self.assertIn("API-5", serials["serials"])
        calib = self.client.get("/api/v1/component2/twin/calibration")
        self.assertEqual(calib.status_code, 200)

    def test_delete_endpoint_removes_serial(self) -> None:
        self.client.post(
            "/api/v1/component2/twin/sync",
            json={"serial_number": "API-6", "observation": _observation(1)},
        )
        deleted = self.client.delete("/api/v1/component2/twin/API-6")
        self.assertEqual(deleted.status_code, 200)
        self.assertTrue(deleted.json()["reset"])
        self.assertEqual(
            self.client.get("/api/v1/component2/twin/API-6/state").status_code,
            404,
        )


class TestPersistenceFailures(_IsolatedStorage):
    """Task 5.1: a durability failure is explicit, safe and rolled back."""

    def _seed(self, serial: str = "PFAIL-1", days: int = 5) -> None:
        for day in range(1, days + 1):
            svc.sync_observation(serial, _observation(day))

    def _snapshot(self) -> dict:
        """Everything a sync commit could touch, for before/after comparison."""
        return {
            "histories": {k: list(v) for k, v in svc._histories.items()},
            "managers": {
                k: [dict(o) for o in v.history] for k, v in svc._managers.items()
            },
            "last_update": {
                k: v.last_update for k, v in svc._managers.items()
            },
            "thresholds": {k: v.to_state() for k, v in svc._thresholds.items()},
            "threshold_states": {
                k: dict(v) for k, v in svc._threshold_states.items()
            },
            "trackers": {
                k: v.to_state() for k, v in svc._warning_trackers.items()
            },
            "tracker_states": {
                k: dict(v) for k, v in svc._tracker_states.items()
            },
            "last_observation": {
                k: dict(v) for k, v in svc._last_observation.items()
            },
            "last_predicted": dict(svc._last_predicted),
            "last_identity": dict(svc._last_identity),
            "deleted": set(svc._deleted),
        }

    def _fail_replace(self):
        """Context manager: make ``os.replace`` raise, restoring it afterwards."""
        import contextlib

        real_replace = svc.os.replace

        @contextlib.contextmanager
        def _ctx():
            def boom(src, dst):
                raise PermissionError("simulated replace failure")

            svc.os.replace = boom
            try:
                yield
            finally:
                svc.os.replace = real_replace

        return _ctx()

    # ------------------------------------------------------------------
    def test_failed_save_raises_explicit_persistence_error(self) -> None:
        serial = "PFAIL-1"
        self._seed(serial)
        before = self._snapshot()

        with self._fail_replace():
            with self.assertRaises(svc.TwinPersistenceError) as ctx:
                svc.sync_observation(serial, _observation(6))

        # The original exception is preserved as the cause.
        self.assertIsInstance(ctx.exception.__cause__, PermissionError)
        # Nothing about the twin changed in memory.
        self.assertEqual(self._snapshot(), before)

    def test_failed_save_preserves_the_valid_json_file(self) -> None:
        serial = "PFAIL-2"
        self._seed(serial)
        self.assertTrue(self._state_file.exists())
        on_disk = self._state_file.read_text(encoding="utf-8")

        with self._fail_replace():
            with self.assertRaises(svc.TwinPersistenceError):
                svc.sync_observation(serial, _observation(6))

        # The last known-good document is untouched and still parses.
        self.assertEqual(self._state_file.read_text(encoding="utf-8"), on_disk)
        doc = json.loads(self._state_file.read_text(encoding="utf-8"))
        self.assertEqual(doc["version"], svc._STATE_VERSION)

    def test_failed_save_cleans_up_the_temp_file(self) -> None:
        serial = "PFAIL-3"
        self._seed(serial)
        tmp = self._state_file.with_suffix(".json.tmp")

        with self._fail_replace():
            with self.assertRaises(svc.TwinPersistenceError):
                svc.sync_observation(serial, _observation(6))

        self.assertFalse(tmp.exists())

    def test_subsequent_successful_sync_works_after_failure(self) -> None:
        serial = "PFAIL-4"
        self._seed(serial)
        before_len = svc.get_state(serial)["history_length"]

        with self._fail_replace():
            with self.assertRaises(svc.TwinPersistenceError):
                svc.sync_observation(serial, _observation(6))

        # Same observation, now with a working filesystem: it is accepted and
        # the twin advances exactly one step (the failed attempt left no mark).
        state = svc.sync_observation(serial, _observation(6))
        self.assertTrue(state["sync"]["accepted"])
        self.assertEqual(state["history_length"], before_len + 1)
        self.assertIn("PFAIL-4", json.loads(
            self._state_file.read_text(encoding="utf-8")
        )["serials"])

    def test_failed_save_after_restart_still_reflects_the_committed_state(self):
        serial = "PFAIL-5"
        self._seed(serial)

        with self._fail_replace():
            with self.assertRaises(svc.TwinPersistenceError):
                svc.sync_observation(serial, _observation(6))

        committed = svc.get_state(serial)
        self._restart()
        reloaded = svc.get_state(serial)
        self.assertEqual(reloaded["history_length"], committed["history_length"])
        self.assertEqual(
            reloaded["adaptive_thresholds"]["baseline_samples"],
            committed["adaptive_thresholds"]["baseline_samples"],
        )

    def test_failed_write_mid_document_does_not_replace_good_file(self) -> None:
        """A failing ``json.dump`` (partial temp file) leaves disk consistent."""
        import contextlib

        serial = "PFAIL-6"
        self._seed(serial)
        on_disk = self._state_file.read_text(encoding="utf-8")
        tmp = self._state_file.with_suffix(".json.tmp")

        def boom(obj, handle, **kwargs):
            handle.write('{"version": 2, "partial"')  # deliberately truncated
            raise OSError("simulated disk-full mid-write")

        real_dump = svc.json.dump
        svc.json.dump = boom
        try:
            with self.assertRaises(svc.TwinPersistenceError):
                svc.sync_observation(serial, _observation(6))
        finally:
            svc.json.dump = real_dump

        # The old file is intact and the partial temp file was removed.
        self.assertEqual(self._state_file.read_text(encoding="utf-8"), on_disk)
        self.assertFalse(tmp.exists())
        json.loads(self._state_file.read_text(encoding="utf-8"))

    def test_api_sync_returns_503_on_persistence_failure(self) -> None:
        from fastapi.testclient import TestClient

        from app.main import app

        client = TestClient(app)
        serial = "PFAIL-API-1"
        self._seed(serial)

        with self._fail_replace():
            resp = client.post(
                "/api/v1/component2/twin/sync",
                json={"serial_number": serial, "observation": _observation(6)},
            )

        self.assertEqual(resp.status_code, 503)
        # Stable, non-sensitive message: no paths, no traceback.
        detail = resp.json()["detail"]
        self.assertEqual(detail, "Twin state could not be persisted")
        self.assertNotIn("\\", detail)
        self.assertNotIn(".json", detail)
        self.assertNotIn("PermissionError", detail)

    def test_api_sync_503_does_not_mutate_the_twin(self) -> None:
        from fastapi.testclient import TestClient

        from app.main import app

        client = TestClient(app)
        serial = "PFAIL-API-2"
        self._seed(serial)
        before = self._snapshot()

        with self._fail_replace():
            resp = client.post(
                "/api/v1/component2/twin/sync",
                json={"serial_number": serial, "observation": _observation(6)},
            )
        self.assertEqual(resp.status_code, 503)
        self.assertEqual(self._snapshot(), before)

    def test_delete_persistence_failure_rolls_back_deletion(self) -> None:
        serial = "PFAIL-DEL"
        self._seed(serial)

        with self._fail_replace():
            with self.assertRaises(svc.TwinPersistenceError):
                svc.reset_serial(serial)

        # The twin is still present and not tombstoned.
        self.assertFalse(svc.is_deleted(serial))
        self.assertIsNotNone(svc.get_state(serial))
        self.assertIn(serial, svc.list_serials())

    def test_failed_first_sync_leaves_no_phantom_twin(self) -> None:
        """A failed create must not leave an empty twin visible."""
        serial = "PFAIL-NEW"

        with self._fail_replace():
            with self.assertRaises(svc.TwinPersistenceError):
                svc.sync_observation(serial, _observation(1))

        self.assertNotIn(serial, svc._managers)
        self.assertNotIn(serial, svc._histories)
        self.assertIsNone(svc.get_state(serial))
        self.assertNotIn(serial, svc.list_serials())

    def test_failed_recreation_of_deleted_serial_stays_deleted(self) -> None:
        """A failed recreation of a deleted serial must not resurrect it."""
        serial = "PFAIL-DELREC"
        self._seed(serial)
        svc.reset_serial(serial)
        self.assertTrue(svc.is_deleted(serial))

        with self._fail_replace():
            with self.assertRaises(svc.TwinPersistenceError):
                svc.sync_observation(serial, _observation(1))

        # Still tombstoned: the failed recreation did not resurrect it.
        self.assertTrue(svc.is_deleted(serial))
        self.assertIsNone(svc.get_state(serial))
        self.assertNotIn(serial, svc.list_serials())

    def test_persistence_error_is_subclass_of_runtime_error(self) -> None:
        self.assertTrue(issubclass(svc.TwinPersistenceError, RuntimeError))

    def test_reset_service_file_removal_is_best_effort(self) -> None:
        """``reset_service`` is documented best-effort: a locked file is logged,
        not raised, and never corrupts the store."""
        serial = "PFAIL-RESET"
        self._seed(serial)

        import pathlib

        original = pathlib.Path.exists

        def patched(self_path, *a, **k):
            if self_path == self._state_file:
                raise OSError("simulated locked file")
            return original(self_path, *a, **k)

        pathlib.Path.exists = patched
        try:
            svc.reset_service()  # must not raise
        finally:
            pathlib.Path.exists = original


if __name__ == "__main__":
    unittest.main()
