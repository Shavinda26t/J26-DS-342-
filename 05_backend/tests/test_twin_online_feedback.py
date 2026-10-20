"""Integration tests for the twin's online-learning feedback loop.

These tests pin the contract that the *raw* BiLSTM prediction -- never the
adapted one -- is what the twin stores and feeds back:

* ``GET  /twin/{serial}/state`` and ``POST /twin/sync`` expose the raw and
  adapted values as separate fields,
* ``POST /twin/{serial}/learn`` updates the adapter (and the conformal
  calibrator) with the raw prediction of the most recent sync.

They exercise ``app.services.twin_sync_service`` directly, which is exactly
what the Component 2 API routes are thin wrappers around.

Run with:
    cd 05_backend
    python -m pytest tests/test_twin_online_feedback.py -q
"""

from __future__ import annotations

import datetime
import tempfile
import unittest
from pathlib import Path

from app.services import twin_sync_service as sync_service

OBSERVATION_DATE = datetime.date(2024, 1, 1)


def _observation(day: int) -> dict:
    """A minimal but complete observation payload (all CORE_SMART keys)."""
    return {
        "date": (OBSERVATION_DATE + datetime.timedelta(days=day)).isoformat(),
        "smart_1_normalized": 100.0 + day * 0.1,
        "smart_5_normalized": 0.0 + day * 0.001,
        "smart_187_normalized": 0.0 + day * 0.01,
        "smart_197_normalized": 1.0,
        "smart_198_normalized": 0.0 + day * 0.001,
    }


def _seed_pairs(drives: int = 30, per_drive: int = 12, bias: float = 6.0):
    """Synthetic calibration pairs with a clean, learnable relation."""
    pairs, serials = [], []
    for index in range(drives):
        serial = f"SEED{index:03d}"
        for step in range(per_drive):
            raw = 5.0 + ((index * per_drive + step) * 7) % 55
            pairs.append((float(raw), float(raw) + bias))
            serials.append(serial)
    return pairs, serials


class TestTwinOnlineFeedback(unittest.TestCase):
    SERIAL = "TEST-SERIAL-FEEDBACK"

    @classmethod
    def setUpClass(cls) -> None:
        # Isolate persistence in a temp store: this suite syncs many
        # observations, and writing the real project runtime file lets a file
        # handle from another test block ``os.replace`` on Windows.
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls._orig_runtime = sync_service._RUNTIME_DIR
        cls._orig_state = sync_service._STATE_FILE
        cls._orig_legacy = sync_service._LEGACY_HISTORY_FILE
        sync_service._RUNTIME_DIR = Path(cls._tmpdir.name)
        sync_service._STATE_FILE = Path(cls._tmpdir.name) / "twin_state.json"
        sync_service._LEGACY_HISTORY_FILE = (
            Path(cls._tmpdir.name) / "twin_histories.json"
        )
        sync_service.reset_service()

    @classmethod
    def tearDownClass(cls) -> None:
        sync_service.reset_service()
        sync_service._RUNTIME_DIR = cls._orig_runtime
        sync_service._STATE_FILE = cls._orig_state
        sync_service._LEGACY_HISTORY_FILE = cls._orig_legacy
        cls._tmpdir.cleanup()

    def setUp(self) -> None:
        """Cold adapter per test, without paying the predictor reload cost."""
        sync_service._adapter.reset()
        sync_service._last_predicted.clear()
        sync_service._managers.clear()
        sync_service._thresholds.clear()
        sync_service._histories.clear()
        # keep the (real, on-disk) calibration seed out of the way so each
        # test controls the adapter state explicitly
        sync_service._conformal_seeded = True

    def _sync_history(self, serial: str, length: int = 31) -> dict:
        state = None
        for day in range(length):
            state = sync_service.sync_observation(serial, _observation(day))
        return state

    # ------------------------------------------------------------------
    def test_sync_exposes_raw_separately_and_stores_raw_for_feedback(self):
        state = self._sync_history(self.SERIAL)

        self.assertEqual(state["history_length"], 30)
        raw = state["raw_predicted_rul_days"]
        self.assertIsNotNone(raw)

        adaptation = state["online_adaptation"]
        self.assertEqual(adaptation["raw"], raw)
        # no validated evidence yet -> the correction is not applied
        self.assertFalse(adaptation["enabled"])
        self.assertEqual(adaptation["correction_days"], 0.0)
        self.assertEqual(state["predicted_rul_days"], raw)

        # ... and the value kept for online feedback is the RAW prediction
        # (stored at full precision, displayed rounded)
        self.assertEqual(
            round(sync_service._last_predicted[self.SERIAL], 3), raw
        )

    # ------------------------------------------------------------------
    def test_learn_outcome_updates_with_the_raw_prediction(self):
        state = self._sync_history(self.SERIAL)
        raw = sync_service._last_predicted[self.SERIAL]
        self.assertEqual(state["raw_predicted_rul_days"], round(raw, 3))

        personal = sync_service._adapter._get(self.SERIAL)
        self.assertEqual(personal.n_updates, 0)

        sync_service.learn_outcome(self.SERIAL, raw - 5.0)

        self.assertEqual(personal.n_updates, 1)
        self.assertEqual(personal.last_predicted, raw)  # raw, never adapted
        self.assertEqual(personal.last_actual, raw - 5.0)

    # ------------------------------------------------------------------
    def test_engaged_adaptation_still_feeds_back_the_raw_prediction(self):
        diagnostics = sync_service._adapter.seed_population(*_seed_pairs())
        self.assertEqual(diagnostics["status"], "validated")

        state = self._sync_history(self.SERIAL)
        adaptation = state["online_adaptation"]

        self.assertTrue(adaptation["enabled"])
        self.assertEqual(adaptation["source"], "global")
        self.assertNotEqual(adaptation["adapted"], adaptation["raw"])
        self.assertNotEqual(state["predicted_rul_days"],
                            state["raw_predicted_rul_days"])

        raw = state["raw_predicted_rul_days"]
        # the feedback reference is still the raw prediction
        self.assertEqual(
            round(sync_service._last_predicted[self.SERIAL], 3), raw
        )

        personal = sync_service._adapter._get(self.SERIAL)
        sync_service.learn_outcome(self.SERIAL, raw - 3.0)
        self.assertEqual(round(personal.last_predicted, 3), raw)

    # ------------------------------------------------------------------
    def test_reset_service_and_reset_serial_clear_learned_state(self):
        state = self._sync_history(self.SERIAL)
        raw = state["raw_predicted_rul_days"]
        sync_service.learn_outcome(self.SERIAL, raw - 5.0)
        self.assertGreater(
            sync_service._adapter._get(self.SERIAL).n_updates, 0
        )

        self.assertTrue(sync_service.reset_serial(self.SERIAL))
        self.assertNotIn(self.SERIAL, sync_service._adapter.serial_adapters)
        self.assertNotIn(self.SERIAL, sync_service._last_predicted)

        sync_service.reset_service()
        self.assertEqual(sync_service._adapter.serial_adapters, {})
        self.assertEqual(sync_service._adapter.global_adapter.n_updates, 0)
        self.assertEqual(sync_service._last_predicted, {})

    # ------------------------------------------------------------------
    def test_read_only_state_access_does_not_advance_the_learner(self):
        serial = "TEST-SERIAL-READONLY"
        self._sync_history(serial)
        updates_before = sync_service._adapter._get(serial).n_updates

        state = sync_service.get_state(serial)
        self.assertIsNotNone(state)
        self.assertEqual(state["online_adaptation"]["personal_updates"],
                         updates_before)
        self.assertEqual(sync_service._adapter._get(serial).n_updates,
                         updates_before)
        sync_service.reset_serial(serial)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
