"""Tests for the final warning-decision layer (global-warning preservation).

These pin the contract the Task 3 follow-up introduces:

* the global warning is a *completed persistence episode* (streak >= PERSISTENCE
  -> state >= 2), not a severity score,
* the adaptive decision path is a subset of the global one,
* hard preservation is enforced at the decision layer (``max`` of the streaks),
  so a global warning can neither be cancelled nor delayed,
* turning preservation off reproduces the "adaptive-only" decision used as the
  before-baseline in the report,
* repeated observations / repeated reads never advance the streak or fabricate
  warning episodes,
* the *deployed* service (``sync_observation``) preserves the global warning,
  not just an internal helper.

Run with:
    cd 05_backend
    python -m pytest tests/test_warning_decision.py -q
"""

from __future__ import annotations

import datetime
import tempfile
import unittest
from pathlib import Path

from app.ml.warning_decision import (
    PRESERVE_GLOBAL_WARNINGS,
    WARNING_STATE,
    AdaptiveWarningTracker,
    WarningDecision,
    WarningEpisodeTracker,
    combine_warnings,
    escalation_level,
    is_warning,
    persistence,
)
from app.services import twin_sync_service as sync_service


class TestPersistenceBoundaries(unittest.TestCase):
    """The decision rule: a warning needs a completed persistence episode."""

    def setUp(self) -> None:
        self.p = persistence()
        self.assertEqual(self.p, 7)

    def test_escalation_boundaries(self) -> None:
        expected = {
            0: 0, 1: 0, 2: 0, 3: 0, 4: 0,
            5: 1, 6: 1,
            7: 2, 8: 2,
            9: 3, 10: 3, 30: 3,
        }
        for streak, level in expected.items():
            self.assertEqual(escalation_level(streak), level, streak)

    def test_warning_state_boundary(self) -> None:
        self.assertFalse(is_warning(0))
        self.assertFalse(is_warning(1))
        self.assertTrue(is_warning(WARNING_STATE))
        self.assertTrue(is_warning(3))

    def test_streak_just_below_persistence_is_not_a_warning(self) -> None:
        decision = combine_warnings(self.p - 1, 0)
        self.assertEqual(decision.global_state, 1)
        self.assertFalse(is_warning(decision.state))
        self.assertEqual(decision.source, "none")

    def test_streak_at_persistence_is_a_warning(self) -> None:
        decision = combine_warnings(self.p, 0)
        self.assertEqual(decision.global_state, 2)
        self.assertTrue(is_warning(decision.state))
        self.assertEqual(decision.source, "global")


class TestHardPreservation(unittest.TestCase):
    """Global-warning preservation is a decision-layer floor, not a score test."""

    def test_global_warning_survives_a_silent_adaptive_layer(self) -> None:
        decision = combine_warnings(10, 0, preserve_global_warnings=True)
        self.assertEqual(decision.global_state, 3)
        self.assertEqual(decision.adaptive_state, 0)
        self.assertEqual(decision.state, 3)
        self.assertTrue(decision.preserved)
        self.assertFalse(decision.suppressed)
        self.assertEqual(decision.source, "global")

    def test_without_preservation_the_warning_is_suppressed(self) -> None:
        decision = combine_warnings(10, 0, preserve_global_warnings=False)
        self.assertEqual(decision.state, 0)
        self.assertFalse(decision.preserved)
        self.assertTrue(decision.suppressed)
        self.assertEqual(decision.source, "none")

    def test_source_reports_both_paths(self) -> None:
        decision = combine_warnings(8, 8)
        self.assertEqual(decision.source, "global+adaptive")
        self.assertEqual(decision.state, decision.global_state)

    def test_preservation_never_lowers_the_state(self) -> None:
        for global_streak in range(0, 20):
            for adaptive_streak in range(0, 20):
                decision = combine_warnings(global_streak, adaptive_streak)
                # The global decision is a floor; the adaptive path may only
                # add evidence, never remove it.
                self.assertGreaterEqual(
                    decision.state, decision.global_state,
                    (global_streak, adaptive_streak),
                )
                self.assertGreaterEqual(
                    decision.state, escalation_level(adaptive_streak)
                )
                self.assertTrue(decision.preserved)

    def test_production_subset_invariant_makes_preservation_exact(self) -> None:
        # In production the adaptive streak never exceeds the global streak
        # (adaptive severity <= global severity), so preservation reproduces
        # the global decision exactly.
        for adaptive_streak in range(0, 12):
            for extra in range(0, 8):
                global_streak = adaptive_streak + extra
                decision = combine_warnings(global_streak, adaptive_streak)
                self.assertEqual(decision.state, decision.global_state)

    def test_production_default_is_preservation_on(self) -> None:
        self.assertTrue(PRESERVE_GLOBAL_WARNINGS)
        self.assertTrue(combine_warnings(7, 0).preserve_global_warnings)

    def test_decision_is_frozen(self) -> None:
        decision = combine_warnings(7, 0)
        self.assertIsInstance(decision, WarningDecision)
        with self.assertRaises(Exception):
            decision.state = 0  # type: ignore[misc]


class TestAdaptiveWarningTracker(unittest.TestCase):
    """Repeated observations / reads must not move the adaptive streak."""

    def test_streak_requires_consecutive_qualifying_observations(self) -> None:
        tracker = AdaptiveWarningTracker(drive="d", window=30)
        for i, severity in enumerate([0, 1, 1, 1, 1, 1, 1]):
            tracker.observe(severity, observation_key=i)
        self.assertEqual(tracker.streak(), 6)
        self.assertEqual(tracker.state(), 1)
        tracker.observe(1, observation_key=7)
        self.assertEqual(tracker.streak(), 7)
        self.assertEqual(tracker.state(), 2)

    def test_a_non_qualifying_observation_breaks_the_run(self) -> None:
        tracker = AdaptiveWarningTracker(drive="d", window=30)
        for i, severity in enumerate([1] * 7 + [0] + [1] * 7):
            tracker.observe(severity, observation_key=i)
        self.assertEqual(tracker.streak(), 7)

    def test_repeated_observation_key_is_idempotent(self) -> None:
        tracker = AdaptiveWarningTracker(drive="d", window=30)
        self.assertTrue(tracker.observe(1, observation_key="2024-01-01"))
        for _ in range(10):
            self.assertFalse(tracker.observe(1, observation_key="2024-01-01"))
        self.assertEqual(tracker.streak(), 1)
        self.assertTrue(tracker.observe(1, observation_key="2024-01-02"))
        self.assertEqual(tracker.streak(), 2)

    def test_window_caps_the_streak(self) -> None:
        tracker = AdaptiveWarningTracker(drive="d", window=5)
        for i in range(20):
            tracker.observe(1, observation_key=i)
        self.assertEqual(tracker.streak(), 5)

    def test_invalid_severities_are_no_evidence(self) -> None:
        tracker = AdaptiveWarningTracker(drive="d", window=30)
        for i, severity in enumerate([None, "x", float("nan")]):
            tracker.observe(severity, observation_key=i)
        self.assertEqual(tracker.streak(), 0)
        self.assertEqual(tracker.state(), 0)

    def test_reset_clears_the_tracker(self) -> None:
        tracker = AdaptiveWarningTracker(drive="d", window=30)
        for i in range(7):
            tracker.observe(1, observation_key=i)
        self.assertEqual(tracker.streak(), 7)
        tracker.reset()
        self.assertEqual(tracker.streak(), 0)
        self.assertEqual(tracker.snapshot()["observed"], 0)


class TestWarningEpisodeTracker(unittest.TestCase):
    """Warning episodes must be deduplicated, never fabricated."""

    @staticmethod
    def _episodes(states, warning_state=WARNING_STATE):
        tracker = WarningEpisodeTracker(drive="d", path="global", warning_state=warning_state)
        for i, state in enumerate(states):
            tracker.observe(i, timestamp=f"t{i}", state=state)
        tracker.close()
        return tracker.episodes

    def test_sustained_warning_is_one_episode(self) -> None:
        episodes = self._episodes([0, 1, 2, 2, 2, 3, 3, 1])
        self.assertEqual(len(episodes), 1)
        episode = episodes[0]
        self.assertEqual((episode.start_index, episode.end_index), (2, 6))
        self.assertEqual((episode.start, episode.end), ("t2", "t6"))
        self.assertEqual(episode.peak_state, 3)
        self.assertEqual(episode.length, 5)

    def test_two_runs_are_two_episodes(self) -> None:
        episodes = self._episodes([2, 2, 0, 1, 2, 2, 2])
        self.assertEqual(len(episodes), 2)
        self.assertEqual((episodes[0].start_index, episodes[0].end_index), (0, 1))
        self.assertEqual((episodes[1].start_index, episodes[1].end_index), (4, 6))

    def test_no_warning_yields_no_episodes(self) -> None:
        self.assertEqual(self._episodes([0, 1, 1, 0, 1]), [])

    def test_open_warning_is_flushed_on_close(self) -> None:
        episodes = self._episodes([0, 2, 2])
        self.assertEqual(len(episodes), 1)
        self.assertEqual((episodes[0].start_index, episodes[0].end_index), (1, 2))

    def test_repeated_reads_do_not_duplicate_episodes(self) -> None:
        tracker = WarningEpisodeTracker(drive="d", path="global")
        for _ in range(5):
            tracker.observe(0, timestamp="t0", state=2)
            tracker.observe(1, timestamp="t1", state=2)
        tracker.close()
        self.assertEqual(len(tracker.episodes), 1)
        self.assertEqual(tracker.episodes[0].length, 2)


def _observation(day: int, smart_1: float) -> dict:
    date = datetime.date(2024, 1, 1) + datetime.timedelta(days=day)
    return {
        "date": date.isoformat(),
        "smart_1_normalized": float(smart_1),
        "smart_5_normalized": 0.0,
        "smart_187_normalized": 0.0,
        "smart_197_normalized": 1.0,
        "smart_198_normalized": 0.0,
    }


def _norm_from_changes(changes, base: float = 100.0):
    """Build ``smart_1_normalized`` values whose 7-observation diff is ``changes``."""
    norm = []
    for i, change in enumerate(changes):
        if i < 7:
            norm.append(base)
        else:
            shift = 0.0 if change is None else float(change)
            norm.append(norm[i - 7] + shift)
    return norm


# A drive with a noisy (but non-degrading) baseline, then a run of drops that
# the global detector counts as mild degradation but the tightened per-drive
# thresholds do not: the global warning the adaptive layer would suppress.
NOISY_BASELINE = [6.0 if (i % 2 == 0) else 0.0 for i in range(30)]
GLOBAL_RUN = [-1.0] * 10


def _preservation_observations():
    changes = [None] * 7 + NOISY_BASELINE + GLOBAL_RUN
    norm = _norm_from_changes(changes)
    return [_observation(i, v) for i, v in enumerate(norm)]


class TestDeployedRegression(unittest.TestCase):
    """Required regression: the deployed decision preserves a global warning
    that the adaptive layer alone would have suppressed."""

    SERIAL = "TEST-SERIAL-PRESERVE"

    def setUp(self) -> None:
        # Deterministic, hermetic service state; no predictor / calibration IO.
        import tempfile
        from pathlib import Path

        self._tmpdir = tempfile.TemporaryDirectory()
        self._orig_runtime = sync_service._RUNTIME_DIR
        self._orig_state = sync_service._STATE_FILE
        self._orig_legacy = sync_service._LEGACY_HISTORY_FILE
        sync_service._RUNTIME_DIR = Path(self._tmpdir.name)
        sync_service._STATE_FILE = Path(self._tmpdir.name) / "twin_state.json"
        sync_service._LEGACY_HISTORY_FILE = (
            Path(self._tmpdir.name) / "twin_histories.json"
        )
        sync_service._managers.clear()
        sync_service._thresholds.clear()
        sync_service._warning_trackers.clear()
        sync_service._histories.clear()
        sync_service._threshold_states.clear()
        sync_service._tracker_states.clear()
        sync_service._last_observation.clear()
        sync_service._last_predicted.clear()
        sync_service._deleted.clear()
        sync_service._conformal_seeded = True
        sync_service._predictor_error = "disabled in test"
        sync_service._rul_predictor = None

    def tearDown(self) -> None:
        sync_service._warning_trackers.clear()
        sync_service._managers.clear()
        sync_service._thresholds.clear()
        sync_service._histories.clear()
        sync_service._predictor_error = None
        sync_service._RUNTIME_DIR = self._orig_runtime
        sync_service._STATE_FILE = self._orig_state
        sync_service._LEGACY_HISTORY_FILE = self._orig_legacy
        self._tmpdir.cleanup()

    def _replay(self):
        state = None
        for observation in _preservation_observations():
            state = sync_service.sync_observation(self.SERIAL, observation)
        return state

    def test_global_warning_preserved_in_deployed_output(self) -> None:
        state = self._replay()

        decision = state["warning_decision"]
        # The global detector issued a completed warning ...
        self.assertEqual(decision["global_state"], 3)
        self.assertGreaterEqual(decision["global_streak"], 7)
        # ... the adaptive layer alone would have suppressed it ...
        self.assertEqual(decision["adaptive_warning_state"], 0)
        # ... but the deployed decision preserved it.
        self.assertTrue(decision["preserved"])
        self.assertFalse(decision["suppressed"])
        self.assertTrue(decision["preserve_global_warnings"])
        self.assertEqual(decision["source"], "global")
        self.assertEqual(state["adaptive_state"], decision["global_state"])
        self.assertEqual(state["adaptive_state"], 3)

    def test_adaptive_only_would_have_suppressed_it(self) -> None:
        state = self._replay()
        decision = state["warning_decision"]
        dropped = combine_warnings(
            decision["global_streak"],
            decision["adaptive_streak"],
            preserve_global_warnings=False,
        )
        # This is the counterfactual the safety requirement removes: without
        # preservation the completed global warning disappears entirely.
        self.assertEqual(dropped.state, 0)
        self.assertTrue(dropped.suppressed)
        self.assertGreater(decision["state"], dropped.state)

    def test_read_only_state_access_does_not_advance_the_decision(self) -> None:
        first = self._replay()
        for _ in range(5):
            again = sync_service.get_state(self.SERIAL)
            self.assertEqual(
                again["warning_decision"], first["warning_decision"]
            )
            self.assertEqual(again["adaptive_state"], first["adaptive_state"])


class TestDeployedApiPreservation(unittest.TestCase):
    """Required test 9: the *API* warning outputs preserve global warnings."""

    SERIAL = "TEST-SERIAL-PRESERVE-API"

    @classmethod
    def setUpClass(cls) -> None:
        from fastapi.testclient import TestClient

        from app.main import app

        # Isolate this API test's persistence in a temp store so the real
        # project runtime directory is never written to (deletes now persist a
        # tombstone that must survive a restart).
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls._orig_runtime = sync_service._RUNTIME_DIR
        cls._orig_state = sync_service._STATE_FILE
        cls._orig_legacy = sync_service._LEGACY_HISTORY_FILE
        sync_service._RUNTIME_DIR = Path(cls._tmpdir.name)
        sync_service._STATE_FILE = Path(cls._tmpdir.name) / "twin_state.json"
        sync_service._LEGACY_HISTORY_FILE = (
            Path(cls._tmpdir.name) / "twin_histories.json"
        )
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls) -> None:
        sync_service._RUNTIME_DIR = cls._orig_runtime
        sync_service._STATE_FILE = cls._orig_state
        sync_service._LEGACY_HISTORY_FILE = cls._orig_legacy
        cls._tmpdir.cleanup()

    def setUp(self) -> None:
        sync_service._managers.clear()
        sync_service._thresholds.clear()
        sync_service._warning_trackers.clear()
        sync_service._histories.clear()
        sync_service._conformal_seeded = True
        sync_service._predictor_error = "disabled in test"
        sync_service._rul_predictor = None

    def tearDown(self) -> None:
        self.client.delete(f"/api/v1/component2/twin/{self.SERIAL}")
        sync_service._warning_trackers.clear()
        sync_service._managers.clear()
        sync_service._thresholds.clear()
        sync_service._histories.clear()
        sync_service._predictor_error = None

    def test_sync_endpoint_preserves_the_global_warning(self) -> None:
        state = None
        for observation in _preservation_observations():
            response = self.client.post(
                "/api/v1/component2/twin/sync",
                json={"serial_number": self.SERIAL, "observation": observation},
            )
            self.assertEqual(response.status_code, 200)
            state = response.json()

        decision = state["warning_decision"]
        self.assertEqual(decision["global_state"], 3)
        self.assertEqual(decision["adaptive_warning_state"], 0)
        self.assertTrue(decision["preserve_global_warnings"])
        self.assertTrue(decision["preserved"])
        # The API-level warning level is the preserved (global) decision ...
        self.assertEqual(state["adaptive_state"], 3)
        self.assertEqual(state["adaptive_state_name"], "Strong Persistent Degradation")

        # ... and reading it back over HTTP does not change the decision.
        reread = self.client.get(
            f"/api/v1/component2/twin/{self.SERIAL}/state"
        ).json()
        self.assertEqual(reread["warning_decision"], decision)
        self.assertEqual(reread["adaptive_state"], 3)


if __name__ == "__main__":
    unittest.main()
