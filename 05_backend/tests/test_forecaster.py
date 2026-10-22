"""Tests for the repaired multi-horizon health forecaster (Feature 4).

These pin the Task 4 repair: a forecast must be based on defensible evidence
and real time semantics, never on "time passed, so the streak grew".

* a healthy drive with no current degradation evidence cannot be forecast to
  escalate within seven days,
* the continuation scenario only fires when there is a current trend *and* the
  observation cadence supports converting a day horizon into observations,
* current state, neutral scenario, continuation scenario, unavailable forecast
  and the (definitional) RUL countdown are all distinguishable,
* irregular / missing cadence is never silently treated as daily,
* intervals declare that they are derived from the base band, not
  horizon-calibrated,
* repeated calls on unchanged state are deterministic,
* invalid / non-finite inputs are handled safely,
* the deployed API response keeps its backwards-compatible keys.

Run with:
    cd 05_backend
    python -m pytest tests/test_forecaster.py -q
"""

from __future__ import annotations

import unittest

from app.ml.forecaster import (
    CADENCE_DAILY,
    CADENCE_INSUFFICIENT,
    CADENCE_IRREGULAR,
    DEFAULT_HORIZONS,
    EVIDENCE_NO_TREND,
    EVIDENCE_SUPPORTED,
    EVIDENCE_UNSUPPORTED,
    HORIZON_UNCERTAINTY_GROWTH,
    forecast,
    escalation_level,
)

# A daily cadence over a long history (~1 observation per day).
DAILY_GAPS = [1.0] * 29
# A weekly stream: observations are seven calendar days apart.
WEEKLY_GAPS = [7.0] * 29


def _entry(fc, day):
    for row in fc["trajectory"]:
        if row["day"] == day:
            return row
    raise AssertionError(f"horizon {day} missing")


class TestHealthyDriveDoesNotEscalate(unittest.TestCase):
    """Requirement 1: no evidence must not become escalation by passage of time."""

    def test_zero_streak_daily_does_not_forecast_escalation(self) -> None:
        fc = forecast(
            predicted_rul=40.0,
            current_state=0,
            current_streak=0,
            persistence=7,
            base_half_width=5.0,
            observation_gaps_days=DAILY_GAPS,
        )
        self.assertIsNone(fc["first_escalation"])
        for row in fc["trajectory"]:
            self.assertFalse(row["escalates_from_current"])
            self.assertEqual(row["projected_state"], 0)
            # The trend does not exist, so there is nothing to continue:
            # the horizon is still resolvable (daily cadence), but the
            # continuation scenario carries no escalation.
            self.assertEqual(
                row["availability"]["evidence_status"], EVIDENCE_NO_TREND
            )
            self.assertTrue(row["scenarios"]["continuation"]["available"])
            self.assertEqual(row["scenarios"]["continuation"]["state"], 0)
            # No scenario invents escalation beyond the (zero) current state.
            self.assertEqual(row["escalation_ceiling"], 0)

    def test_escalation_requires_a_current_trend(self) -> None:
        # Same cadence, but now there *is* a trend.
        fc = forecast(
            predicted_rul=40.0,
            current_state=1,
            current_streak=5,
            persistence=7,
            base_half_width=5.0,
            observation_gaps_days=DAILY_GAPS,
        )
        self.assertIsNotNone(fc["first_escalation"])
        # From streak 5, +2 observations reaches the persistence requirement (7).
        self.assertEqual(fc["first_escalation"]["day"], 7)


class TestContinuationIsBoundedByCadence(unittest.TestCase):
    """Requirement 2: continuation follows the documented, cadence-aware logic."""

    def test_weekly_stream_uses_observation_counts_not_days(self) -> None:
        fc = forecast(
            predicted_rul=60.0,
            current_state=1,
            current_streak=5,
            persistence=7,
            base_half_width=5.0,
            observation_gaps_days=WEEKLY_GAPS,
        )
        # 7 days of a weekly stream is ~1 observation, not 7.
        row = _entry(fc, 7)
        self.assertEqual(row["scenarios"]["continuation"]["streak"], 6)
        self.assertEqual(row["scenarios"]["continuation"]["state"], 1)
        self.assertFalse(row["escalates_from_current"])
        # It cannot be a warning: only one new qualifying observation happened.
        self.assertNotEqual(row["escalation_ceiling"], 2)
    def test_daily_stream_continuation_grows_one_per_day(self) -> None:
        fc = forecast(
            predicted_rul=60.0,
            current_state=1,
            current_streak=5,
            persistence=7,
            base_half_width=5.0,
            observation_gaps_days=DAILY_GAPS,
        )
        row = _entry(fc, 7)
        self.assertEqual(row["scenarios"]["continuation"]["streak"], 12)
        self.assertEqual(row["scenarios"]["continuation"]["state"], 3)
        self.assertEqual(row["availability"]["evidence_status"], EVIDENCE_SUPPORTED)


class TestStateAndScenarioSeparation(unittest.TestCase):
    """Requirement 3: current state, scenarios and unavailability are distinct."""

    def test_all_branches_reported_separately(self) -> None:
        fc = forecast(
            predicted_rul=30.0,
            current_state=1,
            current_streak=5,
            persistence=7,
            base_half_width=5.0,
            observation_gaps_days=DAILY_GAPS,
        )
        row = _entry(fc, 30)
        sc = row["scenarios"]
        # Current = neutral (no new evidence).
        self.assertEqual(sc["current"]["state"], 1)
        self.assertEqual(sc["neutral"]["state"], 1)
        self.assertEqual(sc["neutral"]["streak"], 5)
        # Adversarial lower bound.
        self.assertEqual(sc["lower_bound"]["state"], 0)
        self.assertEqual(sc["lower_bound"]["streak"], 0)
        # Continuation is the optimistic branch.
        self.assertEqual(sc["continuation"]["state"], 3)
        # And they are labelled.
        self.assertIn("assumption", sc["neutral"])
        self.assertIn("assumption", sc["continuation"])

    def test_unavailable_continuation_is_explicit(self) -> None:
        # No cadence evidence at all -> continuation cannot be resolved.
        fc = forecast(
            predicted_rul=30.0,
            current_state=1,
            current_streak=5,
            persistence=7,
            base_half_width=5.0,
        )
        row = _entry(fc, 7)
        self.assertIsNone(row["projected_streak"])
        self.assertIsNone(row["projected_state"])
        self.assertIsNone(row["projected_state_name"])
        self.assertIsNone(row["scenarios"]["continuation"]["state"])
        self.assertFalse(row["scenarios"]["continuation"]["available"])
        self.assertEqual(
            row["availability"]["evidence_status"], EVIDENCE_UNSUPPORTED
        )
        self.assertFalse(row["escalates_from_current"])

    def test_rul_countdown_is_definitional_not_escalation(self) -> None:
        fc = forecast(
            predicted_rul=20.0,
            current_state=0,
            current_streak=0,
            persistence=7,
            base_half_width=2.0,
            observation_gaps_days=DAILY_GAPS,
        )
        # RUL still counts down deterministically...
        self.assertEqual(_entry(fc, 7)["projected_rul_days"], 13.0)
        # ...but that must never become a state escalation claim.
        self.assertIsNone(fc["first_escalation"])
        self.assertEqual(fc["escalation_forecast_status"], "conditional")
        self.assertIn("definitional", " ".join(fc["assumptions"]))


class TestCadenceValidation(unittest.TestCase):
    """Requirement 4: irregular cadence must not be read as daily sampling."""

    def test_irregular_cadence_reported(self) -> None:
        fc = forecast(
            predicted_rul=30.0,
            current_state=1,
            current_streak=5,
            observation_gaps_days=[1.0, 20.0, 1.0, 30.0],
        )
        self.assertEqual(fc["cadence"]["status"], CADENCE_IRREGULAR)
        self.assertIsNotNone(fc["cadence"]["median_gap_days"])
        self.assertGreater(fc["cadence"]["median_gap_days"], 1.5)

    def test_irregular_cadence_is_not_flagged_as_daily(self) -> None:
        # A coarse (e.g. monthly) stream must not be treated as daily: the
        # continuation is converted cadence-aware and cannot escalate on the
        # strength of 7 days.
        fc = forecast(
            predicted_rul=60.0,
            current_state=1,
            current_streak=5,
            persistence=7,
            observation_gaps_days=[30.0] * 10,
        )
        row = _entry(fc, 7)
        self.assertFalse(row["availability"]["cadence_supported"])
        self.assertEqual(row["scenarios"]["continuation"]["streak"], 5)
        self.assertFalse(row["escalates_from_current"])

    def test_fractional_cadence_is_counted(self) -> None:
        # 0.5 days per observation -> two observations per day.
        fc = forecast(
            predicted_rul=60.0,
            current_state=1,
            current_streak=5,
            observation_gaps_days=[0.5] * 20,
        )
        self.assertEqual(fc["cadence"]["status"], CADENCE_DAILY)
        row = _entry(fc, 7)
        self.assertEqual(row["scenarios"]["continuation"]["streak"], 19)

    def test_weekly_gap_is_not_daily(self) -> None:
        fc = forecast(
            predicted_rul=60.0,
            current_state=1,
            current_streak=5,
            observation_gaps_days=[7.0] * 5,
        )
        self.assertEqual(fc["cadence"]["status"], CADENCE_IRREGULAR)


class TestInsufficientEvidence(unittest.TestCase):
    """Requirement 5: missing timestamps / short history handled safely."""

    def test_no_cadence_evidence_is_unsupported(self) -> None:
        fc = forecast(predicted_rul=30.0, current_state=1, current_streak=5)
        self.assertEqual(fc["cadence"]["status"], CADENCE_INSUFFICIENT)
        self.assertIsNone(fc["cadence"]["median_gap_days"])
        self.assertIsNone(fc["first_escalation"])
        for row in fc["trajectory"]:
            self.assertFalse(row["availability"]["cadence_supported"])
            self.assertEqual(
                row["availability"]["evidence_status"], EVIDENCE_UNSUPPORTED
            )

    def test_brand_new_drive_has_no_forecast(self) -> None:
        fc = forecast(predicted_rul=None, current_state=0, current_streak=0)
        self.assertEqual(fc["current_state"], 0)
        self.assertIsNone(fc["predicted_rul_now"])
        self.assertIsNone(fc["failure_eta_day"])
        self.assertIsNone(fc["first_escalation"])
        for row in fc["trajectory"]:
            self.assertIsNone(row["projected_rul_days"])
            self.assertIsNone(row["rul_low"])
            self.assertIsNone(row["rul_high"])

    def test_short_history_is_not_daily(self) -> None:
        fc = forecast(
            predicted_rul=30.0,
            current_state=1,
            current_streak=5,
            observation_gaps_days=[1.0],  # one gap is below the minimum
        )
        self.assertEqual(fc["cadence"]["status"], CADENCE_INSUFFICIENT)

    def test_horizon_beyond_data_range(self) -> None:
        # Horizon beyond MAX_HORIZON is dropped, empty fallback to defaults.
        self.assertEqual(
            forecast(20.0, horizons=[1, 5, 500])["horizons_days"], [1, 5]
        )
        self.assertEqual(
            forecast(20.0, horizons=[]).__getitem__("horizons_days"),
            list(DEFAULT_HORIZONS),
        )


class TestInvalidInputs(unittest.TestCase):
    """Requirement 9: invalid / non-finite predictions handled safely."""

    def test_nan_rul_is_treated_as_missing(self) -> None:
        fc = forecast(predicted_rul=float("nan"))
        self.assertIsNone(fc["predicted_rul_now"])
        self.assertIsNone(fc["trajectory"][0]["projected_rul_days"])

    def test_negative_rul_is_clamped(self) -> None:
        fc = forecast(predicted_rul=-5.0)
        self.assertEqual(fc["predicted_rul_now"], 0.0)

    def test_invalid_horizons_are_dropped(self) -> None:
        fc = forecast(20.0, horizons=[0, -3, 200, 7])
        self.assertEqual(fc["horizons_days"], [7])

    def test_small_persistence_does_not_crash(self) -> None:
        fc = forecast(20.0, current_streak=3, persistence=1)
        # Escalation ladder is delegated to the detector and stays in 0..3.
        for row in fc["trajectory"]:
            for key in ("projected_state", "escalation_floor", "escalation_ceiling"):
                level = row[key]
                if level is not None:
                    self.assertTrue(0 <= level <= 3, (key, level))


class TestIntervalConstruction(unittest.TestCase):
    """Requirement 7: intervals must follow documented construction."""

    def test_interval_is_derived_from_base(self) -> None:
        fc = forecast(20.0, observation_gaps_days=DAILY_GAPS, base_half_width=4.0)
        self.assertFalse(fc["interval_calibration"]["horizon_specific"])
        self.assertEqual(fc["interval_calibration"]["type"], "derived_from_base")

    def test_interval_matches_documented_formula(self) -> None:
        base = 4.0
        # Large RUL so no lower bound is clipped at zero.
        fc = forecast(200.0, observation_gaps_days=DAILY_GAPS, base_half_width=base)
        for day in DEFAULT_HORIZONS:
            row = _entry(fc, day)
            spread = base * (1.0 + HORIZON_UNCERTAINTY_GROWTH * day)
            self.assertAlmostEqual(row["rul_high"], (200.0 - day) + spread, places=2)
            self.assertAlmostEqual(row["rul_low"], (200.0 - day) - spread, places=2)

    def test_interval_widens_with_horizon(self) -> None:
        fc = forecast(200.0, observation_gaps_days=DAILY_GAPS, base_half_width=4.0)
        widths = [
            _entry(fc, d)["rul_high"] - _entry(fc, d)["rul_low"]
            for d in DEFAULT_HORIZONS
        ]
        self.assertEqual(widths, sorted(widths))

    def test_interval_lower_bound_is_clipped(self) -> None:
        fc = forecast(2.0, observation_gaps_days=DAILY_GAPS, base_half_width=10.0)
        row = _entry(fc, 30)
        self.assertEqual(row["projected_rul_days"], 0.0)
        self.assertEqual(row["rul_low"], 0.0)


class TestDeterminism(unittest.TestCase):
    """Requirement 8: repeated calls on unchanged state are deterministic."""

    def test_repeat_calls_match(self) -> None:
        kwargs = dict(
            predicted_rul=17.5,
            current_state=1,
            current_streak=4,
            persistence=7,
            base_half_width=6.0,
            observation_gaps_days=DAILY_GAPS,
        )
        self.assertEqual(forecast(**kwargs), forecast(**kwargs))

    def test_identical_inputs_same_result(self) -> None:
        a = forecast(10.0, current_streak=3, observation_gaps_days=WEEKLY_GAPS)
        b = forecast(10.0, current_streak=3, observation_gaps_days=WEEKLY_GAPS)
        self.assertEqual(a, b)


class TestEscalationLadder(unittest.TestCase):
    """The ladder must remain identical to the production detector."""

    def test_ladder_matches_detector(self) -> None:
        from app.ml._component2_paths import get_detector

        det = get_detector()
        for p in range(3, 13):
            for streak in range(0, 45):
                self.assertEqual(
                    escalation_level(streak, p),
                    det.escalation_level(streak, p),
                    (streak, p),
                )


class TestDeployedApiCompatibility(unittest.TestCase):
    """Requirement 10: deployed API response keeps its compatible keys."""

    def setUp(self) -> None:
        import tempfile

        from fastapi.testclient import TestClient

        from app.main import app

        self._sync = __import__(
            "app.services.twin_sync_service", fromlist=["reset_service"]
        )
        # Isolate persistence in a temp store (the service now persists the full
        # twin state, including deletions, so tests must not write project data).
        self._tmpdir = tempfile.TemporaryDirectory()
        from pathlib import Path

        self._orig_runtime = self._sync._RUNTIME_DIR
        self._orig_state = self._sync._STATE_FILE
        self._orig_legacy = self._sync._LEGACY_HISTORY_FILE
        self._sync._RUNTIME_DIR = Path(self._tmpdir.name)
        self._sync._STATE_FILE = Path(self._tmpdir.name) / "twin_state.json"
        self._sync._LEGACY_HISTORY_FILE = (
            Path(self._tmpdir.name) / "twin_histories.json"
        )
        self._sync.reset_service()
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self._sync.reset_service()
        self._sync._RUNTIME_DIR = self._orig_runtime
        self._sync._STATE_FILE = self._orig_state
        self._sync._LEGACY_HISTORY_FILE = self._orig_legacy
        self._tmpdir.cleanup()

    def _observation(self, day: int):
        return {
            "date": f"2024-01-{day:02d}",
            "smart_1_normalized": 0.2 + day * 0.001,
            "smart_5_normalized": 0.1,
            "smart_187_normalized": 0.3 + day * 0.001,
            "smart_197_normalized": 0.2,
            "smart_198_normalized": 0.1,
        }

    def test_sync_forecast_payload_keys(self) -> None:
        serial = "FORECAST-API-1"
        resp = self.client.post(
            "/api/v1/component2/twin/sync",
            json={"serial_number": serial, "observation": self._observation(1)},
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("forecast", body)
        fc = body["forecast"]

        # ---- Backwards-compatible keys the frontend reads ----
        for key in (
            "horizons_days",
            "current_state",
            "current_state_name",
            "current_streak",
            "persistence_used",
            "predicted_rul_now",
            "failure_eta_day",
            "first_escalation",
            "trajectory",
            "assumptions",
        ):
            self.assertIn(key, fc, key)
        self.assertEqual(fc["horizons_days"], list(DEFAULT_HORIZONS))
        for row in fc["trajectory"]:
            for key in (
                "day",
                "projected_rul_days",
                "rul_low",
                "rul_high",
                "projected_streak",
                "projected_state",
                "projected_state_name",
                "escalates_from_current",
            ):
                self.assertIn(key, row, key)

    def test_single_observation_drive_does_not_escalate(self) -> None:
        serial = "FORECAST-API-2"
        body = self.client.post(
            "/api/v1/component2/twin/sync",
            json={"serial_number": serial, "observation": self._observation(1)},
        ).json()
        fc = body["forecast"]
        # One observation is not a cadence and carries no trend.
        self.assertEqual(fc["cadence"]["status"], CADENCE_INSUFFICIENT)
        self.assertIsNone(fc["first_escalation"])
        for row in fc["trajectory"]:
            self.assertFalse(row["escalates_from_current"])

    def test_state_endpoint_forecast_is_deterministic(self) -> None:
        serial = "FORECAST-API-3"
        for day in range(1, 5):
            self.client.post(
                "/api/v1/component2/twin/sync",
                json={
                    "serial_number": serial,
                    "observation": self._observation(day),
                },
            )
        first = self.client.get(f"/api/v1/component2/twin/{serial}/state").json()
        second = self.client.get(f"/api/v1/component2/twin/{serial}/state").json()
        self.assertEqual(first["forecast"], second["forecast"])

    def test_service_wires_real_observation_cadence(self) -> None:
        """Regression: the service must pass measured gaps, not empty ones.

        A weekly ingestion must be reported as ``irregular`` with a median gap
        of 7 days.  (A previous wiring silently fell back to "insufficient"
        cadence and turned a weekly stream into an unsupported forecast.)
        """
        serial = "FORECAST-API-WEEKLY"
        for i in range(6):
            self.client.post(
                "/api/v1/component2/twin/sync",
                json={
                    "serial_number": serial,
                    "observation": {
                        "date": f"2024-01-{1 + i * 7:02d}",
                        "smart_1_normalized": 0.2,
                        "smart_5_normalized": 0.1,
                        "smart_187_normalized": 0.3,
                        "smart_197_normalized": 0.2,
                        "smart_198_normalized": 0.1,
                    },
                },
            )
        fc = self.client.get(
            f"/api/v1/component2/twin/{serial}/state"
        ).json()["forecast"]
        self.assertEqual(fc["cadence"]["status"], CADENCE_IRREGULAR)
        self.assertEqual(fc["cadence"]["median_gap_days"], 7.0)
        self.assertIsNone(fc["first_escalation"])

    def test_service_daily_cadence_is_supported(self) -> None:
        serial = "FORECAST-API-DAILY"
        for day in range(1, 12):
            self.client.post(
                "/api/v1/component2/twin/sync",
                json={
                    "serial_number": serial,
                    "observation": {
                        "date": f"2024-02-{day:02d}",
                        "smart_1_normalized": 0.2 + day * 0.01,
                        "smart_5_normalized": 0.1,
                        "smart_187_normalized": 0.3 + day * 0.01,
                        "smart_197_normalized": 0.2,
                        "smart_198_normalized": 0.1,
                    },
                },
            )
        fc = self.client.get(
            f"/api/v1/component2/twin/{serial}/state"
        ).json()["forecast"]
        self.assertEqual(fc["cadence"]["status"], CADENCE_DAILY)
        self.assertEqual(fc["cadence"]["median_gap_days"], 1.0)


if __name__ == "__main__":
    unittest.main()
