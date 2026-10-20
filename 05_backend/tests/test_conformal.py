"""Tests for the conformal calibration layer (Component 2, Feature 2).

The tests pin the properties the deployed intervals actually rely on:

* the calibration drives are disjoint from the final evaluation drives and a
  deliberate overlap fails loudly,
* the finite-sample quantile is the ``ceil((n+1)(1-alpha))``-th smallest
  residual (hand-calculated examples), not an interpolated value,
* invalid / insufficient calibration data is handled by documented policy,
* interval bounds are ordered and respect the RUL domain without silently
  invalidating coverage,
* the quality gate only widens,
* calibration is reproducible from the repository state,
* calibrating on the evaluation set is what makes a biased predictor look
  calibrated,
* the deployed endpoint reports the true calibration source and sample count.

Run with:
    cd 05_backend
    python -m pytest tests/test_conformal.py -q
"""

from __future__ import annotations

import inspect
import math
import unittest

import numpy as np

from app.ml import calibration_data
from app.ml.calibration_data import (
    CalibrationLeakageError,
    CalibrationSourceError,
)
from app.ml.conformal import (
    FALLBACK_MIN_HALF_WIDTH,
    FALLBACK_RELATIVE_HALF_WIDTH,
    MIN_CALIBRATION_SAMPLES,
    QUALITY_WIDENING,
    ConformalCalibrator,
)

_PREDICTOR = None


def _predictor():
    """The frozen BiLSTM, loaded once for the whole module."""
    global _PREDICTOR
    if _PREDICTOR is None:
        from app.ml.component2_rul_predictor import Component2RULPredictor

        _PREDICTOR = Component2RULPredictor()
    return _PREDICTOR


class TestCalibrationSplitDisjointness(unittest.TestCase):
    """Required tests 1 and 2: disjointness, and a loud failure on overlap."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.data = calibration_data.build_calibration_data(
            predictor=_predictor()
        )
        cls.diagnostics = cls.data.diagnostics

    # -- 1 --------------------------------------------------------------
    def test_calibration_and_evaluation_serials_are_disjoint(self) -> None:
        metadata = calibration_data.load_split_metadata()
        drives = calibration_data.split_drives(metadata)
        calibration = drives[self.diagnostics["calibration_split"]]
        evaluation = drives[self.diagnostics["evaluation_split"]]

        self.assertEqual(calibration & evaluation, set())
        self.assertEqual(self.diagnostics["drive_overlap"], 0)
        self.assertEqual(self.diagnostics["overlap_check"], "pass")
        # the guard itself must accept the real population
        calibration_data.verify_disjoint(
            sorted(calibration), sorted(evaluation)
        )

    def test_calibration_population_matches_the_documented_split(self) -> None:
        self.assertEqual(self.diagnostics["sequence_length"], 30)
        self.assertEqual(
            self.diagnostics["sequence_lengths_in_population"], [30]
        )
        self.assertEqual(self.diagnostics["calibration_split"], "validation")
        self.assertEqual(self.diagnostics["evaluation_split"], "test")
        self.assertEqual(
            self.diagnostics["checkpoint_file"], "best_bilstm_model.pt"
        )
        self.assertEqual(len(self.diagnostics["checkpoint_sha256"]), 16)
        self.assertEqual(self.data.n_drives, 89)
        self.assertEqual(self.data.n_observations, 2922)
        self.assertEqual(len(set(self.data.serials)), 89)
        # the calibration split was never fed to the model during training
        self.assertEqual(self.diagnostics["drives_also_in_training_split"], 0)

    def test_calibration_uses_the_deployed_checkpoint_and_preprocessing(
        self,
    ) -> None:
        """Calibration is anchored to the same artefacts production loads."""
        from app.ml.component2_rul_predictor import (
            MODEL_PATH,
            PREPROCESSING_PATH,
        )

        self.assertEqual(calibration_data.checkpoint_path(), MODEL_PATH)
        self.assertEqual(
            calibration_data.preprocessing_path(), PREPROCESSING_PATH
        )
        self.assertEqual(
            calibration_data.preprocessed_dir(), PREPROCESSING_PATH.parent
        )
        # the loader resolves those artefacts through the predictor module
        self.assertIn(
            "component2_rul_predictor", inspect.getsource(calibration_data)
        )

    def test_historical_trajectory_seed_is_no_longer_used(self) -> None:
        """Behavioural pin: the old seed was 303 rows / 90 test drives."""
        self.assertNotEqual(self.data.n_observations, 303)
        self.assertNotEqual(self.data.n_drives, 90)
        self.assertEqual(self.diagnostics["calibration_split"], "validation")
        self.assertEqual(
            self.diagnostics["sequence_lengths_in_population"], [30]
        )
        service = __import__(
            "app.services.twin_sync_service", fromlist=["x"]
        )
        self.assertIn(
            "calibration_data.load_calibration",
            inspect.getsource(service._seed_calibration),
        )
        self.assertNotIn(
            "component2_results_dir", inspect.getsource(service._seed_calibration)
        )

    # -- 2 --------------------------------------------------------------
    def test_deliberate_overlap_fails_loudly(self) -> None:
        with self.assertRaises(CalibrationLeakageError) as ctx:
            calibration_data.verify_disjoint(
                ["DRIVE-A", "DRIVE-B", "DRIVE-C"],
                ["DRIVE-C", "DRIVE-D"],
            )
        message = str(ctx.exception)
        self.assertIn("DRIVE-C", message)
        self.assertIn("calibration", message)

    def test_asking_for_an_overlapping_split_fails_loudly(self) -> None:
        """Calibrating on the evaluation split must be refused, not warned."""
        with self.assertRaises(CalibrationLeakageError):
            calibration_data.build_calibration_data(
                calibration_split="test", evaluation_split="test"
            )

    def test_unknown_split_is_rejected(self) -> None:
        with self.assertRaises(CalibrationSourceError):
            calibration_data.build_calibration_data(
                calibration_split="not-a-split"
            )


class TestSafeFallback(unittest.TestCase):
    """Calibration failure degrades to the documented band, never silently."""

    def test_service_leaves_the_calibrator_empty_when_loading_fails(self) -> None:
        from app.services import twin_sync_service as sync_service

        original = calibration_data.load_calibration

        def _fail(*_args, **_kwargs):
            raise CalibrationLeakageError("synthetic overlap")

        sync_service.reset_service()
        calibration_data.load_calibration = _fail
        try:
            snapshot = sync_service.calibration_snapshot()
        finally:
            calibration_data.load_calibration = original
            sync_service.reset_service()

        conformal = snapshot["conformal"]
        self.assertEqual(conformal["calibration_samples"], 0)
        self.assertFalse(conformal["calibration_sufficient"])
        self.assertIsNone(conformal["conformity_quantile"])
        self.assertIn("CalibrationLeakageError", snapshot["calibration_error"])

        interval = sync_service._conformal.interval(30.0, quality=1.0)
        self.assertFalse(interval["calibrated"])
        self.assertTrue(interval["uncalibrated_fallback"])
        self.assertGreater(interval["base_half_width"], 0.0)


class TestFiniteSampleQuantile(unittest.TestCase):
    """Required tests 3 and 5: hand-calculated quantiles, insufficiency."""

    def test_quantile_matches_hand_calculated_conformal_examples(self) -> None:
        # n = 20, alpha = 0.1  ->  k = ceil(21 * 0.9) = 19 -> 19th smallest
        calibrator = ConformalCalibrator(alpha=0.1)
        calibrator.seed([float(i) for i in range(1, 21)])
        self.assertEqual(calibrator.n_calibration, 20)
        self.assertEqual(calibrator.quantile_index(), 19)
        self.assertEqual(calibrator.conformity_quantile(), 19.0)

        # n = 10, alpha = 0.1  ->  k = ceil(11 * 0.9) = 10 -> the maximum
        calibrator = ConformalCalibrator(alpha=0.1)
        calibrator.seed([float(i) for i in range(1, 11)])
        self.assertEqual(calibrator.quantile_index(), 10)
        self.assertEqual(calibrator.conformity_quantile(), 10.0)

        # alpha = 0.5, n = 20 -> k = ceil(21 * 0.5) = 11
        calibrator = ConformalCalibrator(alpha=0.5)
        calibrator.seed([float(i) for i in range(1, 21)])
        self.assertEqual(calibrator.quantile_index(), 11)
        self.assertEqual(calibrator.conformity_quantile(), 11.0)

        # fewer than MIN_CALIBRATION_SAMPLES -> no coverage claim
        calibrator = ConformalCalibrator(alpha=0.2)
        calibrator.seed([1.0, 2.0, 3.0, 4.0])
        self.assertIsNone(calibrator.conformity_quantile())

    def test_quantile_is_an_observed_order_statistic_not_an_interpolation(
        self,
    ) -> None:
        # 19 zeros then a 100: k = 19 -> the 19th smallest is 0.0.
        residuals = [0.0] * 19 + [100.0]
        calibrator = ConformalCalibrator(alpha=0.1)
        calibrator.seed(residuals)
        self.assertEqual(calibrator.conformity_quantile(), 0.0)

        # What the previous implementation returned: np.quantile at
        # level = ceil((n+1)(1-alpha))/n, which linearly interpolates between
        # two order statistics and returns a value never observed.
        level = min(math.ceil(21 * 0.9) / 20, 1.0)
        interpolated = float(np.quantile(residuals, level))
        self.assertAlmostEqual(interpolated, 5.0, places=9)
        self.assertNotEqual(calibrator.conformity_quantile(), interpolated)

    def test_quantile_index_is_ceil_n_plus_one(self) -> None:
        for n in (10, 11, 20, 33, 100, 500, 2922):
            calibrator = ConformalCalibrator(alpha=0.1)
            calibrator.seed([1.0] * n)
            self.assertEqual(
                calibrator.quantile_index(), math.ceil((n + 1) * 0.9)
            )

    def test_insufficient_samples_do_not_claim_coverage(self) -> None:
        calibrator = ConformalCalibrator(alpha=0.1)
        calibrator.seed([1.0] * (MIN_CALIBRATION_SAMPLES - 1))
        self.assertIsNone(calibrator.conformity_quantile())
        info = calibrator.quantile_info()
        self.assertFalse(info["calibration_sufficient"])
        self.assertIn("need >=", info["insufficient_reason"])

    def test_impossible_coverage_is_reported_instead_of_clamped(self) -> None:
        """alpha=0.05 with n=10 needs k=11 > n: no finite guarantee exists."""
        calibrator = ConformalCalibrator(alpha=0.05)
        calibrator.seed([float(i) for i in range(1, 11)])
        self.assertIsNone(calibrator.conformity_quantile())
        self.assertIsNone(calibrator.quantile_index())
        info = calibrator.quantile_info()
        self.assertFalse(info["calibration_sufficient"])
        self.assertIn("k=11", info["insufficient_reason"])

    def test_alpha_must_be_a_probability(self) -> None:
        for bad in (0.0, 1.0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                ConformalCalibrator(alpha=bad)


class TestInvalidResiduals(unittest.TestCase):
    """Required test 4: invalid residuals are rejected by documented policy."""

    def test_non_finite_and_non_numeric_residuals_are_rejected(self) -> None:
        calibrator = ConformalCalibrator(alpha=0.1)
        accepted = [
            calibrator.add_residual(10.0, 12.0),
            calibrator.add_residual(10.0, 8.0),
        ]
        rejected = [
            calibrator.add_residual(float("nan"), 5.0),
            calibrator.add_residual(5.0, float("nan")),
            calibrator.add_residual(float("inf"), 5.0),
            calibrator.add_residual(5.0, float("-inf")),
            calibrator.add_residual(None, 5.0),
            calibrator.add_residual(5.0, None),
            calibrator.add_residual("not-a-number", 5.0),
            calibrator.add_residual(5.0, object()),
        ]

        self.assertEqual(accepted, [True, True])
        self.assertEqual(rejected, [False] * 8)
        self.assertEqual(calibrator.n_rejected, 8)
        self.assertEqual(calibrator.n_calibration, 2)
        self.assertEqual(calibrator.residuals, [2.0, 2.0])

    def test_seed_rejects_invalid_entries_and_reports_the_count(self) -> None:
        calibrator = ConformalCalibrator(alpha=0.1)
        accepted = calibrator.seed(
            [1.0, None, float("nan"), float("inf"), 3.0, "x", -4.0]
        )
        self.assertEqual(accepted, 3)  # 1.0, 3.0 and |−4.0|
        self.assertEqual(calibrator.n_rejected, 4)
        self.assertEqual(calibrator.residuals, [1.0, 3.0, 4.0])

    def test_pairs_seeding_reports_accepted_count(self) -> None:
        calibrator = ConformalCalibrator(alpha=0.1)
        accepted = calibrator.seed_from_pairs(
            [(10.0, 11.0), (float("nan"), 3.0), (2.0, 5.0)]
        )
        self.assertEqual(accepted, 2)
        self.assertEqual(calibrator.residuals, [1.0, 3.0])

    def test_an_inf_residual_cannot_poison_the_quantile(self) -> None:
        calibrator = ConformalCalibrator(alpha=0.1)
        calibrator.seed([float("inf")] + [2.0] * 40)
        quantile = calibrator.conformity_quantile()
        self.assertTrue(math.isfinite(quantile))
        self.assertEqual(quantile, 2.0)

    def test_seeding_beyond_the_window_is_bounded(self) -> None:
        calibrator = ConformalCalibrator(alpha=0.1, max_calibration=50)
        calibrator.seed([float(i) for i in range(1, 201)])
        self.assertEqual(calibrator.n_calibration, 50)
        self.assertEqual(calibrator.snapshot()["max_calibration_window"], 50)

    def test_calibration_window_must_fit_the_minimum_population(self) -> None:
        with self.assertRaises(ValueError):
            ConformalCalibrator(alpha=0.1, max_calibration=3)


class TestIntervalConstruction(unittest.TestCase):
    """Required tests 5, 6 and 7: sufficiency, ordering, RUL domain."""

    @staticmethod
    def _calibrated(alpha: float = 0.1) -> ConformalCalibrator:
        """Calibrator with 100 residuals 1..100 -> quantile 91.0."""
        calibrator = ConformalCalibrator(alpha=alpha)
        calibrator.seed([float(i) for i in range(1, 101)])
        return calibrator

    def test_uncalibrated_interval_uses_the_documented_fallback(self) -> None:
        calibrator = ConformalCalibrator(alpha=0.1)
        interval = calibrator.interval(20.0, quality=1.0)
        self.assertFalse(interval["calibrated"])
        self.assertTrue(interval["uncalibrated_fallback"])
        self.assertIsNone(interval["conformity_quantile"])
        self.assertEqual(
            interval["base_half_width"],
            round(20.0 * FALLBACK_RELATIVE_HALF_WIDTH, 3),
        )
        # the absolute floor applies for short RUL / zero-quality inputs
        small = calibrator.interval(2.0, quality=1.0)
        self.assertEqual(small["base_half_width"], FALLBACK_MIN_HALF_WIDTH)

    def test_lower_bound_never_exceeds_upper_bound(self) -> None:
        calibrators = [
            ConformalCalibrator(alpha=0.1),  # uncalibrated
            self._calibrated(),
        ]
        for calibrator in calibrators:
            for predicted in (-5.0, 0.0, 0.25, 3.0, 30.0, 59.0, 1e6):
                for quality in (0.0, 0.3, 1.0):
                    interval = calibrator.interval(predicted, quality)
                    self.assertLessEqual(interval["low"], interval["high"])
                    self.assertGreaterEqual(interval["low"], 0.0)
                    self.assertGreaterEqual(interval["width"], 0.0)
                    self.assertAlmostEqual(
                        interval["width"],
                        round(interval["high"] - interval["low"], 3),
                        places=3,
                    )

    def test_invalid_prediction_is_rejected(self) -> None:
        calibrator = self._calibrated()
        for bad in (None, float("nan"), float("inf"), "abc"):
            with self.assertRaises(ValueError):
                calibrator.interval(bad, quality=1.0)

    def test_lower_bound_is_clipped_at_zero_and_that_is_conservative(
        self,
    ) -> None:
        calibrator = self._calibrated()
        interval = calibrator.interval(3.0, quality=1.0)
        self.assertTrue(interval["lower_bound_clipped"])
        self.assertEqual(interval["low"], 0.0)
        self.assertEqual(
            interval["high"],
            round(3.0 + interval["conformity_quantile"], 3),
        )
        # Clipping can only ever *add* covered area: only the part of the
        # interval that lies below the RUL domain is removed, so coverage
        # cannot drop below nominal.
        quantile = interval["conformity_quantile"]
        unclipped_low = 3.0 - quantile
        self.assertLess(unclipped_low, 0.0)
        self.assertEqual(interval["low"], max(unclipped_low, 0.0))
        self.assertEqual(interval["high"], round(3.0 + quantile, 3))

    def test_upper_bound_is_not_capped_at_the_label_maximum(self) -> None:
        """Capping high at the training-label maximum would break coverage."""
        calibrator = self._calibrated()
        interval = calibrator.interval(55.0, quality=1.0)
        # training labels span 1..59; the interval is allowed to exceed 59
        self.assertGreater(interval["high"], 59.0)
        self.assertAlmostEqual(
            interval["high"],
            round(55.0 + interval["conformity_quantile"], 3),
            places=3,
        )

    def test_quality_gate_only_widens(self) -> None:
        calibrator = self._calibrated()
        base = calibrator.interval(150.0, quality=1.0)
        mid = calibrator.interval(150.0, quality=0.5)
        poor = calibrator.interval(150.0, quality=0.0)

        self.assertEqual(
            base["base_half_width"], base["conformity_quantile"]
        )
        self.assertEqual(base["quality_widening_factor"], 1.0)
        self.assertEqual(
            mid["quality_widening_factor"],
            round(1 + 0.5 * QUALITY_WIDENING, 3),
        )
        self.assertEqual(
            poor["quality_widening_factor"],
            round(1 + QUALITY_WIDENING, 3),
        )
        # the calibrated base half-width is shared by all quality levels
        self.assertEqual(base["base_half_width"], mid["base_half_width"])
        self.assertEqual(base["base_half_width"], poor["base_half_width"])
        # and the width is monotonically non-decreasing as quality falls
        self.assertLessEqual(base["width"], mid["width"])
        self.assertLessEqual(mid["width"], poor["width"])

    def test_invalid_quality_is_treated_as_worst_case(self) -> None:
        calibrator = self._calibrated()
        interval = calibrator.interval(25.0, quality=float("nan"))
        self.assertEqual(interval["quality"], 0.0)
        self.assertEqual(
            interval["quality_widening_factor"],
            round(1 + QUALITY_WIDENING, 3),
        )

    def test_confidence_is_bounded(self) -> None:
        calibrator = self._calibrated()
        interval = calibrator.interval(1000.0, quality=1.0)
        score = calibrator.confidence(1000.0, 1.0, interval)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        # an uncalibrated calibrator is less trustworthy for the same input
        empty_score = ConformalCalibrator(alpha=0.1).confidence(1000.0, 1.0)
        self.assertLess(empty_score, score)


class TestReproducibility(unittest.TestCase):
    """Required test 8: fixed dataset + configuration -> same calibration."""

    def test_calibration_population_is_reproducible(self) -> None:
        predictor = _predictor()
        first = calibration_data.build_calibration_data(predictor=predictor)
        second = calibration_data.build_calibration_data(predictor=predictor)

        self.assertEqual(first.pairs, second.pairs)
        self.assertEqual(first.serials, second.serials)
        self.assertEqual(first.diagnostics, second.diagnostics)

        left = ConformalCalibrator(alpha=0.1)
        right = ConformalCalibrator(alpha=0.1)
        left.seed_from_pairs(first.pairs)
        right.seed_from_pairs(second.pairs)
        self.assertEqual(
            left.conformity_quantile(), right.conformity_quantile()
        )
        self.assertEqual(left.snapshot(), right.snapshot())

    def test_snapshot_reports_the_seeded_population(self) -> None:
        data = calibration_data.build_calibration_data(predictor=_predictor())
        calibrator = ConformalCalibrator(alpha=0.1)
        accepted = calibrator.seed_from_pairs(data.pairs)
        calibrator.source = data.diagnostics
        snapshot = calibrator.snapshot()

        self.assertEqual(accepted, data.n_observations)
        self.assertEqual(snapshot["calibration_samples"], data.n_observations)
        self.assertTrue(snapshot["calibration_sufficient"])
        self.assertEqual(snapshot["target_coverage"], 0.9)
        self.assertEqual(snapshot["quantile_index"], math.ceil(2923 * 0.9))
        self.assertEqual(snapshot["source"]["drive_overlap"], 0)
        self.assertLess(
            snapshot["median_abs_residual"], snapshot["conformity_quantile"]
        )
        self.assertIsNotNone(snapshot["interval_width_at_full_quality"])
        self.assertGreater(
            snapshot["interval_width_at_zero_quality"],
            snapshot["interval_width_at_full_quality"],
        )


class TestLeakyCalibrationMasksBias(unittest.TestCase):
    """Required test 9: test-set residuals make a bad predictor look valid."""

    def test_a_biased_predictor_looks_calibrated_when_the_test_set_is_used(
        self,
    ) -> None:
        rng = np.random.default_rng(7)
        # A predictor that is optimistic on the drives it was tuned against
        # and much worse on unseen drives.
        calibration_residuals = np.abs(rng.normal(0.0, 3.0, 400))
        evaluation_residuals = np.abs(rng.normal(0.0, 9.0, 400))

        leaky = ConformalCalibrator(alpha=0.1)
        leaky.seed(evaluation_residuals)  # WRONG: evaluation residuals
        honest = ConformalCalibrator(alpha=0.1)
        honest.seed(calibration_residuals)  # correct: disjoint drives

        def coverage(calibrator: ConformalCalibrator, residuals) -> float:
            quantile = calibrator.conformity_quantile()
            return float((np.asarray(residuals) <= quantile).mean())

        leaky_coverage = coverage(leaky, evaluation_residuals)
        honest_coverage = coverage(honest, evaluation_residuals)

        # The leaky protocol reports (approximately) nominal coverage ...
        self.assertGreaterEqual(leaky_coverage, 0.9)
        self.assertLessEqual(leaky_coverage, 0.95)
        # ... while the honest protocol exposes the predictor's real error.
        self.assertLess(honest_coverage, 0.6)
        self.assertGreater(honest_coverage, 0.2)
        self.assertLess(
            honest.conformity_quantile(), leaky.conformity_quantile()
        )

    def test_a_constant_bias_is_visible_as_an_interval_shift(self) -> None:
        calibrator = ConformalCalibrator(alpha=0.1)
        calibrator.seed([5.0] * 100)
        interval = calibrator.interval(20.0, quality=1.0)
        self.assertAlmostEqual(interval["low"], 15.0, places=3)
        self.assertAlmostEqual(interval["high"], 25.0, places=3)


class TestDeployedCalibrationEndpoint(unittest.TestCase):
    """Required test 10: the endpoint reports the real source and count."""

    @classmethod
    def setUpClass(cls) -> None:
        from fastapi.testclient import TestClient

        from app.main import app

        cls.client = TestClient(app)

    def test_endpoint_reports_source_sample_count_and_no_overlap(self) -> None:
        response = self.client.get("/api/v1/component2/twin/calibration")
        self.assertEqual(response.status_code, 200)
        payload = response.json()

        conformal = payload["conformal"]
        self.assertEqual(conformal["calibration_samples"], 2922)
        self.assertEqual(conformal["target_coverage"], 0.9)
        self.assertTrue(conformal["calibration_sufficient"])
        self.assertIsNotNone(conformal["conformity_quantile"])
        self.assertIsNone(payload["calibration_error"])

        source = conformal["source"]
        self.assertEqual(source["calibration_split"], "validation")
        self.assertEqual(source["evaluation_split"], "test")
        self.assertEqual(source["drive_overlap"], 0)
        self.assertEqual(source["overlap_check"], "pass")
        self.assertEqual(source["calibration_drives"], 89)
        self.assertEqual(source["calibration_observations"], 2922)
        self.assertEqual(source["sequence_length"], 30)
        self.assertEqual(source["checkpoint_file"], "best_bilstm_model.pt")
        self.assertEqual(len(source["checkpoint_sha256"]), 16)
        self.assertEqual(source["seeded_residuals"], 2922)
        self.assertFalse(source["seed_truncated"])

    def test_endpoint_does_not_expose_the_raw_dataset(self) -> None:
        response = self.client.get("/api/v1/component2/twin/calibration")
        source = response.json()["conformal"]["source"]
        # a reproducible digest, not the (potentially large) list of drives
        self.assertIn("calibration_drive_digest", source)
        self.assertEqual(len(source["calibration_drive_digest"]), 16)
        self.assertNotIn("serials", source)
        self.assertNotIn("pairs", source)
        self.assertLess(len(str(source)), 4000)

    def test_sync_interval_schema_keeps_existing_fields_and_adds_diagnostics(
        self,
    ) -> None:
        """The interval dict stays backwards compatible (additive keys only)."""
        import datetime

        serial = "CONFORMAL-CONTRACT-TEST"
        base = datetime.date(2024, 1, 1)
        state = None
        try:
            for day in range(31):
                response = self.client.post(
                    "/api/v1/component2/twin/sync",
                    json={
                        "serial_number": serial,
                        "observation": {
                            "date": (base + datetime.timedelta(days=day)).isoformat(),
                            "smart_1_normalized": 100.0 + day * 0.1,
                            "smart_5_normalized": day * 0.001,
                            "smart_187_normalized": day * 0.01,
                            "smart_197_normalized": 1.0,
                            "smart_198_normalized": day * 0.001,
                        },
                    },
                )
                self.assertEqual(response.status_code, 200)
                state = response.json()
        finally:
            self.client.delete(f"/api/v1/component2/twin/{serial}")

        interval = state["interval"]
        for key in (
            # pre-existing contract
            "low",
            "high",
            "width",
            "base_half_width",
            "quality",
            "coverage_alpha",
            "calibration_samples",
            # additive diagnostics
            "calibrated",
            "conformity_quantile",
            "quantile_index",
            "nominal_coverage",
            "quality_widening_factor",
            "lower_bound_clipped",
        ):
            self.assertIn(key, interval)

        self.assertTrue(interval["calibrated"])
        self.assertLessEqual(interval["low"], interval["high"])
        self.assertEqual(interval["nominal_coverage"], 0.9)
        self.assertEqual(
            interval["base_half_width"], interval["conformity_quantile"]
        )
        # raw and adapted RUL stay separate fields on the sync payload
        self.assertIn("raw_predicted_rul_days", state)
        self.assertIn("predicted_rul_days", state)
        self.assertIn("online_adaptation", state)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
