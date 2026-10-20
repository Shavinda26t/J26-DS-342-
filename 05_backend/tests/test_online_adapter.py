"""Unit tests for the online RUL adaptation module.

Deterministic tests for the repaired recursive-least-squares (RLS) adapter:

1.  recovery of known linear coefficients,
2.  stable convergence from a cold start,
3.  nearly-collinear (ill-conditioned) inputs,
4.  coefficient-bound enforcement without corrupting the covariance,
5.  missing / non-finite / out-of-domain inputs,
6.  long update sequences without numerical divergence,
7.  the raw prediction staying immutable while adapting,
8.  the adapted value never being fed back as a raw observation,
9.  a safe fallback when the adapter becomes numerically invalid,
10. reproducibility of repeated updates,
11. the prequential no-harm guard,
12. the population seed gate (drive-level, out-of-sample).

Plus two regression tests: the previously observed cold-start failure in which
the coefficients locked onto their bounds, and the stream-vs-population failure
in which a short biased burst of outcomes passed the streaming guard and
applied an unvalidated population correction.  The legacy algorithm is embedded
below so the first regression test documents *why* it exists.

Run with:
    cd 05_backend
    python -m pytest tests/test_online_adapter.py -q
"""

from __future__ import annotations

import hashlib
import math
import unittest

import numpy as np

from app.ml.online_adapter import (
    B1_MAX,
    B1_MIN,
    DEFAULT_DELTA,
    DEFAULT_FORGETTING,
    MAX_CORRECTION_DAYS,
    MAX_CORRECTION_FRACTION,
    MIN_GUARD_UPDATES,
    POPULATION_FORGETTING,
    DualRulAdapter,
    OnlineRulAdapter,
)


class _LegacyAdapter:
    """The original (pre-repair) RLS, kept only for the regression test.

    Uncentred regressor, ``P = 100 I``, and a post-hoc clamp that leaves the
    covariance inconsistent with the clamped coefficients.  The clamp bounds
    are the original literals (intercept +/- 60, slope [0.3, 3.0]).
    """

    LEGACY_INTERCEPT_BOUND = 60.0

    def __init__(self, forgetting: float = 0.985) -> None:
        self.forgetting = forgetting
        self.theta = np.array([0.0, 1.0], dtype=float)
        self.p = np.eye(2) * 100.0
        self.n_updates = 0

    def update(self, predicted, actual) -> None:
        if predicted is None or actual is None:
            return
        if math.isnan(predicted) or math.isnan(actual):
            return
        phi = np.array([1.0, float(predicted)])
        d = float(actual)
        p_phi = self.p @ phi
        denom = self.forgetting + float(phi @ p_phi)
        if denom <= 0 or not math.isfinite(denom):
            return
        k = p_phi / denom
        self.theta = self.theta + k * (d - float(self.theta @ phi))
        self.p = (self.p - np.outer(k, phi) @ self.p) / self.forgetting
        self.theta[1] = float(np.clip(self.theta[1], B1_MIN, B1_MAX))
        self.theta[0] = float(
            np.clip(
                self.theta[0],
                -self.LEGACY_INTERCEPT_BOUND,
                self.LEGACY_INTERCEPT_BOUND,
            )
        )
        self.n_updates += 1


def _drive_hash(serial: str) -> int:
    """Mirror of the adapter's deterministic drive-level split."""
    return int(hashlib.sha256(serial.encode("utf-8")).hexdigest(), 16) % 100


class TestRlsMathematics(unittest.TestCase):
    # ------------------------------------------------------------------
    def test_01_recovers_known_linear_coefficients(self):
        rng = np.random.default_rng(11)
        x = np.linspace(5.0, 60.0, 400)
        y = 3.0 + 0.9 * x + rng.normal(0.0, 0.8, x.size)

        adapter = OnlineRulAdapter()
        for xi, yi in zip(x, y):
            self.assertTrue(adapter.update(xi, yi))

        b0, b1 = adapter.theta
        self.assertAlmostEqual(b1, 0.9, delta=0.05)
        self.assertAlmostEqual(b0, 3.0, delta=2.0)

        # the fitted correction predicts the known relation better than raw
        raw_mae = float(np.mean(np.abs(y - x)))
        fitted_mae = float(np.mean([abs(yi - adapter.adapt(xi))
                                    for xi, yi in zip(x, y)]))
        self.assertLess(fitted_mae, raw_mae)

    # ------------------------------------------------------------------
    def test_02_cold_start_is_a_no_op_then_converges(self):
        adapter = OnlineRulAdapter()
        # no evidence yet -> exactly the raw prediction, no correction
        self.assertEqual(adapter.adapt(20.0), 20.0)
        self.assertEqual(adapter.correction(20.0), 0.0)
        self.assertEqual(adapter.n_updates, 0)
        self.assertFalse(adapter.effective)

        rng = np.random.default_rng(2)
        x = np.linspace(5.0, 60.0, 300)
        y = x + 4.0 + rng.normal(0.0, 0.5, x.size)
        for xi, yi in zip(x, y):
            adapter.update(xi, yi)

        self.assertAlmostEqual(adapter.theta[0], 4.0, delta=1.5)
        self.assertAlmostEqual(adapter.theta[1], 1.0, delta=0.05)
        self.assertTrue(adapter.effective)
        self.assertAlmostEqual(adapter.correction(30.0), 4.0, delta=1.0)
        self.assertTrue(adapter.is_valid)
        self.assertTrue(np.all(np.isfinite(adapter.p)))

    # ------------------------------------------------------------------
    def test_03_nearly_collinear_inputs_stay_bounded(self):
        # a cold start whose first observations are almost identical is the
        # classic ill-conditioned case that used to explode
        x = np.concatenate(
            [[5.00, 5.02, 5.04, 5.06, 5.08], np.linspace(5.1, 60.0, 250)]
        )
        y = 2.0 + 1.5 * x

        adapter = OnlineRulAdapter()
        for xi, yi in zip(x, y):
            self.assertTrue(adapter.update(xi, yi))

        self.assertTrue(np.all(np.isfinite(adapter.theta)))
        self.assertTrue(np.all(np.isfinite(adapter.p)))
        self.assertAlmostEqual(adapter.theta[1], 1.5, delta=0.1)
        self.assertAlmostEqual(adapter.theta[0], 2.0, delta=2.0)

    # ------------------------------------------------------------------
    def test_04_regression_cold_start_does_not_lock_on_bounds(self):
        """Regression: coefficients used to stick at their bounds forever."""
        rng = np.random.default_rng(1)
        x = np.linspace(5.0, 60.0, 200)
        # warm-up draw that made the old regressor pathologically correlated
        _ = np.concatenate([x[:100] - 8.0, x[100:] + 8.0]) + rng.normal(
            0.0, 1.0, 200
        )
        y = 2.0 + 1.5 * x + rng.normal(0.0, 1.0, x.size)

        legacy = _LegacyAdapter()
        for xi, yi in zip(x, y):
            legacy.update(xi, yi)
        # documents the old failure: far from the truth, pinned near a corner
        self.assertGreater(abs(legacy.theta[0] - 2.0), 10.0)

        repaired = OnlineRulAdapter()
        for xi, yi in zip(x, y):
            self.assertTrue(repaired.update(xi, yi))
        b0, b1 = repaired.theta
        self.assertAlmostEqual(b0, 2.0, delta=3.0)
        self.assertAlmostEqual(b1, 1.5, delta=0.15)

    # ------------------------------------------------------------------
    def test_05_bounds_enforced_without_covariance_corruption(self):
        rng = np.random.default_rng(5)
        x = np.linspace(5.0, 60.0, 200)

        adapter = OnlineRulAdapter()
        for xi, yi in zip(x, 5.0 * x + 2.0 + rng.normal(0.0, 0.5, x.size)):
            adapter.update(xi, yi)
            # every step stays finite and positive semidefinite
            self.assertTrue(np.all(np.isfinite(adapter.p)))
            self.assertGreaterEqual(
                float(np.linalg.eigvalsh(adapter.p).min()), -1e-9
            )
            self.assertLessEqual(adapter.theta[1], B1_MAX + 1e-9)
            self.assertGreaterEqual(adapter.theta[1], B1_MIN - 1e-9)

        # the slope (truth 5.0) is pinned at the upper bound ...
        self.assertAlmostEqual(adapter.theta[1], B1_MAX, delta=1e-6)
        # ... but must not be *locked* there once honest data arrives
        for xi, yi in zip(x, 1.0 * x + 2.0 + rng.normal(0.0, 0.5, x.size)):
            adapter.update(xi, yi)
        self.assertLess(adapter.theta[1], B1_MAX - 0.05)
        self.assertAlmostEqual(adapter.theta[1], 1.0, delta=0.15)

    # ------------------------------------------------------------------
    def test_06_missing_and_non_finite_inputs_are_rejected(self):
        adapter = OnlineRulAdapter()
        for xi, yi in zip([5.0, 6.0, 7.0], [1.0, 2.0, 3.0]):
            adapter.update(xi, yi)

        theta_before = adapter.theta.copy()
        p_before = adapter.p.copy()
        updates_before = adapter.n_updates

        bad_pairs = [
            (None, 5.0),
            (5.0, None),
            (float("nan"), 5.0),
            (5.0, float("nan")),
            (float("inf"), 5.0),
            (5.0, float("-inf")),
            ("abc", 5.0),
            (5.0, "abc"),
            (-1.0, 5.0),
            (5.0, -1.0),
            (1e6, 5.0),
            (5.0, -999.0),
        ]
        for xi, yi in bad_pairs:
            self.assertFalse(adapter.update(xi, yi))

        self.assertEqual(adapter.n_updates, updates_before)
        self.assertEqual(adapter.n_rejected, len(bad_pairs))
        np.testing.assert_allclose(adapter.theta, theta_before)
        np.testing.assert_allclose(adapter.p, p_before)

        # invalid predictions fall back to the input at inference time
        self.assertTrue(math.isnan(adapter.adapt(float("nan"))))
        self.assertIsNone(adapter.adapt(None))

    # ------------------------------------------------------------------
    def test_07_raw_prediction_is_immutable_during_adaptation(self):
        adapter = OnlineRulAdapter()
        for xi, yi in zip(np.linspace(5.0, 60.0, 50),
                          np.linspace(6.0, 61.0, 50)):
            adapter.update(xi, yi)

        theta_before = adapter.theta.copy()
        p_before = adapter.p.copy()
        updates_before = adapter.n_updates
        last_predicted = adapter.last_predicted

        raw = 27.5
        adapted = adapter.adapt(raw)

        self.assertIsInstance(adapted, float)
        self.assertEqual(raw, 27.5)  # adapt() never mutates its input
        np.testing.assert_array_equal(adapter.theta, theta_before)
        np.testing.assert_array_equal(adapter.p, p_before)
        self.assertEqual(adapter.n_updates, updates_before)
        self.assertEqual(adapter.last_predicted, last_predicted)

    # ------------------------------------------------------------------
    def test_08_adapted_value_is_never_fed_back_as_raw(self):
        dual = DualRulAdapter()
        serial = "SERIAL-A"

        # age the personal adapter with genuine (raw, actual) pairs
        for i in range(40):
            raw_i = 5.0 + i * 1.2
            dual.update(serial, raw_i, raw_i + 4.0)

        personal = dual._get(serial)
        self.assertTrue(personal.effective)

        raw = 30.0
        decision = dual.adapt(serial, raw)
        self.assertEqual(decision["raw"], raw)
        self.assertNotEqual(decision["adapted"], decision["raw"])

        # the service feeds the *raw* value back (this is what _build_state
        # stores in _last_predicted); the adapted value must never be used
        feedback = decision["raw"]
        updates_before = personal.n_updates
        dual.update(serial, feedback, 34.0)
        self.assertEqual(personal.n_updates, updates_before + 1)
        self.assertEqual(personal.last_predicted, raw)
        self.assertNotEqual(personal.last_predicted, decision["adapted"])

    # ------------------------------------------------------------------
    def test_09_safe_fallback_when_numerically_invalid(self):
        adapter = OnlineRulAdapter()
        for xi, yi in zip(np.linspace(5.0, 60.0, 40),
                          np.linspace(6.0, 61.0, 40)):
            adapter.update(xi, yi)
        self.assertTrue(adapter.effective)

        # fault injection: simulate an upstream numerical blow-up
        adapter._p = np.full((2, 2), np.nan)
        self.assertFalse(adapter.update(31.0, 32.0))
        self.assertTrue(adapter.failed)
        self.assertFalse(adapter.is_valid)

        # every output collapses to the raw prediction
        self.assertEqual(adapter.adapt(30.0), 30.0)
        self.assertEqual(adapter.correction(30.0), 0.0)
        self.assertEqual(adapter.proposed_correction(30.0), 0.0)
        # the failure latches
        self.assertFalse(adapter.update(33.0, 34.0))

        # an explicit reset clears it
        adapter.reset()
        self.assertFalse(adapter.failed)
        self.assertEqual(adapter.adapt(30.0), 30.0)
        self.assertEqual(adapter.n_updates, 0)

    # ------------------------------------------------------------------
    def test_10_disabled_adapter_always_returns_the_raw_prediction(self):
        adapter = OnlineRulAdapter(enabled=False)
        for xi, yi in zip(np.linspace(5.0, 60.0, 60),
                          np.linspace(15.0, 70.0, 60)):
            adapter.update(xi, yi)

        self.assertGreater(adapter.n_updates, 0)  # still learns ...
        self.assertFalse(adapter.effective)       # ... but never applies
        self.assertEqual(adapter.adapt(41.5), 41.5)
        self.assertEqual(adapter.correction(41.5), 0.0)

    # ------------------------------------------------------------------
    def test_11_repeated_updates_are_reproducible(self):
        x = np.linspace(5.0, 60.0, 150)
        y = 3.0 + 0.9 * x

        first = OnlineRulAdapter()
        second = OnlineRulAdapter()
        for xi, yi in zip(x, y):
            first.update(xi, yi)
            second.update(xi, yi)

        np.testing.assert_array_equal(first.theta, second.theta)
        np.testing.assert_array_equal(first.p, second.p)
        self.assertEqual(first.adapt(30.0), second.adapt(30.0))

        first.reset()
        for xi, yi in zip(x, y):
            first.update(xi, yi)
        np.testing.assert_array_equal(first.theta, second.theta)
        np.testing.assert_array_equal(first.p, second.p)

    # ------------------------------------------------------------------
    def test_12_long_sequence_does_not_diverge(self):
        rng = np.random.default_rng(3)
        adapter = OnlineRulAdapter()
        for _ in range(50_000):
            adapter.update(rng.uniform(1.0, 60.0), rng.uniform(0.0, 70.0))

        self.assertTrue(adapter.is_valid)
        self.assertTrue(np.all(np.isfinite(adapter.theta)))
        self.assertTrue(np.all(np.isfinite(adapter.p)))
        self.assertGreaterEqual(
            float(np.linalg.eigvalsh(adapter.p).min()), -1e-9
        )
        self.assertLessEqual(adapter.theta[1], B1_MAX + 1e-9)
        self.assertGreaterEqual(adapter.theta[1], B1_MIN - 1e-9)
        self.assertLessEqual(
            abs(adapter.adapt(30.0) - 30.0), MAX_CORRECTION_DAYS + 1e-9
        )

    # ------------------------------------------------------------------
    def test_13_correction_magnitude_is_bounded(self):
        adapter = OnlineRulAdapter()
        for xi in np.linspace(5.0, 60.0, 200):
            adapter.update(xi, 12.0)  # constant outcome: maximal disagreement

        for raw in (5.0, 12.0, 30.0, 60.0):
            correction = adapter.proposed_correction(raw)
            self.assertLessEqual(abs(correction), MAX_CORRECTION_DAYS + 1e-9)
            self.assertLessEqual(
                abs(correction), abs(raw) * MAX_CORRECTION_FRACTION + 1e-9
            )
            self.assertGreaterEqual(adapter.adapt(raw), 0.0)

    # ------------------------------------------------------------------
    def test_14_guard_exposes_prequential_error_comparison(self):
        adapter = OnlineRulAdapter()
        rng = np.random.default_rng(2)
        x = np.linspace(5.0, 60.0, 120)
        y = x + 4.0 + rng.normal(0.0, 0.5, x.size)
        for xi, yi in zip(x, y):
            adapter.update(xi, yi)

        self.assertEqual(adapter.n_updates, 120)
        self.assertIsNotNone(adapter.raw_mean_abs_error)
        self.assertIsNotNone(adapter.adapted_mean_abs_error)
        self.assertLess(
            adapter.adapted_mean_abs_error, adapter.raw_mean_abs_error
        )
        self.assertTrue(adapter.beneficial)
        self.assertTrue(adapter.effective)

    # ------------------------------------------------------------------
    def test_15_guard_is_conservative_before_enough_evidence(self):
        adapter = OnlineRulAdapter()
        self.assertFalse(adapter.beneficial)
        self.assertFalse(adapter.effective)
        for i in range(MIN_GUARD_UPDATES - 1):
            adapter.update(5.0 + i, 25.0)
        self.assertFalse(adapter.beneficial)
        self.assertFalse(adapter.effective)

    # ------------------------------------------------------------------
    def test_16_guard_keeps_adaptation_off_when_there_is_nothing_to_correct(self):
        # The raw model is already unbiased on this stream, so any learned
        # correction can only add variance.  The guard must refuse to apply one
        # even though the coefficients are fully converged.
        rng = np.random.default_rng(6)
        adapter = OnlineRulAdapter()
        x = np.linspace(5.0, 60.0, 300)
        for xi, yi in zip(x, x + rng.normal(0.0, 1.0, x.size)):
            adapter.update(xi, yi)

        self.assertEqual(adapter.n_updates, 300)
        raw_mae = adapter.raw_mean_abs_error
        adapted_mae = adapter.adapted_mean_abs_error
        self.assertLess(abs(adapted_mae - raw_mae), 0.10 * raw_mae)
        self.assertFalse(adapter.beneficial)
        self.assertFalse(adapter.effective)
        self.assertEqual(adapter.adapt(30.0), 30.0)
        self.assertEqual(adapter.correction(30.0), 0.0)


class TestPopulationSeed(unittest.TestCase):
    # ------------------------------------------------------------------
    def test_17_good_population_relation_is_validated(self):
        rng = np.random.default_rng(9)
        pairs, serials = [], []
        for index in range(30):
            serial = f"DRIVE{index:03d}"
            for _ in range(12):
                raw = rng.uniform(5.0, 60.0)
                pairs.append((raw, raw + 5.0 + rng.normal(0.0, 0.5)))
                serials.append(serial)

        dual = DualRulAdapter()
        diagnostics = dual.seed_population(pairs, serials)

        self.assertEqual(diagnostics["status"], "validated")
        self.assertGreater(diagnostics["holdout_gain"], 0.02)
        self.assertGreater(diagnostics["holdout_adapted_mae"], 0.0)
        # drives never straddle the fit / hold-out boundary
        self.assertEqual(
            diagnostics["fit_drives"] + diagnostics["holdout_drives"], 30
        )
        self.assertTrue(dual.global_adapter.validated)
        self.assertTrue(dual.global_adapter.effective)
        self.assertEqual(dual.adapt("UNSEEN", 30.0)["source"], "global")

    # ------------------------------------------------------------------
    def test_18_harmful_population_relation_is_rejected(self):
        # fit drives and hold-out drives follow opposite relations, so any
        # population correction fitted on the former must hurt the latter
        rng = np.random.default_rng(10)
        pairs, serials = [], []
        for index in range(40):
            serial = f"DRIVE{index:03d}"
            bias = 20.0 if _drive_hash(serial) < 80 else -20.0
            for _ in range(12):
                raw = rng.uniform(5.0, 60.0)
                pairs.append((raw, raw + bias))
                serials.append(serial)

        dual = DualRulAdapter()
        diagnostics = dual.seed_population(pairs, serials)

        self.assertEqual(diagnostics["status"], "rejected_no_gain")
        self.assertFalse(dual.global_adapter.validated)
        self.assertEqual(dual.global_adapter.n_updates, 0)

        decision = dual.adapt("UNSEEN", 30.0)
        self.assertEqual(decision["source"], "none")
        self.assertEqual(decision["adapted"], 30.0)
        self.assertEqual(decision["correction_days"], 0.0)

    # ------------------------------------------------------------------
    def test_19_seed_rejects_invalid_pairs(self):
        dual = DualRulAdapter()
        diagnostics = dual.seed_population(
            [
                (5.0, -999.0),
                (5.0, 1e6),
                (float("nan"), 5.0),
                (5.0, float("inf")),
                (float("inf"), 5.0),
            ]
        )
        self.assertEqual(diagnostics["pairs_rejected"], 5)
        self.assertEqual(diagnostics["status"], "insufficient_pairs")
        self.assertEqual(dual.global_adapter.n_updates, 0)
        self.assertEqual(dual.adapt("X", 7.0)["adapted"], 7.0)

    # ------------------------------------------------------------------
    def test_20_seeded_scaler_is_shared_with_personal_adapters(self):
        rng = np.random.default_rng(12)
        pairs, serials = [], []
        for index in range(25):
            serial = f"D{index}"
            for _ in range(10):
                raw = rng.uniform(10.0, 50.0)
                pairs.append((raw, raw + 3.0 + rng.normal(0.0, 0.4)))
                serials.append(serial)

        dual = DualRulAdapter()
        dual.seed_population(pairs, serials)
        personal = dual._get("NEW")

        self.assertAlmostEqual(personal.center, dual.center)
        self.assertAlmostEqual(personal.scale, dual.scale)
        self.assertGreater(dual.scale, 0.0)

    # ------------------------------------------------------------------
    def test_21_seed_requires_matching_serials(self):
        dual = DualRulAdapter()
        with self.assertRaises(ValueError):
            dual.seed_population([(1.0, 2.0)], serials=["a", "b"])

    # ------------------------------------------------------------------
    def test_22_reset_restores_cold_start(self):
        rng = np.random.default_rng(13)
        pairs, serials = [], []
        for index in range(25):
            serial = f"D{index}"
            for _ in range(10):
                raw = rng.uniform(10.0, 50.0)
                pairs.append((raw, raw + 3.0))
                serials.append(serial)

        dual = DualRulAdapter()
        dual.seed_population(pairs, serials)
        dual.update("S", 20.0, 23.0)
        self.assertTrue(dual.serial_adapters)

        dual.reset()
        self.assertFalse(dual.serial_adapters)
        self.assertIsNone(dual.seed_diagnostics)
        self.assertEqual(dual.global_adapter.n_updates, 0)
        self.assertEqual(dual.global_adapter.delta, DEFAULT_DELTA)
        self.assertEqual(dual.adapt("S", 30.0)["adapted"], 30.0)

    # ------------------------------------------------------------------
    def test_23_population_adapter_is_not_fooled_by_a_short_biased_stream(
        self,
    ):
        """Regression: a biased burst used to pass the streaming guard.

        Measured on the real data (time-aware protocol): 13 streamed outcomes
        gave the global adapter a fictitious 65 % prequential gain, pinned the
        slope at its lower bound and then applied +-15 d to every drive
        (+20.8 % MAE).  A population correction must instead require the
        drive-level, out-of-sample verdict, so the burst must change nothing.
        """
        dual = DualRulAdapter()
        global_adapter = dual.global_adapter
        self.assertTrue(global_adapter.require_validation)

        # one biased burst: outcomes far from the raw predictions
        for index in range(60):
            dual.update("DRIVE-A", 30.0 + (index % 3), 55.0 + (index % 3))

        # the streaming comparison genuinely *looks* favourable ...
        self.assertGreaterEqual(global_adapter.n_updates, MIN_GUARD_UPDATES)
        self.assertGreater(global_adapter.raw_mean_abs_error, 20.0)
        self.assertLess(
            global_adapter.adapted_mean_abs_error,
            global_adapter.raw_mean_abs_error * 0.5,
        )
        self.assertTrue(global_adapter.beneficial)
        # ... but it is not a drive-level, out-of-sample verdict
        self.assertFalse(global_adapter.validated)
        self.assertFalse(global_adapter.effective)
        self.assertFalse(global_adapter.correction(30.0))

        decision = dual.adapt("DRIVE-B", 30.0)
        self.assertEqual(decision["source"], "none")
        self.assertEqual(decision["reason"], "population_not_validated")
        self.assertEqual(decision["adapted"], 30.0)
        self.assertEqual(decision["correction_days"], 0.0)

    # ------------------------------------------------------------------
    def test_24_population_and_per_drive_adapters_use_different_configs(self):
        """The population fit is stationary (plain RLS) and gated by drives."""
        dual = DualRulAdapter()
        self.assertEqual(dual.global_adapter.forgetting, POPULATION_FORGETTING)
        self.assertTrue(dual.global_adapter.require_validation)

        personal = dual._get("DRIVE-C")
        self.assertEqual(personal.forgetting, DEFAULT_FORGETTING)
        self.assertFalse(personal.require_validation)

        # the configuration survives reset() and a population seed
        dual.reset()
        self.assertEqual(dual.global_adapter.forgetting, POPULATION_FORGETTING)
        self.assertTrue(dual.global_adapter.require_validation)

        rng = np.random.default_rng(15)
        pairs, serials = [], []
        for index in range(25):
            serial = f"D{index}"
            for _ in range(10):
                raw = rng.uniform(10.0, 50.0)
                pairs.append((raw, 3.0 + 0.9 * raw + rng.normal(0.0, 0.3)))
                serials.append(serial)
        dual.seed_population(pairs, serials)
        self.assertEqual(dual.global_adapter.forgetting, POPULATION_FORGETTING)
        self.assertEqual(dual.global_adapter.require_validation, True)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
