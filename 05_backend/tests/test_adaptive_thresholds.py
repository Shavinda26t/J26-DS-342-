"""Tests for the per-drive adaptive degradation thresholds (Component 2,
Feature 3).

These pin the behaviour the repair relies on:

* zero / near-zero MAD and constant features defer to the global detector
  instead of producing an unbounded or silent personalization factor,
* missing, NaN and infinite SMART values never pollute the baseline windows
  and never produce a non-finite severity,
* cold start (too little baseline history) defers to the global detector,
* the strictness factor and every reported value stay inside documented bounds,
* ``suggested_persistence`` / ``suggested_min_severity`` are consistent with
  the global decision logic (they never raise the evidence bar),
* the severity floor invariant (adaptive <= global) always holds,
* the layer is deterministic for a fixed input stream,
* reading the state never mutates it and repeated identical observations do
  not drift the threshold,
* the semantics of "the global detector is a ceiling on sensitivity, not a
  warning-preservation guarantee" are pinned explicitly.

Run with:
    cd 05_backend
    python -m pytest tests/test_adaptive_thresholds.py -q
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from app.ml.adaptive_thresholds import (
    BASELINE_WARMUP,
    MAX_STRICTNESS,
    REFERENCE_SCALES,
    SIGNAL_CHANGE_7D,
    AdaptiveThresholds,
    fit_reference_scales,
)
from app.ml._component2_paths import get_detector

_DETECTOR = get_detector()

SEVERITY_PROBES = [-100.0, -45.0, -30.0, -23.0, -15.0, -11.0, -9.0,
                   -8.0, -5.0, -3.0, -2.0, -1.0, -0.5, 0.0, 5.0, 50.0]


def _feed(at, values, signal="smart_1", qualifying=False):
    """Feed a sequence of change values on one signal."""
    other = None
    for v in values:
        if signal == "smart_1":
            at.observe(v, other, qualifying)
        else:
            at.observe(other, v, qualifying)


def _global_severity(v1, v187):
    return int(_DETECTOR.combined_severity(v1, v187, missing=0))


class TestZeroMad(unittest.TestCase):
    """Required test 1: zero-MAD features."""

    def test_constant_zero_change_defers_to_global(self):
        at = AdaptiveThresholds(serial_number="zero")
        _feed(at, [0.0] * 30)
        snap = at.snapshot()
        self.assertEqual(snap["smart_1_change_noise"], 0.0)
        self.assertEqual(snap["smart_1_strictness"], 1.0)
        self.assertEqual(snap["smart_187_strictness"], 1.0)
        self.assertTrue(snap["is_calibrated"])
        for probe in SEVERITY_PROBES:
            self.assertEqual(at.severity(probe, None), _global_severity(probe, None))

    def test_constant_degrading_feature_is_not_treated_as_healthy(self):
        # A drive that drops by the same amount every observation never
        # contributes baseline samples, so it must fall back to the global
        # detector - never be silently considered healthy.
        at = AdaptiveThresholds(serial_number="flat-degrading")
        for _ in range(30):
            at.observe(-5.0, None, globally_qualifying=True)
        self.assertEqual(at.strictness("smart_1"), 1.0)
        self.assertFalse(at.snapshot()["is_calibrated"])
        self.assertEqual(at.severity(-5.0, None), _global_severity(-5.0, None))
        self.assertGreaterEqual(at.severity(-5.0, None), 1)

    def test_zero_mad_reference_disables_personalization(self):
        at = AdaptiveThresholds(
            serial_number="no-ref",
            reference_scales={"smart_1": None, "smart_187": None},
        )
        _feed(at, [0.0, -1.0, 2.0, -3.0] * 10)
        self.assertEqual(at.strictness("smart_1"), 1.0)
        self.assertEqual(at.severity(-2.0, None), _global_severity(-2.0, None))


class TestNearZeroMad(unittest.TestCase):
    """Required test 2: near-zero-MAD features."""

    def test_sub_floor_noise_defers_to_global(self):
        at = AdaptiveThresholds(serial_number="near-zero")
        _feed(at, [1e-9 * (1 if i % 2 else -1) for i in range(30)])
        self.assertEqual(at.strictness("smart_1"), 1.0)

    def test_noise_below_reference_never_relaxes_below_one(self):
        # MAD well below the fitted reference -> factor clamps at 1.0, so the
        # global threshold is never relaxed.
        at = AdaptiveThresholds(serial_number="small-noise")
        rng = np.random.default_rng(0)
        _feed(at, list(rng.uniform(-0.05, 0.05, 40)))
        self.assertGreaterEqual(at.strictness("smart_1"), 1.0)
        self.assertEqual(at.strictness("smart_1"), 1.0)

    def test_noise_above_reference_tightens_and_is_capped(self):
        ref = REFERENCE_SCALES["smart_1"]
        self.assertIsNotNone(ref)
        at = AdaptiveThresholds(serial_number="noisy")
        _feed(at, [0.0, 5 * ref, 0.0, 5 * ref] * 15)
        factor = at.strictness("smart_1")
        self.assertGreater(factor, 1.0)
        self.assertLessEqual(factor, MAX_STRICTNESS)


class TestInvalidValues(unittest.TestCase):
    """Required test 3: missing and non-finite SMART values."""

    def test_non_finite_and_missing_values_do_not_pollute_windows(self):
        at = AdaptiveThresholds(serial_number="invalid")
        at.observe(None, None, False)
        at.observe(float("nan"), float("inf"), False)
        at.observe(float("-inf"), float("nan"), True)
        at.observe("not-a-number", object(), False)
        snap = at.snapshot()
        self.assertEqual(snap["baseline_samples"], 0)
        self.assertEqual(snap["window_samples"], 0)
        for key, value in snap.items():
            if isinstance(value, float):
                self.assertTrue(math.isfinite(value), key)

    def test_invalid_values_never_produce_a_severity(self):
        at = AdaptiveThresholds(serial_number="invalid-sev")
        for bad in (None, float("nan"), float("inf"), float("-inf")):
            self.assertEqual(at.severity(bad, bad), 0)
        self.assertIsNone(at.acceleration_z(float("nan"), None))

    def test_missing_observations_leave_the_factor_unchanged(self):
        at = AdaptiveThresholds(serial_number="mostly-missing")
        _feed(at, [0.0, 4.0, 0.0, 4.0] * 12)
        before = at.strictness("smart_1")
        for _ in range(10):
            at.observe(None, None, False)
        self.assertEqual(at.strictness("smart_1"), before)


class TestColdStart(unittest.TestCase):
    """Required test 4: short history and cold-start behaviour."""

    def test_empty_state_is_safe(self):
        at = AdaptiveThresholds(serial_number="empty")
        snap = at.snapshot()
        self.assertFalse(snap["is_calibrated"])
        self.assertEqual(snap["smart_1_strictness"], 1.0)
        self.assertEqual(at.severity(None, None), 0)
        self.assertIsNone(at.acceleration_z(None, None))
        self.assertIsNone(at.observed_qualifying_rate())

    def test_short_history_defers_to_global(self):
        at = AdaptiveThresholds(serial_number="short")
        _feed(at, [0.0, 9.0, 0.0, -9.0, 0.0])
        self.assertFalse(at.snapshot()["is_calibrated"])
        self.assertEqual(at.strictness("smart_1"), 1.0)

    def test_becomes_calibrated_at_warmup_boundary(self):
        at = AdaptiveThresholds(serial_number="boundary")
        _feed(at, [0.0] * (BASELINE_WARMUP - 1))
        self.assertFalse(at.snapshot()["is_calibrated"])
        at.observe(0.0, None, False)
        self.assertTrue(at.snapshot()["is_calibrated"])


class TestBounds(unittest.TestCase):
    """Required test 5: thresholds stay within valid bounds; test 6: no
    invalid or infinite severity values."""

    def test_factor_and_severity_stay_in_bounds_over_random_streams(self):
        rng = np.random.default_rng(7)
        for _ in range(25):
            at = AdaptiveThresholds(serial_number="fuzz")
            for _ in range(120):
                c1 = float(rng.normal(0, 5))
                c187 = float(rng.normal(0, 8))
                qualifying = bool(_global_severity(c1, c187) >= 1)
                at.observe(c1, c187, qualifying)
                f1 = at.strictness("smart_1")
                f187 = at.strictness("smart_187")
                self.assertGreaterEqual(f1, 1.0)
                self.assertLessEqual(f1, at.max_strictness)
                self.assertGreaterEqual(f187, 1.0)
                self.assertLessEqual(f187, at.max_strictness)
                sev = at.severity(c1, c187)
                self.assertIn(sev, (0, 1, 2, 3, 4))
                self.assertLessEqual(sev, _global_severity(c1, c187) or 0)
            for value in at.snapshot().values():
                if isinstance(value, float):
                    self.assertTrue(math.isfinite(value))

    def test_cap_is_never_below_one(self):
        at = AdaptiveThresholds(serial_number="weird-cap", max_strictness=0.25)
        _feed(at, [0.0, 50.0] * 20)
        self.assertGreaterEqual(at.strictness("smart_1"), 1.0)
        self.assertLessEqual(at.strictness("smart_1"), 1.0)

    def test_signal_names_are_the_component2_change_columns(self):
        self.assertEqual(
            SIGNAL_CHANGE_7D,
            ("smart_1_normalized_change_7d", "smart_187_normalized_change_7d"),
        )


class TestSeverityFloorInvariant(unittest.TestCase):
    """The documented ceiling invariant: adaptive severity <= global severity."""

    def test_adaptive_severity_never_exceeds_global(self):
        rng = np.random.default_rng(11)
        at = AdaptiveThresholds(serial_number="invariant")
        # A noisy baseline so personalization actually engages.
        _feed(at, list(rng.normal(0, 6, 60)))
        for v1, v187 in zip(
            np.linspace(-60, 10, 71), np.linspace(-60, 10, 71)[::-1]
        ):
            self.assertLessEqual(
                at.severity(float(v1), float(v187)),
                _global_severity(float(v1), float(v187)),
            )

    def test_personalization_can_suppress_global_evidence(self):
        # Documents the semantics: the layer is a stricter detector, so it can
        # (deliberately) drop an observation the global detector would flag.
        at = AdaptiveThresholds(serial_number="suppression")
        _feed(at, [0.0, 5.0, 0.0, 5.0, 0.0, 5.0] * 12)
        self.assertGreater(at.strictness("smart_1"), 1.0)
        # -1.5 qualifies globally (mild) but must not once tightened.
        global_sev = _global_severity(-1.5, None)
        self.assertGreaterEqual(global_sev, 1)
        self.assertLess(at.severity(-1.5, None), global_sev)


class TestPersistenceConsistency(unittest.TestCase):
    """Required test 7: persistence recommendations agree with the documented
    decision logic; required test 10: global warning preservation."""

    def test_persistence_never_raises_the_evidence_bar(self):
        at = AdaptiveThresholds(serial_number="persist")
        base = int(_DETECTOR.PERSISTENCE)
        # A long qualifying streak drives the observed rate to 1.0; the old
        # implementation raised persistence here. The repaired one must not.
        for _ in range(60):
            at.observe(-20.0, -20.0, globally_qualifying=True)
        self.assertEqual(at.observed_qualifying_rate(), 1.0)
        self.assertEqual(at.suggested_persistence(base), base)
        self.assertEqual(at.suggested_min_severity(1), 1)

    def test_quiet_drive_does_not_lower_persistence(self):
        at = AdaptiveThresholds(serial_number="quiet")
        for _ in range(60):
            at.observe(0.0, 0.0, globally_qualifying=False)
        self.assertEqual(at.suggested_persistence(7), 7)
        self.assertEqual(at.suggested_min_severity(1), 1)

    def test_snapshot_reports_the_global_operating_point(self):
        at = AdaptiveThresholds(serial_number="snap")
        _feed(at, [0.0, 3.0, 0.0, 3.0] * 15)
        snap = at.snapshot()
        self.assertEqual(snap["suggested_persistence"], int(_DETECTOR.PERSISTENCE))
        self.assertEqual(snap["suggested_min_severity"], int(_DETECTOR.MIN_SEVERITY))
        self.assertEqual(snap["noise_source"], "baseline (non-degrading observations)")
        self.assertEqual(snap["strictness_cap"], MAX_STRICTNESS)


class TestDeterminism(unittest.TestCase):
    """Required test 8: deterministic results for a fixed input stream."""

    def test_two_instances_produce_identical_output(self):
        stream = [None, 0.0, -1.0, 12.0, -3.0, float("nan"), 0.0, -2.0] * 12
        snaps, severities = [], []
        for _ in range(2):
            at = AdaptiveThresholds(serial_number="det")
            sev = []
            for v in stream:
                qualifying = bool(_global_severity(v, v) >= 1) if v is not None else False
                at.observe(v, v, qualifying)
                sev.append(at.severity(v, v))
            snaps.append(at.snapshot())
            severities.append(sev)
        self.assertEqual(snaps[0], snaps[1])
        self.assertEqual(severities[0], severities[1])


class TestNoUnexpectedDrift(unittest.TestCase):
    """Required test 9: reads do not mutate; repeated observations do not
    change a threshold without an intended update."""

    def test_reading_state_is_idempotent(self):
        at = AdaptiveThresholds(serial_number="read")
        _feed(at, [0.0, 2.0, -1.0, 2.0] * 15)
        first = at.snapshot()
        for _ in range(20):
            at.snapshot()
            at.severity(-3.0, None)
            at.strictness("smart_1")
        self.assertEqual(at.snapshot(), first)

    def test_repeated_identical_observations_stabilise(self):
        # Saturate the baseline window with a constant value -> factor 1.0, and
        # keep feeding the same value: the factor must not drift.
        at = AdaptiveThresholds(serial_number="stable")
        _feed(at, [0.0] * 60)
        self.assertEqual(at.strictness("smart_1"), 1.0)
        for _ in range(100):
            at.observe(0.0, None, False)
        self.assertEqual(at.strictness("smart_1"), 1.0)

        # Repeat a full jitter cycle: the window distribution is unchanged, so
        # the factor must remain exactly the same.
        at2 = AdaptiveThresholds(serial_number="stable-cycle")
        _feed(at2, [0.0, 4.0] * 30)
        settled = at2.strictness("smart_1")
        for _ in range(50):
            at2.observe(0.0, None, False)
            at2.observe(4.0, None, False)
        self.assertEqual(at2.strictness("smart_1"), settled)

    def test_only_observe_mutates_baseline_windows(self):
        at = AdaptiveThresholds(serial_number="mut")
        _feed(at, [0.0, 7.0, 0.0, 7.0] * 12)
        before = (at.snapshot()["baseline_samples"], at.strictness("smart_1"))
        at.severity(-9.0, -9.0)
        at.acceleration_z(-9.0, -9.0)
        after = (at.snapshot()["baseline_samples"], at.strictness("smart_1"))
        self.assertEqual(before, after)


class TestReferenceScaleFitting(unittest.TestCase):
    """The population reference scale must be fit from baseline data only."""

    def test_median_of_per_drive_mads(self):
        # Per-drive MAD is scaled by 1.4826 (consistent sigma estimator).
        # A drive jittering by +-2 has scaled MAD 2 * 1.4826 = 2.9652, so the
        # median over [0, 2.9652, 2.9652, 2.9652] is 2.9652.
        jitter = [0.0, 2.0, -2.0, 2.0, -2.0] * 4
        drives = [[0.0] * 20, jitter, jitter, jitter]
        fitted = fit_reference_scales({"smart_1": drives})
        self.assertAlmostEqual(fitted["smart_1"], 2.9652, places=3)

    def test_larger_baseline_jitter_gives_a_larger_reference(self):
        quiet = [0.0, 1.0, -1.0, 1.0, -1.0] * 6
        noisy = [0.0, 8.0, -8.0, 8.0, -8.0] * 6
        f_quiet = fit_reference_scales({"smart_1": [quiet] * 4})["smart_1"]
        f_noisy = fit_reference_scales({"smart_1": [noisy] * 4})["smart_1"]
        self.assertGreater(f_noisy, f_quiet)

    def test_all_zero_mads_disable_the_reference(self):
        fitted = fit_reference_scales({"smart_1": [[0.0] * 20, [0.0] * 20]})
        self.assertIsNone(fitted["smart_1"])

    def test_short_drives_and_empty_input_are_ignored(self):
        fitted = fit_reference_scales(
            {"smart_187": [[1.0, 2.0], [0.0, 3.0, 0.0, 3.0] * 5]}
        )
        self.assertNotEqual(fitted["smart_187"], 0.0)
        self.assertIsNone(fit_reference_scales({"smart_1": []})["smart_1"])

    def test_shipped_reference_scales_are_documented_and_finite(self):
        self.assertIn("smart_1", REFERENCE_SCALES)
        self.assertIn("smart_187", REFERENCE_SCALES)
        self.assertGreater(REFERENCE_SCALES["smart_1"], 0.0)
        # smart_187 had no drive with measurable baseline noise -> disabled.
        self.assertIsNone(REFERENCE_SCALES["smart_187"])


class TestGlobalWarningSemantics(unittest.TestCase):
    """Required test 10, resolved: global warning preservation is NOT an
    explicit design requirement for this layer, and the deployed twin does not
    let it gate warnings - so its absence is pinned rather than assumed."""

    def test_layer_only_tightens_never_relaxes(self):
        at = AdaptiveThresholds(serial_number="tighten-only")
        rng = np.random.default_rng(3)
        _feed(at, list(rng.normal(0, 4, 60)))
        for probe in SEVERITY_PROBES:
            self.assertLessEqual(at.severity(probe, probe), _global_severity(probe, probe))

    def test_operating_point_is_left_to_the_global_detector(self):
        # Because the layer cannot relax the global bounds and does not change
        # persistence / min-severity, the *decision* rule is identical to the
        # global one; only the severity evidence may be reduced.
        at = AdaptiveThresholds(serial_number="op")
        _feed(at, [0.0, 6.0] * 30)
        self.assertEqual(at.suggested_persistence(), int(_DETECTOR.PERSISTENCE))
        self.assertEqual(at.suggested_min_severity(), int(_DETECTOR.MIN_SEVERITY))


if __name__ == "__main__":
    unittest.main()
