"""Unit tests for the risk-gated, degradation-aware adaptation controller.

Run with:
    cd 05_backend
    python -m pytest tests/test_risk_gate.py -q
"""

from __future__ import annotations

import unittest

import numpy as np

from app.ml.risk_gate import (
    ACTIVE,
    ADAPT,
    REJECT,
    ROLLBACK,
    SHADOW,
    WAIT,
    Outcome,
    RiskGatedAdapter,
)


def _batch(rng, n, bias, noise=2.0, degrading=False, quality=1.0, features=None):
    """Windows whose raw prediction misses the label by ``bias`` days."""
    out = []
    for _ in range(n):
        actual = float(rng.uniform(5.0, 55.0))
        predicted = max(0.0, actual + bias + float(rng.normal(0.0, noise)))
        out.append(
            Outcome(predicted, actual, degrading=degrading, quality=quality,
                    features=dict(features or {}))
        )
    return out


def _mae(gate, batches):
    errors = [
        abs(gate.apply(o.predicted, o.degrading)["adapted"] - o.actual)
        for b in batches for o in b
    ]
    return float(np.mean(errors))


class TestRiskGate(unittest.TestCase):
    def test_cold_start_is_raw_and_shadow(self):
        gate = RiskGatedAdapter()
        view = gate.apply(30.0)
        self.assertEqual(view["adapted"], 30.0)
        self.assertEqual(view["mode"], SHADOW)
        self.assertFalse(gate.deployed)

    def test_null_stream_false_acceptance_bounded_by_alpha(self):
        """Calibrated model: the gate must (almost) never ADAPT."""
        runs, adapted = 60, 0
        for seed in range(runs):
            rng = np.random.default_rng(seed)
            gate = RiskGatedAdapter(alpha=0.05)
            gate.seed([_batch(rng, 20, bias=0.0, noise=8.0) for _ in range(20)])
            actions = [
                gate.observe(_batch(rng, 20, bias=0.0, noise=8.0))["action"]
                for _ in range(40)
            ]
            adapted += ADAPT in actions
        self.assertLessEqual(adapted / runs, 0.05 + 0.05)  # alpha + MC slack

    def test_systematic_bias_is_learned_and_promoted(self):
        rng = np.random.default_rng(1)
        gate = RiskGatedAdapter()
        gate.seed([_batch(rng, 20, bias=8.0) for _ in range(10)])
        actions = [gate.observe(_batch(rng, 20, bias=8.0))["action"] for _ in range(20)]
        self.assertIn(ADAPT, actions)
        self.assertEqual(gate.mode, ACTIVE)
        holdout = [_batch(rng, 50, bias=8.0)]
        raw_mae = float(np.mean([abs(o.predicted - o.actual) for o in holdout[0]]))
        self.assertLess(_mae(gate, holdout), raw_mae * 0.6)

    def test_degrading_drive_never_gets_a_higher_rul(self):
        """Adaptation may make a degrading drive more conservative, never later."""
        rng = np.random.default_rng(2)
        gate = RiskGatedAdapter()
        # Model under-predicts by 8 d -> the learned correction is positive.
        gate.seed([_batch(rng, 20, bias=-8.0) for _ in range(10)])
        for _ in range(20):
            gate.observe(_batch(rng, 20, bias=-8.0))
        self.assertTrue(gate.deployed)
        stable = gate.apply(25.0, degrading=False)
        degrading = gate.apply(25.0, degrading=True)
        self.assertGreater(stable["adapted"], 25.0)
        self.assertEqual(degrading["adapted"], 25.0)
        self.assertTrue(degrading["blocked_optimistic"])

    def test_ablation_without_degradation_awareness_raises_rul(self):
        rng = np.random.default_rng(2)
        gate = RiskGatedAdapter(degradation_aware=False)
        gate.seed([_batch(rng, 20, bias=-8.0) for _ in range(10)])
        for _ in range(20):
            gate.observe(_batch(rng, 20, bias=-8.0))
        self.assertGreater(gate.apply(25.0, degrading=True)["adapted"], 25.0)

    def test_harmful_seed_is_rejected(self):
        """A challenger calibrated on another population is discarded."""
        rng = np.random.default_rng(3)
        gate = RiskGatedAdapter()
        gate.seed([_batch(rng, 30, bias=-10.0, noise=1.0) for _ in range(10)])
        actions = [gate.observe(_batch(rng, 20, bias=0.0, noise=1.0))["action"]
                   for _ in range(10)]
        self.assertIn(REJECT, actions)
        self.assertNotIn(ADAPT, actions[: actions.index(REJECT)])
        self.assertFalse(gate.deployed)

    def test_rollback_when_live_adaptation_turns_harmful(self):
        rng = np.random.default_rng(4)
        gate = RiskGatedAdapter()
        gate.seed([_batch(rng, 20, bias=10.0, noise=1.0) for _ in range(10)])
        for _ in range(10):
            gate.observe(_batch(rng, 20, bias=10.0, noise=1.0))
        self.assertEqual(gate.mode, ACTIVE)
        # The relation flips: the learned -10 d correction now doubles the error.
        actions = [gate.observe(_batch(rng, 3, bias=-10.0, noise=1.0))["action"]
                   for _ in range(15)]
        self.assertIn(ROLLBACK, actions)
        after = actions.index(ROLLBACK)
        self.assertEqual(gate.log[-(len(actions) - after)]["mode"], SHADOW)

    def test_low_quality_is_not_evidence(self):
        rng = np.random.default_rng(5)
        gate = RiskGatedAdapter()
        decision = gate.observe(_batch(rng, 10, bias=8.0, quality=0.2))
        self.assertEqual(decision["action"], WAIT)
        self.assertEqual(decision["reason"], "no_usable_evidence")
        self.assertEqual(decision["trained"], 0)

    def test_drift_score_ignores_degrading_windows_and_flags_shift(self):
        rng = np.random.default_rng(6)
        gate = RiskGatedAdapter()
        reference = [
            _batch(rng, 10, bias=0.0,
                   features={"smart_1_normalized": float(rng.normal(80, 5))})
            for _ in range(30)
        ]
        gate.set_reference(reference)
        shifted = _batch(rng, 10, 0.0, features={"smart_1_normalized": 120.0})
        self.assertGreaterEqual(gate.drift_score(shifted), 2.0)
        # The same shift on degrading windows is the failure, not drift.
        failing = _batch(rng, 10, 0.0, degrading=True,
                         features={"smart_1_normalized": 120.0})
        self.assertIsNone(gate.drift_score(failing))
        decision = gate.observe(shifted)
        self.assertTrue(decision["drift_detected"])
        self.assertEqual(decision["bet_cap"], 0.9)

    def test_snapshot_is_serialisable_summary(self):
        gate = RiskGatedAdapter()
        brief = gate.snapshot_brief()
        self.assertEqual(brief["mode"], SHADOW)
        self.assertEqual(brief["threshold"], 20.0)
        self.assertIn("checkpoint", gate.snapshot())


if __name__ == "__main__":
    unittest.main()
