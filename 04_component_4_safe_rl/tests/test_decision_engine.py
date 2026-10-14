
"""
Component 4: HDD Decision Engine Automated Tests

Research objective:
Validate risk-based decision making and integration
with the independent mitigation safety validator.

All scenarios are simulated.
No physical disk actions are performed.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from contracts import DriveState, StorageContext, MitigationAction
from decision_engine import HDDDecisionEngine, DecisionConfig


def make_state(
    risk=0.05,
    rul=100.0,
    horizon=7,
    **storage_overrides
):
    """Build reproducible test scenarios."""

    storage_data = {
        "free_capacity_gb": 1000.0,
        "data_to_migrate_gb": 500.0,
        "healthy_redundant_copies": 2,
        "target_healthy": True,
        "migration_progress_fraction": 0.0,
        "io_utilization": 0.40,
        "admin_approved": False,
    }

    storage_data.update(storage_overrides)

    return DriveState(
        disk_id="TEST_HDD_001",
        observed_at="2026-10-14T12:00:00+05:30",
        failure_probability=risk,
        prediction_horizon_days=horizon,
        rul_days=rul,
        storage=StorageContext(**storage_data)
    )


class TestHDDDecisionEngine(unittest.TestCase):

    def setUp(self):
        self.engine = HDDDecisionEngine()

    def test_01_healthy_drive_no_action(self):
        result = self.engine.decide(make_state())
        self.assertEqual(result.risk_level, "LOW")
        self.assertEqual(
            result.recommended_action,
            MitigationAction.NO_OP
        )

    def test_02_elevated_risk_monitoring(self):
        result = self.engine.decide(
            make_state(risk=0.20)
        )
        self.assertEqual(result.risk_level, "ELEVATED")
        self.assertEqual(
            result.recommended_action,
            MitigationAction.RECHECK_SOON
        )

    def test_03_high_risk_needs_approval(self):
        result = self.engine.decide(
            make_state(risk=0.65)
        )
        self.assertEqual(
            result.proposed_action,
            MitigationAction.MIGRATE_DATA
        )
        self.assertEqual(
            result.decision_status,
            "PENDING_ADMIN_APPROVAL"
        )
        self.assertEqual(
            result.recommended_action,
            MitigationAction.RECHECK_SOON
        )

    def test_04_approved_migration(self):
        result = self.engine.decide(
            make_state(risk=0.65, admin_approved=True)
        )
        self.assertEqual(
            result.decision_status,
            "ELIGIBLE_DRY_RUN"
        )
        self.assertEqual(
            result.recommended_action,
            MitigationAction.MIGRATE_DATA
        )

    def test_05_insufficient_capacity_blocks_migration(self):
        result = self.engine.decide(
            make_state(risk=0.70, free_capacity_gb=100)
        )
        self.assertEqual(
            result.decision_status,
            "BLOCKED_BY_SAFETY"
        )
        self.assertIn(
            "Insufficient destination capacity",
            result.safety.violations
        )

    def test_06_unhealthy_destination_blocks_migration(self):
        result = self.engine.decide(
            make_state(risk=0.70, target_healthy=False)
        )
        self.assertEqual(
            result.decision_status,
            "BLOCKED_BY_SAFETY"
        )

    def test_07_critical_drive_isolation_needs_approval(self):
        result = self.engine.decide(
            make_state(
                risk=0.95,
                migration_progress_fraction=1.0
            )
        )
        self.assertEqual(result.risk_level, "CRITICAL")
        self.assertEqual(
            result.proposed_action,
            MitigationAction.ISOLATE_DRIVE
        )
        self.assertEqual(
            result.decision_status,
            "PENDING_ADMIN_APPROVAL"
        )

    def test_08_critical_drive_isolation_approved(self):
        result = self.engine.decide(
            make_state(
                risk=0.95,
                migration_progress_fraction=1.0,
                admin_approved=True
            )
        )
        self.assertEqual(
            result.recommended_action,
            MitigationAction.ISOLATE_DRIVE
        )
        self.assertEqual(
            result.decision_status,
            "ELIGIBLE_DRY_RUN"
        )

    def test_09_isolation_blocked_without_redundancy(self):
        result = self.engine.decide(
            make_state(
                risk=0.95,
                migration_progress_fraction=1.0,
                healthy_redundant_copies=1,
                admin_approved=True
            )
        )
        self.assertEqual(
            result.decision_status,
            "BLOCKED_BY_SAFETY"
        )

    def test_10_critical_rul_overrides_low_risk(self):
        result = self.engine.decide(
            make_state(risk=0.05, rul=2.0)
        )
        self.assertEqual(result.risk_level, "CRITICAL")

    def test_11_wrong_prediction_horizon_rejected(self):
        with self.assertRaises(ValueError):
            self.engine.decide(
                make_state(horizon=30)
            )

    def test_12_missing_rul_supported(self):
        result = self.engine.decide(
            make_state(rul=None)
        )
        self.assertEqual(result.risk_level, "LOW")
        self.assertIn("unavailable", result.explanation)

    def test_13_invalid_policy_thresholds_rejected(self):
        with self.assertRaises(ValueError):
            DecisionConfig(
                elevated_risk=0.50,
                high_risk=0.30,
                critical_risk=0.85
            )

    def test_14_invalid_input_state_rejected(self):
        with self.assertRaises(TypeError):
            self.engine.decide({})

    def test_15_high_io_blocks_migration(self):
        result = self.engine.decide(
            make_state(risk=0.65, io_utilization=0.95)
        )
        self.assertEqual(
            result.decision_status,
            "BLOCKED_BY_SAFETY"
        )
        self.assertIn(
            "I/O utilization exceeds migration budget",
            result.safety.violations
        )


if __name__ == "__main__":
    unittest.main()
