
"""
Component 4 - Independent Safety Validator Tests

Research objective:
Verify that unsafe HDD mitigation proposals are rejected
and that administrator approval is correctly identified.

Mode: Dry-run simulation only.
"""

import sys
import unittest
from pathlib import Path

COMPONENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMPONENT_ROOT / "src"))

from contracts import DriveState, StorageContext, MitigationAction
from safety_validator import SafetyValidator


def make_state(**storage_overrides):
    """Create a reproducible simulated HDD scenario."""

    storage_values = {
        "free_capacity_gb": 1000.0,
        "data_to_migrate_gb": 500.0,
        "healthy_redundant_copies": 2,
        "target_healthy": True,
        "migration_progress_fraction": 0.0,
        "io_utilization": 0.50,
        "admin_approved": False,
    }

    storage_values.update(storage_overrides)

    return DriveState(
        disk_id="TEST_HDD_001",
        observed_at="2026-10-14T13:00:00+05:30",
        failure_probability=0.80,
        prediction_horizon_days=7,
        rul_days=5.0,
        storage=StorageContext(**storage_values)
    )


class TestSafetyValidator(unittest.TestCase):

    def setUp(self):
        self.validator = SafetyValidator()

    def test_01_safe_migration_pending_approval(self):
        result = self.validator.assess(
            make_state(),
            MitigationAction.MIGRATE_DATA
        )
        self.assertTrue(result.technically_eligible)
        self.assertTrue(result.approval_pending)

    def test_02_approved_migration(self):
        result = self.validator.assess(
            make_state(admin_approved=True),
            MitigationAction.MIGRATE_DATA
        )
        self.assertTrue(result.technically_eligible)
        self.assertFalse(result.approval_pending)

    def test_03_insufficient_capacity(self):
        result = self.validator.assess(
            make_state(free_capacity_gb=100),
            MitigationAction.MIGRATE_DATA
        )
        self.assertFalse(result.technically_eligible)
        self.assertIn(
            "Insufficient destination capacity",
            result.violations
        )

    def test_04_unhealthy_destination(self):
        result = self.validator.assess(
            make_state(target_healthy=False),
            MitigationAction.MIGRATE_DATA
        )
        self.assertFalse(result.technically_eligible)

    def test_05_insufficient_migration_redundancy(self):
        result = self.validator.assess(
            make_state(healthy_redundant_copies=0),
            MitigationAction.MIGRATE_DATA
        )
        self.assertFalse(result.technically_eligible)

    def test_06_migration_io_budget_exceeded(self):
        result = self.validator.assess(
            make_state(io_utilization=0.95),
            MitigationAction.MIGRATE_DATA
        )
        self.assertFalse(result.technically_eligible)

    def test_07_no_remaining_migration_data(self):
        result = self.validator.assess(
            make_state(migration_progress_fraction=1.0),
            MitigationAction.MIGRATE_DATA
        )
        self.assertFalse(result.technically_eligible)

    def test_08_isolation_blocked_before_migration(self):
        result = self.validator.assess(
            make_state(migration_progress_fraction=0.50),
            MitigationAction.ISOLATE_DRIVE
        )
        self.assertFalse(result.technically_eligible)

    def test_09_isolation_requires_redundancy(self):
        result = self.validator.assess(
            make_state(
                migration_progress_fraction=1.0,
                healthy_redundant_copies=1
            ),
            MitigationAction.ISOLATE_DRIVE
        )
        self.assertFalse(result.technically_eligible)

    def test_10_isolation_allowed_with_approval(self):
        result = self.validator.assess(
            make_state(
                migration_progress_fraction=1.0,
                admin_approved=True
            ),
            MitigationAction.ISOLATE_DRIVE
        )
        self.assertTrue(result.technically_eligible)
        self.assertFalse(result.approval_pending)

    def test_11_throttling_blocked_without_workload_proof(self):
        result = self.validator.assess(
            make_state(),
            MitigationAction.THROTTLE_BACKGROUND_IO
        )
        self.assertFalse(result.technically_eligible)

    def test_12_rebalancing_blocked_without_target_proof(self):
        result = self.validator.assess(
            make_state(),
            MitigationAction.REBALANCE_WORKLOAD
        )
        self.assertFalse(result.technically_eligible)

    def test_13_no_op_is_eligible(self):
        result = self.validator.assess(
            make_state(),
            MitigationAction.NO_OP
        )
        self.assertTrue(result.technically_eligible)
        self.assertFalse(result.requires_admin_approval)

    def test_14_replacement_recommendation_is_eligible(self):
        result = self.validator.assess(
            make_state(),
            MitigationAction.RECOMMEND_REPLACEMENT
        )
        self.assertTrue(result.technically_eligible)

    def test_15_invalid_action_rejected(self):
        with self.assertRaises(TypeError):
            self.validator.assess(
                make_state(),
                "DELETE_ALL_DATA"
            )

    def test_16_invalid_state_rejected(self):
        with self.assertRaises(TypeError):
            self.validator.assess(
                {},
                MitigationAction.MIGRATE_DATA
            )


if __name__ == "__main__":
    unittest.main()
