
"""
Component 4 - Automated Input Contract Tests

Purpose:
Verify HDD decision inputs, operational constraints,
and mitigation action definitions.

Testing framework: Python unittest
Execution mode: Local, non-destructive
"""

import sys
import unittest
from pathlib import Path

# Allow this test file to import our source module.
COMPONENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMPONENT_ROOT / "src"))

from contracts import DriveState, StorageContext, MitigationAction


def make_storage(**overrides):
    """Create reusable test storage conditions."""
    values = {
        "free_capacity_gb": 1000.0,
        "data_to_migrate_gb": 500.0,
        "healthy_redundant_copies": 2,
        "target_healthy": True,
        "migration_progress_fraction": 0.0,
        "io_utilization": 0.65,
        "admin_approved": False,
    }
    values.update(overrides)
    return StorageContext(**values)


def make_drive(**overrides):
    """Create a representative simulated HDD state."""
    values = {
        "disk_id": "TEST_HDD_001",
        "observed_at": "2026-10-14T13:00:00+05:30",
        "failure_probability": 0.75,
        "prediction_horizon_days": 7,
        "rul_days": 8.0,
        "storage": make_storage(),
    }
    values.update(overrides)
    return DriveState(**values)


class TestHDDContracts(unittest.TestCase):

    def test_valid_drive_state(self):
        drive = make_drive()
        self.assertEqual(drive.disk_id, "TEST_HDD_001")
        self.assertEqual(drive.prediction_horizon_days, 7)

    def test_seven_mitigation_actions_exist(self):
        self.assertEqual(len(MitigationAction), 7)

    def test_invalid_failure_probability_rejected(self):
        with self.assertRaises(ValueError):
            make_drive(failure_probability=1.5)

    def test_negative_failure_probability_rejected(self):
        with self.assertRaises(ValueError):
            make_drive(failure_probability=-0.1)

    def test_missing_timezone_rejected(self):
        with self.assertRaises(ValueError):
            make_drive(observed_at="2026-10-14T13:00:00")

    def test_negative_rul_rejected(self):
        with self.assertRaises(ValueError):
            make_drive(rul_days=-5)

    def test_missing_rul_allowed(self):
        drive = make_drive(rul_days=None)
        self.assertIsNone(drive.rul_days)

    def test_insufficient_capacity_is_representable(self):
        storage = make_storage(
            free_capacity_gb=100,
            data_to_migrate_gb=500
        )
        self.assertLess(
            storage.free_capacity_gb,
            storage.data_to_migrate_gb
        )

    def test_negative_capacity_rejected(self):
        with self.assertRaises(ValueError):
            make_storage(free_capacity_gb=-100)

    def test_invalid_migration_progress_rejected(self):
        with self.assertRaises(ValueError):
            make_storage(migration_progress_fraction=1.5)

    def test_invalid_redundancy_rejected(self):
        with self.assertRaises(ValueError):
            make_storage(healthy_redundant_copies=-1)


if __name__ == "__main__":
    unittest.main()
