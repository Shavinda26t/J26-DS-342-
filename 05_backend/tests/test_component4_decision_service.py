
"""
Component 4 - Backend Decision Service Tests

Purpose:
Verify that backend API contracts integrate correctly
with the risk-aware decision engine and safety validator.

All scenarios are simulated.
No physical HDD operations are performed.
"""

import unittest
from copy import deepcopy

from pydantic import ValidationError

from app.schemas.component4_decision import HDDDecisionRequest
from app.services.component4_decision_service import (
    generate_safe_recommendation
)


BASE_REQUEST = {
    "disk_id": "DEMO_HDD_001",
    "observed_at": "2026-10-15T01:00:00+05:30",
    "failure_probability": 0.65,
    "prediction_horizon_days": 7,
    "rul_days": 10.0,
    "storage": {
        "free_capacity_gb": 1000.0,
        "data_to_migrate_gb": 500.0,
        "healthy_redundant_copies": 2,
        "target_healthy": True,
        "migration_progress_fraction": 0.0,
        "io_utilization": 0.40,
        "admin_approved": False
    }
}


class TestComponent4BackendService(unittest.TestCase):

    def make_request(self, **changes):
        data = deepcopy(BASE_REQUEST)
        data.update(changes)
        return HDDDecisionRequest(**data)

    def test_01_high_risk_requires_approval(self):
        result = generate_safe_recommendation(
            self.make_request()
        )

        self.assertEqual(
            result.proposed_action, "MIGRATE_DATA"
        )
        self.assertEqual(
            result.decision_status,
            "PENDING_ADMIN_APPROVAL"
        )
        self.assertEqual(
            result.recommended_action, "RECHECK_SOON"
        )

    def test_02_insufficient_capacity_blocks_migration(self):
        storage = deepcopy(BASE_REQUEST["storage"])
        storage["free_capacity_gb"] = 100.0

        result = generate_safe_recommendation(
            self.make_request(storage=storage)
        )

        self.assertEqual(
            result.decision_status, "BLOCKED_BY_SAFETY"
        )
        self.assertIn(
            "Insufficient destination capacity",
            result.safety.violations
        )

    def test_03_approved_migration(self):
        storage = deepcopy(BASE_REQUEST["storage"])
        storage["admin_approved"] = True

        result = generate_safe_recommendation(
            self.make_request(storage=storage)
        )

        self.assertEqual(
            result.decision_status, "ELIGIBLE_DRY_RUN"
        )
        self.assertEqual(
            result.recommended_action, "MIGRATE_DATA"
        )
        self.assertEqual(
            result.execution_mode, "DRY_RUN_ONLY"
        )

    def test_04_healthy_drive_no_action(self):
        result = generate_safe_recommendation(
            self.make_request(
                failure_probability=0.05,
                rul_days=100.0
            )
        )

        self.assertEqual(result.risk_level, "LOW")
        self.assertEqual(
            result.recommended_action, "NO_OP"
        )

    def test_05_wrong_prediction_horizon_rejected(self):
        with self.assertRaises(ValidationError):
            self.make_request(prediction_horizon_days=30)


if __name__ == "__main__":
    unittest.main()
