
"""
Step 185 - Component 4 Trained DQN API Integration Tests

Tests the actual FastAPI endpoint, request schema,
DQN adapter, and independent safety validator.

Only neural-network prediction is mocked.
No model training or physical HDD operations.
"""

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1.component4_decision import router
from app.services import component4_rl_service
from contracts import MitigationAction
from rl_policy_adapter import TrainedDQNSafePolicy


app = FastAPI()
app.include_router(
    router,
    prefix="/api/v1/component4",
)

ENDPOINT = "/api/v1/component4/rl-decide"


BASE_REQUEST = {
    "disk_id": "TEST_DQN_API_001",
    "observed_at": "2026-10-20T12:00:00+05:30",
    "failure_probability": 0.75,
    "prediction_horizon_days": 7,
    "rul_days": 10.0,
    "storage": {
        "free_capacity_gb": 1200.0,
        "data_to_migrate_gb": 500.0,
        "healthy_redundant_copies": 2,
        "target_healthy": True,
        "migration_progress_fraction": 0.0,
        "io_utilization": 0.35,
        "admin_approved": True,
    },
}


class TestComponent4TrainedDQNAPI(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        self.addCleanup(self.client.close)

        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

        self.checkpoint = (
            Path(self.temp_dir.name)
            / "mock_checkpoint.zip"
        )
        self.checkpoint.touch()

    def make_request(self):
        return deepcopy(BASE_REQUEST)

    def make_policy(
        self,
        action=MitigationAction.MIGRATE_DATA,
    ):
        """Use real adapter and validator, mock DQN.predict."""

        fake_model = Mock()
        fake_model.observation_space.shape = (10,)
        fake_model.action_space.n = 7

        action_index = (
            TrainedDQNSafePolicy.ACTIONS.index(action)
        )

        fake_model.predict.return_value = (
            np.asarray(action_index),
            None,
        )

        with patch(
            "rl_policy_adapter.DQN.load",
            return_value=fake_model,
        ):
            policy = TrainedDQNSafePolicy(
                model_path=self.checkpoint
            )

        return policy, fake_model

    def post_with_policy(self, payload, policy):
        with patch.object(
            component4_rl_service,
            "get_trained_dqn",
            return_value=policy,
        ):
            return self.client.post(
                ENDPOINT,
                json=payload,
            )

    def test_01_approved_migration_http_200(self):
        policy, fake_model = self.make_policy()

        response = self.post_with_policy(
            self.make_request(),
            policy,
        )

        self.assertEqual(response.status_code, 200)

        body = response.json()

        self.assertEqual(
            body["proposed_action"],
            "MIGRATE_DATA",
        )
        self.assertEqual(
            body["recommended_action"],
            "MIGRATE_DATA",
        )
        self.assertEqual(
            body["decision_status"],
            "ELIGIBLE_DRY_RUN",
        )
        self.assertEqual(
            body["execution_mode"],
            "DRY_RUN_ONLY",
        )
        self.assertFalse(body["filtered_by_safety"])

        # Verify HTTP input reaches the model correctly.
        observation = fake_model.predict.call_args.args[0]

        self.assertAlmostEqual(
            float(observation[0]),
            0.75,
            places=6,
        )
        self.assertAlmostEqual(
            float(observation[1]),
            10.0 / 90.0,
            places=6,
        )

    def test_02_missing_approval_http_200(self):
        payload = self.make_request()
        payload["storage"]["admin_approved"] = False

        policy, _ = self.make_policy()

        response = self.post_with_policy(
            payload, policy
        )

        self.assertEqual(response.status_code, 200)

        body = response.json()

        self.assertEqual(
            body["decision_status"],
            "PENDING_ADMIN_APPROVAL",
        )
        self.assertEqual(
            body["recommended_action"],
            "NO_OP",
        )
        self.assertTrue(body["filtered_by_safety"])
        self.assertTrue(
            body["safety"]["approval_pending"]
        )

    def test_03_insufficient_capacity_http_200(self):
        payload = self.make_request()
        payload["storage"]["free_capacity_gb"] = 100.0

        policy, _ = self.make_policy()

        response = self.post_with_policy(
            payload, policy
        )

        self.assertEqual(response.status_code, 200)

        body = response.json()

        self.assertEqual(
            body["decision_status"],
            "BLOCKED_BY_SAFETY",
        )
        self.assertEqual(
            body["recommended_action"],
            "NO_OP",
        )
        self.assertIn(
            "Insufficient destination capacity",
            body["safety"]["violations"],
        )

    def test_04_invalid_risk_http_422(self):
        payload = self.make_request()
        payload["failure_probability"] = 1.20

        response = self.client.post(
            ENDPOINT,
            json=payload,
        )

        self.assertEqual(response.status_code, 422)

    def test_05_wrong_prediction_horizon_http_422(self):
        payload = self.make_request()
        payload["prediction_horizon_days"] = 30

        response = self.client.post(
            ENDPOINT,
            json=payload,
        )

        self.assertEqual(response.status_code, 422)

    def test_06_missing_rul_http_422(self):
        payload = self.make_request()
        payload["rul_days"] = None

        response = self.client.post(
            ENDPOINT,
            json=payload,
        )

        self.assertEqual(response.status_code, 422)

    def test_07_missing_checkpoint_http_503(self):
        with patch.object(
            component4_rl_service,
            "get_trained_dqn",
            side_effect=FileNotFoundError(
                "Checkpoint unavailable"
            ),
        ):
            response = self.client.post(
                ENDPOINT,
                json=self.make_request(),
            )

        self.assertEqual(response.status_code, 503)

        self.assertIn(
            "Trained DQN is unavailable",
            response.json()["detail"],
        )

    def test_08_missing_timezone_http_422(self):
        payload = self.make_request()
        payload["observed_at"] = "2026-10-20T12:00:00"

        response = self.client.post(
            ENDPOINT,
            json=payload,
        )

        self.assertEqual(response.status_code, 422)

    def test_09_no_op_prediction_preserved(self):
        policy, _ = self.make_policy(
            action=MitigationAction.NO_OP
        )

        response = self.post_with_policy(
            self.make_request(),
            policy,
        )

        self.assertEqual(response.status_code, 200)

        body = response.json()

        self.assertEqual(
            body["proposed_action"],
            "NO_OP",
        )
        self.assertEqual(
            body["recommended_action"],
            "NO_OP",
        )
        self.assertFalse(
            body["filtered_by_safety"]
        )


if __name__ == "__main__":
    unittest.main()
