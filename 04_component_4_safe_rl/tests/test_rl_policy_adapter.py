
"""
Step 183 - Trained DQN Adapter Regression Tests

Validate:
- Model loading and compatibility
- Ten-feature observation mapping
- Safe migration
- Pending administrator approval
- Insufficient destination capacity
- Unsupported mitigation actions
- Invalid model inputs
- Missing/incompatible checkpoints

Uses a mocked DQN for fast, deterministic tests.

No retraining or real HDD operations.
"""

import sys
import tempfile
import unittest

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from contracts import (
    DriveState,
    StorageContext,
    MitigationAction,
)

from rl_policy_adapter import TrainedDQNSafePolicy


class TestTrainedDQNSafePolicy(unittest.TestCase):

    def setUp(self):

        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

        self.checkpoint = (
            Path(self.temp_dir.name)
            / "mock_dqn_checkpoint.zip"
        )

        # The checkpoint contents are mocked.
        self.checkpoint.touch()

    def make_state(self, **overrides):

        storage_values = {
            "free_capacity_gb": 1200.0,
            "data_to_migrate_gb": 500.0,
            "healthy_redundant_copies": 2,
            "target_healthy": True,
            "migration_progress_fraction": 0.0,
            "io_utilization": 0.35,
            "admin_approved": True,
        }

        state_values = {
            "disk_id": "TEST_DQN_001",
            "observed_at": "2026-10-20T12:00:00+05:30",
            "failure_probability": 0.75,
            "prediction_horizon_days": 7,
            "rul_days": 10.0,
        }

        for key, value in overrides.items():

            if key in storage_values:
                storage_values[key] = value

            elif key in state_values:
                state_values[key] = value

            else:
                raise ValueError(
                    f"Unknown test override: {key}"
                )

        return DriveState(
            **state_values,
            storage=StorageContext(**storage_values),
        )

    def make_policy(
        self,
        proposed_action=MitigationAction.MIGRATE_DATA,
    ):

        fake_model = Mock()

        fake_model.observation_space.shape = (10,)
        fake_model.action_space.n = 7

        action_id = (
            TrainedDQNSafePolicy.ACTIONS.index(
                proposed_action
            )
        )

        fake_model.predict.return_value = (
            np.asarray(action_id),
            None,
        )

        with patch(
            "rl_policy_adapter.DQN.load",
            return_value=fake_model,
        ) as mocked_load:

            policy = TrainedDQNSafePolicy(
                model_path=self.checkpoint
            )

        mocked_load.assert_called_once_with(
            str(self.checkpoint),
            device="cpu",
        )

        return policy, fake_model

    def test_01_model_loading(self):

        policy, fake_model = self.make_policy()

        self.assertEqual(
            policy.model_path,
            self.checkpoint,
        )

        self.assertIs(
            policy.model,
            fake_model,
        )

    def test_02_ten_feature_observation(self):

        policy, _ = self.make_policy()

        observation = policy.build_observation(
            self.make_state()
        )

        expected = np.array(
            [
                0.75,
                10.0 / 90.0,
                1200.0 / 2000.0,
                500.0 / 1000.0,
                2.0 / 3.0,
                1.0,
                0.0,
                0.35,
                1.0,
                0.0,
            ],
            dtype=np.float32,
        )

        self.assertEqual(
            observation.shape,
            (10,),
        )

        self.assertEqual(
            observation.dtype,
            np.float32,
        )

        np.testing.assert_allclose(
            observation,
            expected,
            rtol=1e-6,
            atol=1e-7,
        )

    def test_03_approved_migration(self):

        policy, fake_model = self.make_policy()

        result = policy.decide(
            self.make_state(admin_approved=True)
        )

        fake_model.predict.assert_called_once()

        self.assertEqual(
            result["proposed_action"],
            "MIGRATE_DATA",
        )

        self.assertEqual(
            result["recommended_action"],
            "MIGRATE_DATA",
        )

        self.assertEqual(
            result["decision_status"],
            "ELIGIBLE_DRY_RUN",
        )

        self.assertFalse(
            result["filtered_by_safety"]
        )

        self.assertEqual(
            result["execution_mode"],
            "DRY_RUN_ONLY",
        )

    def test_04_missing_approval_blocks_migration(self):

        policy, _ = self.make_policy()

        result = policy.decide(
            self.make_state(admin_approved=False)
        )

        self.assertEqual(
            result["proposed_action"],
            "MIGRATE_DATA",
        )

        self.assertEqual(
            result["recommended_action"],
            "NO_OP",
        )

        self.assertEqual(
            result["decision_status"],
            "PENDING_ADMIN_APPROVAL",
        )

        self.assertTrue(
            result["safety"]["approval_pending"]
        )

        self.assertTrue(
            result["filtered_by_safety"]
        )

    def test_05_insufficient_capacity_blocks_migration(self):

        policy, _ = self.make_policy()

        result = policy.decide(
            self.make_state(
                free_capacity_gb=100.0,
                data_to_migrate_gb=500.0,
            )
        )

        self.assertEqual(
            result["decision_status"],
            "BLOCKED_BY_SAFETY",
        )

        self.assertEqual(
            result["recommended_action"],
            "NO_OP",
        )

        self.assertFalse(
            result["safety"]["technically_eligible"]
        )

        self.assertIn(
            "Insufficient destination capacity",
            result["safety"]["violations"],
        )

    def test_06_unverified_throttling_rejected(self):

        policy, _ = self.make_policy(
            MitigationAction.THROTTLE_BACKGROUND_IO
        )

        result = policy.decide(
            self.make_state()
        )

        self.assertEqual(
            result["proposed_action"],
            "THROTTLE_BACKGROUND_IO",
        )

        self.assertEqual(
            result["decision_status"],
            "BLOCKED_BY_SAFETY",
        )

        self.assertEqual(
            result["recommended_action"],
            "RECHECK_SOON",
        )

        self.assertTrue(
            result["filtered_by_safety"]
        )

    def test_07_wrong_prediction_horizon_rejected(self):

        policy, _ = self.make_policy()

        with self.assertRaises(ValueError):

            policy.build_observation(
                self.make_state(
                    prediction_horizon_days=30
                )
            )

    def test_08_missing_rul_rejected(self):

        policy, _ = self.make_policy()

        with self.assertRaises(ValueError):

            policy.build_observation(
                self.make_state(rul_days=None)
            )

    def test_09_missing_checkpoint_rejected(self):

        missing_path = (
            Path(self.temp_dir.name)
            / "not_present.zip"
        )

        with self.assertRaises(FileNotFoundError):

            TrainedDQNSafePolicy(
                model_path=missing_path
            )

    def test_10_incompatible_checkpoint_rejected(self):

        fake_model = Mock()

        fake_model.observation_space.shape = (8,)
        fake_model.action_space.n = 7

        with patch(
            "rl_policy_adapter.DQN.load",
            return_value=fake_model,
        ):

            with self.assertRaises(ValueError):

                TrainedDQNSafePolicy(
                    model_path=self.checkpoint
                )


if __name__ == "__main__":
    unittest.main()
