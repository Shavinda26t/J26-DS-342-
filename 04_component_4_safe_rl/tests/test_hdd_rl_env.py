
"""
Component 4 - Safe HDD RL Environment Tests

Objectives:
- Verify Gymnasium state transitions.
- Verify reproducible seeded experiments.
- Verify the independent safety shield.
- Verify simulated mitigation outcomes.

All tests use synthetic data.
No physical HDD operations are performed.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from contracts import MitigationAction
from hdd_rl_env import SafeHDDEnv


def action_id(env, action):
    """Find the index of a mitigation action."""
    return env.ACTIONS.index(action)


def configure_safe_migration(env):
    """Set reproducible conditions for migration."""

    env.free_capacity_gb = 1000.0
    env.data_to_migrate_gb = 200.0
    env.redundant_copies = 2
    env.target_healthy = True
    env.migration_progress = 0.0
    env.io_utilization = 0.40
    env.admin_approved = True


class TestSafeHDDEnvironment(unittest.TestCase):

    def setUp(self):
        self.env = SafeHDDEnv(episode_days=30)
        self.env.reset(seed=42)

    def tearDown(self):
        self.env.close()

    def test_01_seven_actions_available(self):
        self.assertEqual(self.env.action_space.n, 7)
        self.assertEqual(len(self.env.ACTIONS), 7)

    def test_02_observation_shape_and_type(self):
        observation, _ = self.env.reset(seed=42)

        self.assertEqual(observation.shape, (10,))
        self.assertEqual(observation.dtype, np.float32)
        self.assertTrue(
            self.env.observation_space.contains(observation)
        )

    def test_03_seeded_reset_is_reproducible(self):
        first, _ = self.env.reset(seed=123)
        second, _ = self.env.reset(seed=123)

        np.testing.assert_array_equal(first, second)

    def test_04_step_returns_valid_output(self):
        action = action_id(
            self.env, MitigationAction.NO_OP
        )

        obs, reward, terminated, truncated, info = (
            self.env.step(action)
        )

        self.assertTrue(
            self.env.observation_space.contains(obs)
        )
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertTrue(info["simulation_only"])

    def test_05_invalid_action_rejected(self):
        with self.assertRaises(ValueError):
            self.env.step(999)

    def test_06_migration_requires_approval(self):
        configure_safe_migration(self.env)
        self.env.admin_approved = False

        action = action_id(
            self.env, MitigationAction.MIGRATE_DATA
        )

        _, _, _, _, info = self.env.step(action)

        self.assertTrue(info["blocked_by_safety"])
        self.assertTrue(info["approval_pending"])
        self.assertEqual(
            info["applied_action"], "RECHECK_SOON"
        )
        self.assertEqual(
            self.env.migration_progress, 0.0
        )

    def test_07_insufficient_capacity_blocks_migration(self):
        configure_safe_migration(self.env)
        self.env.free_capacity_gb = 10.0

        action = action_id(
            self.env, MitigationAction.MIGRATE_DATA
        )

        _, _, _, _, info = self.env.step(action)

        self.assertTrue(info["blocked_by_safety"])
        self.assertIn(
            "Insufficient destination capacity",
            info["safety_violations"]
        )
        self.assertEqual(
            self.env.migration_progress, 0.0
        )

    def test_08_approved_migration_changes_state(self):
        configure_safe_migration(self.env)

        action = action_id(
            self.env, MitigationAction.MIGRATE_DATA
        )

        _, _, _, _, info = self.env.step(action)

        self.assertFalse(info["blocked_by_safety"])
        self.assertEqual(
            info["applied_action"], "MIGRATE_DATA"
        )
        self.assertAlmostEqual(
            self.env.migration_progress, 0.25
        )
        self.assertAlmostEqual(
            self.env.free_capacity_gb, 950.0
        )

    def test_09_incomplete_migration_blocks_isolation(self):
        configure_safe_migration(self.env)
        self.env.migration_progress = 0.50

        action = action_id(
            self.env, MitigationAction.ISOLATE_DRIVE
        )

        _, _, _, _, info = self.env.step(action)

        self.assertTrue(info["blocked_by_safety"])
        self.assertFalse(self.env.isolated)

    def test_10_isolation_requires_redundancy(self):
        configure_safe_migration(self.env)
        self.env.migration_progress = 1.0
        self.env.redundant_copies = 1

        action = action_id(
            self.env, MitigationAction.ISOLATE_DRIVE
        )

        _, _, _, _, info = self.env.step(action)

        self.assertTrue(info["blocked_by_safety"])
        self.assertFalse(self.env.isolated)

    def test_11_safe_isolation_changes_state(self):
        configure_safe_migration(self.env)
        self.env.migration_progress = 1.0

        action = action_id(
            self.env, MitigationAction.ISOLATE_DRIVE
        )

        _, _, _, _, info = self.env.step(action)

        self.assertFalse(info["blocked_by_safety"])
        self.assertTrue(self.env.isolated)

    def test_12_episode_terminates_at_time_limit(self):
        env = SafeHDDEnv(episode_days=1)

        try:
            env.reset(seed=42)

            action = action_id(
                env, MitigationAction.NO_OP
            )

            _, _, terminated, truncated, _ = (
                env.step(action)
            )

            self.assertTrue(terminated or truncated)

            with self.assertRaises(RuntimeError):
                env.step(action)

        finally:
            env.close()

    def test_13_rollout_reproducible_with_seed(self):
        def rollout(seed):
            env = SafeHDDEnv(episode_days=10)

            try:
                observation, _ = env.reset(seed=seed)
                rewards = []

                for _ in range(10):
                    action = action_id(
                        env, MitigationAction.NO_OP
                    )

                    observation, reward, done, truncated, _ = (
                        env.step(action)
                    )

                    rewards.append(reward)

                    if done or truncated:
                        break

                return observation, rewards
            finally:
                env.close()

        obs1, rewards1 = rollout(100)
        obs2, rewards2 = rollout(100)

        np.testing.assert_array_equal(obs1, obs2)
        self.assertEqual(rewards1, rewards2)


    def test_14_repeated_isolation_is_idempotent(self):
        """Already-isolated drives must not incur a second cost."""

        configure_safe_migration(self.env)

        self.env.migration_progress = 1.0
        self.env.isolated = True

        action = action_id(
            self.env,
            MitigationAction.ISOLATE_DRIVE
        )

        _, reward, _, _, info = self.env.step(action)

        self.assertFalse(info["blocked_by_safety"])

        self.assertEqual(
            info["applied_action"],
            "NO_OP"
        )

        self.assertTrue(self.env.isolated)

        expected_reward = (
            -info["failure_consequence_if_failed"]
            if info["failed"]
            else 0.0
        )

        self.assertAlmostEqual(
            reward,
            expected_reward
        )
    

if __name__ == "__main__":
    unittest.main()
