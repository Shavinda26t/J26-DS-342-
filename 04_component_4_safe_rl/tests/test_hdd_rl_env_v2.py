
"""
Component 4 - SafeHDDEnv V2 Validation Tests

Purpose:
Validate the seven-day failure probability conversion
and reproducibility of the revised synthetic simulator.

Important:
These tests verify mathematical consistency, not
real-world HDD failure prediction accuracy.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from contracts import MitigationAction
from hdd_rl_env_v2 import SafeHDDEnvV2


class TestSafeHDDEnvV2(unittest.TestCase):

    def run_scenario(self, risk, seed=2026):

        env = SafeHDDEnvV2(episode_days=30)

        try:
            env.reset(seed=seed)

            env.risk = float(risk)

            action = env.ACTIONS.index(
                MitigationAction.NO_OP
            )

            observation, reward, terminated, truncated, info = (
                env.step(action)
            )

            return {
                "risk_after": env.risk,
                "daily_hazard": info["daily_failure_hazard"],
                "failed": info["failed"],
                "reward": reward,
                "observation": observation.copy()
            }

        finally:
            env.close()

    def test_01_seven_day_probability_consistency(self):

        for initial_risk in [
            0.0, 0.01, 0.25, 0.75, 1.0
        ]:

            with self.subTest(risk=initial_risk):

                result = self.run_scenario(initial_risk)

                # V2 updates synthetic risk before
                # calculating the daily hazard.
                risk = result["risk_after"]
                hazard = result["daily_hazard"]

                reconstructed_seven_day_risk = (
                    1.0 - (1.0 - hazard) ** 7
                )

                self.assertAlmostEqual(
                    reconstructed_seven_day_risk,
                    risk,
                    places=10
                )

    def test_02_hazard_is_valid_probability(self):

        for risk in [0.0, 0.1, 0.5, 0.9, 1.0]:

            result = self.run_scenario(risk)

            hazard = result["daily_hazard"]

            self.assertGreaterEqual(hazard, 0.0)
            self.assertLessEqual(hazard, 1.0)

    def test_03_higher_risk_means_higher_hazard(self):

        low = self.run_scenario(
            risk=0.10,
            seed=42
        )

        high = self.run_scenario(
            risk=0.80,
            seed=42
        )

        self.assertGreater(
            high["daily_hazard"],
            low["daily_hazard"]
        )

    def test_04_seeded_results_are_reproducible(self):

        first = self.run_scenario(
            risk=0.65,
            seed=123
        )

        second = self.run_scenario(
            risk=0.65,
            seed=123
        )

        self.assertEqual(
            first["daily_hazard"],
            second["daily_hazard"]
        )

        self.assertEqual(
            first["failed"],
            second["failed"]
        )

        self.assertEqual(
            first["reward"],
            second["reward"]
        )

        np.testing.assert_array_equal(
            first["observation"],
            second["observation"]
        )


    # --------------------------------------------------------
    # E102 - V2 EXPECTED-LOSS REWARD VALIDATION
    # --------------------------------------------------------

    def test_05_no_op_expected_loss_reward(self):
        """NO_OP reward equals negative expected failure loss."""

        env = SafeHDDEnvV2(episode_days=30)

        try:
            env.reset(seed=2026)

            action = env.ACTIONS.index(
                MitigationAction.NO_OP
            )

            _, reward, _, _, info = env.step(action)

            hazard = info["daily_failure_hazard"]
            consequence = info["failure_consequence_if_failed"]

            expected_loss = hazard * consequence

            self.assertEqual(
                info["reward_mode"],
                "EXPECTED_FAILURE_LOSS_V2"
            )

            self.assertAlmostEqual(
                info["expected_failure_loss"],
                expected_loss,
                places=8
            )

            self.assertAlmostEqual(
                reward,
                -expected_loss,
                places=8
            )

        finally:
            env.close()

    def test_06_migration_reward_includes_cost(self):
        """Approved migration pays its cost and expected loss."""

        env = SafeHDDEnvV2(episode_days=30)

        try:
            env.reset(seed=2026)

            env.free_capacity_gb = 1000.0
            env.data_to_migrate_gb = 200.0
            env.redundant_copies = 2
            env.target_healthy = True
            env.migration_progress = 0.0
            env.io_utilization = 0.40
            env.admin_approved = True

            action = env.ACTIONS.index(
                MitigationAction.MIGRATE_DATA
            )

            _, reward, _, _, info = env.step(action)

            self.assertFalse(
                info["blocked_by_safety"]
            )

            self.assertAlmostEqual(
                env.migration_progress,
                0.25
            )

            # 25% of 200 GB = 50 GB migrated.
            migration_cost = 0.40 + 0.001 * 50.0

            expected_reward = (
                -migration_cost
                -info["expected_failure_loss"]
            )

            self.assertAlmostEqual(
                reward,
                expected_reward,
                places=8
            )

        finally:
            env.close()

    def test_07_unapproved_migration_is_blocked(self):
        """Unapproved migration must not change migration state."""

        env = SafeHDDEnvV2(episode_days=30)

        try:
            env.reset(seed=2026)

            env.free_capacity_gb = 1000.0
            env.data_to_migrate_gb = 200.0
            env.redundant_copies = 2
            env.target_healthy = True
            env.migration_progress = 0.0
            env.io_utilization = 0.40
            env.admin_approved = False

            action = env.ACTIONS.index(
                MitigationAction.MIGRATE_DATA
            )

            _, reward, _, _, info = env.step(action)

            self.assertTrue(
                info["blocked_by_safety"]
            )

            self.assertTrue(
                info["approval_pending"]
            )

            self.assertEqual(
                info["applied_action"],
                "RECHECK_SOON"
            )

            self.assertAlmostEqual(
                env.migration_progress,
                0.0
            )

            # Block penalty: 2.0
            # Fallback monitoring cost: 0.05
            expected_reward = (
                -2.0
                -0.05
                -info["expected_failure_loss"]
            )

            self.assertAlmostEqual(
                reward,
                expected_reward,
                places=8
            )

        finally:
            env.close()


    # --------------------------------------------------------
    # E134 - Configurable synthetic risk dynamics
    # --------------------------------------------------------

    def test_08_stable_risk_without_drift_or_noise(self):
        """Risk must remain stable when both settings are zero."""

        for initial_risk in [0.02, 0.20, 0.50]:

            with self.subTest(initial_risk=initial_risk):

                env = SafeHDDEnvV2(
                    episode_days=30,
                    risk_drift_per_day=0.0,
                    risk_noise_std=0.0,
                )

                try:
                    env.reset(seed=2026)
                    env.risk = initial_risk

                    action = env.ACTIONS.index(
                        MitigationAction.NO_OP
                    )

                    _, _, _, _, info = env.step(action)

                    self.assertAlmostEqual(
                        env.risk,
                        initial_risk,
                        places=10,
                    )

                    expected_hazard = (
                        1.0
                        - (1.0 - initial_risk) ** (1.0 / 7.0)
                    )

                    self.assertAlmostEqual(
                        info["daily_failure_hazard"],
                        expected_hazard,
                        places=10,
                    )

                finally:
                    env.close()

    def test_09_configurable_drift_changes_risk(self):
        """Risk increases by the configured drift without noise."""

        env = SafeHDDEnvV2(
            episode_days=30,
            risk_drift_per_day=0.05,
            risk_noise_std=0.0,
        )

        try:
            env.reset(seed=2026)
            env.risk = 0.20

            action = env.ACTIONS.index(
                MitigationAction.NO_OP
            )

            env.step(action)

            self.assertAlmostEqual(
                env.risk,
                0.25,
                places=10,
            )

        finally:
            env.close()

    def test_10_invalid_risk_settings_rejected(self):
        """Invalid drift and noise parameters must be rejected."""

        invalid_values = [
            -0.01,
            float("nan"),
            float("inf"),
            True,
        ]

        for value in invalid_values:

            with self.subTest(drift=value):

                with self.assertRaises(ValueError):
                    SafeHDDEnvV2(
                        risk_drift_per_day=value
                    )

            with self.subTest(noise=value):

                with self.assertRaises(ValueError):
                    SafeHDDEnvV2(
                        risk_noise_std=value
                    )



if __name__ == "__main__":
    unittest.main()
