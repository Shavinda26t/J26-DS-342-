
"""
Component 4 - Evolving RL Reward Model Tests

Research purpose:
Verify the synthetic failure-consequence assumptions
used by the Safe HDD reinforcement-learning simulator.

No actual storage operations are performed.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from contracts import MitigationAction
from hdd_rl_env import SafeHDDEnv


class TestRewardConsequences(unittest.TestCase):

    def run_scenario(self, progress, isolated=False):
        """Evaluate a reproducible simulator scenario."""

        env = SafeHDDEnv(episode_days=30)

        try:
            env.reset(seed=123)

            env.migration_progress = progress
            env.isolated = isolated

            action = env.ACTIONS.index(
                MitigationAction.NO_OP
            )

            _, reward, _, _, info = env.step(action)

            return reward, info

        finally:
            env.close()

    def test_01_migration_reduces_failure_consequences(self):

        scenarios = [
            (0.00, 40.0),
            (0.25, 31.0),
            (0.50, 22.0),
            (0.75, 13.0),
            (1.00, 4.0),
        ]

        for progress, expected in scenarios:

            with self.subTest(progress=progress):

                reward, info = self.run_scenario(
                    progress=progress
                )

                self.assertAlmostEqual(
                    info["failure_consequence_if_failed"],
                    expected,
                    places=5
                )

                expected_reward = (
                -expected
                if info["failed"]
                else 0.0
                )

                self.assertAlmostEqual(
                    reward,
                    expected_reward,
                    places=5
                )

    def test_02_isolation_limits_failure_consequences(self):

        reward, info = self.run_scenario(
            progress=0.0,
            isolated=True
        )

        self.assertAlmostEqual(
            info["failure_consequence_if_failed"],
            3.0
        )

        expected_reward = (
             -3.0
            if info["failed"]
            else 0.0
        )

        self.assertAlmostEqual(
            reward,
            expected_reward
        )

    def test_03_migration_does_not_change_failure_hazard(self):

        _, before = self.run_scenario(
            progress=0.0
        )

        _, after = self.run_scenario(
            progress=1.0
        )

        self.assertAlmostEqual(
            before["daily_failure_hazard"],
            after["daily_failure_hazard"]
        )

        self.assertEqual(
            before["failed"],
            after["failed"]
        )

        self.assertGreater(
            before["failure_consequence_if_failed"],
            after["failure_consequence_if_failed"]
        )


    def test_04_no_op_has_no_passive_reward(self):
        """Doing nothing must not earn a free survival bonus."""

        reward, info = self.run_scenario(
            progress=0.0,
            isolated=False
        )

        expected = (
            -40.0 if info["failed"] else 0.0
        )

        self.assertAlmostEqual(
            reward,
            expected,
            places=5
        )


if __name__ == "__main__":
    unittest.main()
