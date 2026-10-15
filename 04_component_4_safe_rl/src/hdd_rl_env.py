
"""
Component 4 - Safe HDD Reinforcement Learning Environment

Gymnasium-compatible research simulator.

Research status:
- Synthetic HDD degradation and failure dynamics.
- Seven discrete mitigation actions.
- Independent safety shield integration.
- Reproducible random seeds.
- No real storage commands or physical HDD operations.

IMPORTANT:
The synthetic risk and failure dynamics are provisional.
This environment is not a validated Digital Twin.
"""

from datetime import datetime, timedelta, timezone

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from contracts import (
    DriveState,
    StorageContext,
    MitigationAction,
)
from safety_validator import SafetyValidator


class SafeHDDEnv(gym.Env):

    metadata = {"render_modes": []}

    ACTIONS = tuple(MitigationAction)

    def __init__(self, episode_days=30):

        super().__init__()

        if (
            type(episode_days) is not int
            or episode_days <= 0
        ):
            raise ValueError("episode_days must be positive")

        self.episode_days = episode_days
        self.validator = SafetyValidator()

        # Seven possible mitigation proposals.
        self.action_space = spaces.Discrete(
            len(self.ACTIONS)
        )

        # Ten normalized state measurements.
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(10,),
            dtype=np.float32,
        )

        self.day = 0
        self._done = True

    # --------------------------------------------------------
    # 1. Build normalized RL observations
    # --------------------------------------------------------

    def _get_observation(self):

        values = [
            self.risk,
            self.rul_days / 90.0,
            self.free_capacity_gb / 2000.0,
            self.data_to_migrate_gb / 1000.0,
            self.redundant_copies / 3.0,
            float(self.target_healthy),
            self.migration_progress,
            self.io_utilization,
            float(self.admin_approved),
            float(self.isolated),
        ]

        return np.clip(
            np.asarray(values, dtype=np.float32),
            0.0,
            1.0
        )

    # --------------------------------------------------------
    # 2. Convert simulator state to existing safety contracts
    # --------------------------------------------------------

    def _get_drive_state(self):

        timestamp = (
            datetime(2026, 1, 1, tzinfo=timezone.utc)
            + timedelta(days=self.day)
        )

        storage = StorageContext(
            free_capacity_gb=float(self.free_capacity_gb),
            data_to_migrate_gb=float(self.data_to_migrate_gb),
            healthy_redundant_copies=int(
                self.redundant_copies
            ),
            target_healthy=bool(self.target_healthy),
            migration_progress_fraction=float(
                self.migration_progress
            ),
            io_utilization=float(self.io_utilization),
            admin_approved=bool(self.admin_approved),
        )

        return DriveState(
            disk_id="SIMULATED_HDD_001",
            observed_at=timestamp.isoformat(),
            failure_probability=float(self.risk),
            prediction_horizon_days=7,
            rul_days=float(self.rul_days),
            storage=storage,
        )

    # --------------------------------------------------------
    # 3. Reset the environment
    # --------------------------------------------------------

    def reset(self, *, seed=None, options=None):

        super().reset(seed=seed)

        rng = self.np_random

        self.day = 0
        self._done = False

        self.risk = float(
            rng.uniform(0.05, 0.85)
        )

        self.rul_days = float(
            rng.uniform(5.0, 80.0)
        )

        self.free_capacity_gb = float(
            rng.uniform(200.0, 1500.0)
        )

        self.data_to_migrate_gb = float(
            rng.uniform(100.0, 700.0)
        )

        self.redundant_copies = int(
            rng.integers(0, 4)
        )

        self.target_healthy = bool(
            rng.random() < 0.90
        )

        self.migration_progress = 0.0

        self.io_utilization = float(
            rng.uniform(0.20, 0.95)
        )

        # Synthetic approval condition for experiments.
        # This does NOT represent real authorization.
        self.admin_approved = bool(
            rng.random() < 0.70
        )

        self.isolated = False

        observation = self._get_observation()

        return observation, {
            "simulation_only": True,
            "day": self.day,
        }

    # --------------------------------------------------------
    # 4. Execute one SIMULATED decision step
    # --------------------------------------------------------

    def step(self, action):

        if self._done:
            raise RuntimeError(
                "Episode finished. Call reset() first."
            )

        if not self.action_space.contains(action):
            raise ValueError("Invalid mitigation action")

        proposed = self.ACTIONS[int(action)]

        state = self._get_drive_state()

        assessment = self.validator.assess(
            state,
            proposed
        )

        blocked = (
            not assessment.technically_eligible
            or assessment.approval_pending
        )

        # Safety shield decides which action is applied.
        if blocked:
            applied = assessment.fallback_action
        else:
            applied = proposed

        # ----------------------------------------------------
        # 5. Reward shaping and simulated action effects
        # ----------------------------------------------------

        reward = 0.20

        if blocked:
            reward -= 2.0

        if applied == MitigationAction.RECHECK_SOON:
            reward -= 0.05

        elif applied == MitigationAction.MIGRATE_DATA:

            old_progress = self.migration_progress

            self.migration_progress = min(
                1.0,
                old_progress + 0.25
            )

            moved_fraction = (
                self.migration_progress - old_progress
            )

            moved_gb = (
                self.data_to_migrate_gb * moved_fraction
            )

            self.free_capacity_gb -= moved_gb

            reward -= (
                0.40 + 0.001 * moved_gb
            )

        elif applied == MitigationAction.ISOLATE_DRIVE:

            self.isolated = True
            reward -= 0.40

        elif applied == MitigationAction.RECOMMEND_REPLACEMENT:

            # Advisory only; no replacement is executed.
            reward -= 0.10

        # ----------------------------------------------------
        # 6. Synthetic HDD degradation
        # ----------------------------------------------------

        noise = float(
            self.np_random.normal(0.0, 0.003)
        )

        self.risk = float(
            np.clip(
                self.risk + 0.01 + noise,
                0.0,
                1.0
            )
        )

        self.rul_days = max(
            0.0,
            self.rul_days - 1.0
        )

        # Heuristic DAILY failure hazard.
        # NOT a calibrated real-world failure probability.
        daily_failure_hazard = float(
            np.clip(
                0.002 + 0.12 * self.risk**2,
                0.0,
                0.5
            )
        )

        failed = bool(
            self.np_random.random()
            < daily_failure_hazard
        )

        
        # ----------------------------------------------------
        # Synthetic mitigation consequence model
        # ----------------------------------------------------
        # Migration reduces simulated failure consequences,
        # NOT the probability of physical drive failure.
        #
        # 90% protection effectiveness is a provisional
        # research assumption requiring sensitivity analysis.

        protection_effectiveness = 0.90

        unprotected_fraction = (
            1.0
            - protection_effectiveness
            * self.migration_progress
        )

        if self.isolated:
            failure_consequence = 3.0
        else:
            failure_consequence = (
                40.0 * unprotected_fraction
            )

        if failed:
            reward -= failure_consequence


        # ----------------------------------------------------
        # 7. Episode termination
        # ----------------------------------------------------

        self.day += 1

        terminated = failed

        truncated = (
            self.day >= self.episode_days
            and not terminated
        )

        self._done = terminated or truncated

        info = {
            "simulation_only": True,
            "day": self.day,
            "proposed_action": proposed.value,
            "applied_action": applied.value,
            "blocked_by_safety": bool(blocked),
            "safety_violations": list(
                assessment.violations
            ),
            "approval_pending": bool(
                assessment.approval_pending
            ),
            "failed": failed,
            "isolated": bool(self.isolated),
            "daily_failure_hazard": daily_failure_hazard,
            "failure_consequence_if_failed": float(failure_consequence),
        }

        return (
            self._get_observation(),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )
