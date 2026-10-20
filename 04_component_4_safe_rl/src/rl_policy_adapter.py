
"""
Step 170 - Trained DQN Inference Adapter

Loads the real trained DQN model from Step 153.

Converts HDD decision inputs into the exact 10-feature
observation format used by Simulator V2.

Predicts a mitigation action and applies the existing
independent safety validator.

Research prototype. No physical HDD operations.
"""

from pathlib import Path

import numpy as np
from stable_baselines3 import DQN

from contracts import (
    DriveState,
    StorageContext,
    MitigationAction,
)
from safety_validator import SafetyValidator
from hdd_rl_env_v2 import SafeHDDEnvV2


COMPONENT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_PATH = (
    COMPONENT_ROOT
    / "models"
    / "153_dqn_v2_actionable_10000.zip"
)


class TrainedDQNSafePolicy:

    ACTIONS = SafeHDDEnvV2.ACTIONS

    def __init__(self, model_path=DEFAULT_MODEL_PATH):

        self.model_path = Path(model_path)

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Trained DQN checkpoint missing: "
                f"{self.model_path}"
            )

        self.model = DQN.load(
            str(self.model_path),
            device="cpu",
        )

        # Verify checkpoint compatibility.
        if self.model.observation_space.shape != (10,):
            raise ValueError(
                "DQN checkpoint has incorrect observation shape"
            )

        if self.model.action_space.n != len(self.ACTIONS):
            raise ValueError(
                "DQN checkpoint has incorrect action count"
            )

        self.validator = SafetyValidator()

    def build_observation(self, state: DriveState):
        """
        Match SafeHDDEnvV2._get_observation() exactly.

        Important:
        The existing API does not supply isolation status.
        We assume the drive is not already isolated.
        """

        if not isinstance(state, DriveState):
            raise TypeError("Expected DriveState")

        if state.prediction_horizon_days != 7:
            raise ValueError(
                "Trained DQN requires seven-day risk"
            )

        if state.rul_days is None:
            raise ValueError(
                "Trained DQN requires a RUL estimate"
            )

        storage = state.storage

        values = [
            state.failure_probability,
            state.rul_days / 90.0,
            storage.free_capacity_gb / 2000.0,
            storage.data_to_migrate_gb / 1000.0,
            storage.healthy_redundant_copies / 3.0,
            float(storage.target_healthy),
            storage.migration_progress_fraction,
            storage.io_utilization,
            float(storage.admin_approved),
            0.0,  # Assumed not already isolated
        ]

        observation = np.clip(
            np.asarray(values, dtype=np.float32),
            0.0,
            1.0,
        )

        return observation

    def decide(self, state: DriveState):
        """
        Generate an actual DQN prediction.

        The independent safety validator remains authoritative.
        """

        observation = self.build_observation(state)

        action_id, _ = self.model.predict(
            observation,
            deterministic=True,
        )

        action_id = int(
            np.asarray(action_id).item()
        )

        proposed = self.ACTIONS[action_id]

        assessment = self.validator.assess(
            state,
            proposed,
        )

        blocked = (
            not assessment.technically_eligible
            or assessment.approval_pending
        )

        if blocked:

            # Match the Step 155 filtering approach:
            # a blocked migration becomes NO_OP.
            if proposed == MitigationAction.MIGRATE_DATA:
                recommended = MitigationAction.NO_OP
            else:
                recommended = assessment.fallback_action

            if not assessment.technically_eligible:
                status = "BLOCKED_BY_SAFETY"
            else:
                status = "PENDING_ADMIN_APPROVAL"

        else:
            recommended = proposed
            status = "ELIGIBLE_DRY_RUN"

        explanation = (
            f"Trained DQN proposed {proposed.value}. "
            f"Independent safety assessment: {status}. "
            f"Final advisory action: {recommended.value}. "
            "No physical HDD operation was performed."
        )

        return {
            "disk_id": state.disk_id,
            "policy_name": (
                "DQN_V2_ACTIONABLE_153_SAFETY_FILTERED"
            ),
            "model_checkpoint": self.model_path.name,
            "proposed_action": proposed.value,
            "recommended_action": recommended.value,
            "decision_status": status,
            "explanation": explanation,
            "execution_mode": "DRY_RUN_ONLY",
            "filtered_by_safety": blocked,
            "safety": {
                "technically_eligible": (
                    assessment.technically_eligible
                ),
                "requires_admin_approval": (
                    assessment.requires_admin_approval
                ),
                "approval_pending": (
                    assessment.approval_pending
                ),
                "violations": list(
                    assessment.violations
                ),
                "fallback_action": (
                    assessment.fallback_action.value
                ),
            },
            "assumptions": [
                "Seven-day failure risk is supplied externally",
                "RUL estimate is supplied externally",
                "Isolation state assumed false",
                "Model trained on synthetic simulation",
            ],
        }


if __name__ == "__main__":

    print("=" * 60)
    print("STEP 170 - TRAINED DQN INFERENCE VERIFICATION")
    print("=" * 60)

    policy = TrainedDQNSafePolicy()

    storage = StorageContext(
        free_capacity_gb=1200.0,
        data_to_migrate_gb=500.0,
        healthy_redundant_copies=2,
        target_healthy=True,
        migration_progress_fraction=0.0,
        io_utilization=0.35,
        admin_approved=True,
    )

    state = DriveState(
        disk_id="DEMO_DQN_001",
        observed_at="2026-10-20T10:00:00+05:30",
        failure_probability=0.75,
        prediction_horizon_days=7,
        rul_days=10.0,
        storage=storage,
    )

    result = policy.decide(state)

    print("Model:", result["model_checkpoint"])
    print("Policy:", result["policy_name"])
    print("DQN proposed:", result["proposed_action"])
    print("Safe recommendation:", result["recommended_action"])
    print("Decision status:", result["decision_status"])
    print("Filtered:", result["filtered_by_safety"])
    print("Execution:", result["execution_mode"])

    print("\nSTEP 170 COMPLETE")
