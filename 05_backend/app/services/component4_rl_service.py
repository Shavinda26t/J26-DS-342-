
"""
Step 171 - Trained DQN FastAPI Service

Connects validated API inputs to the actual trained
DQN checkpoint and independent safety validator.

The model is loaded lazily and reused between requests.

Research prototype - no real HDD operations.
"""

from functools import lru_cache

from fastapi import HTTPException

from app.schemas.component4_decision import HDDDecisionRequest

# Importing the existing service also registers the
# Component 4 src directory on Python's import path.
from app.services.component4_decision_service import (
    COMPONENT4_SRC,
)

from contracts import DriveState, StorageContext


@lru_cache(maxsize=1)
def get_trained_dqn():
    """Load the real DQN checkpoint only when requested."""

    from rl_policy_adapter import TrainedDQNSafePolicy

    return TrainedDQNSafePolicy()


def generate_trained_rl_recommendation(
    request: HDDDecisionRequest,
) -> dict:
    """Run actual DQN inference with independent safety checks."""

    if request.rul_days is None:
        raise HTTPException(
            status_code=422,
            detail="A RUL estimate is required for DQN inference.",
        )

    storage = StorageContext(
        free_capacity_gb=request.storage.free_capacity_gb,
        data_to_migrate_gb=request.storage.data_to_migrate_gb,
        healthy_redundant_copies=(
            request.storage.healthy_redundant_copies
        ),
        target_healthy=request.storage.target_healthy,
        migration_progress_fraction=(
            request.storage.migration_progress_fraction
        ),
        io_utilization=request.storage.io_utilization,
        admin_approved=request.storage.admin_approved,
    )

    state = DriveState(
        disk_id=request.disk_id,
        observed_at=request.observed_at.isoformat(),
        failure_probability=request.failure_probability,
        prediction_horizon_days=(
            request.prediction_horizon_days
        ),
        rul_days=request.rul_days,
        storage=storage,
    )

    try:
        policy = get_trained_dqn()

    except (FileNotFoundError, ImportError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Trained DQN is unavailable. "
                "Check the model checkpoint and RL dependencies."
            ),
        ) from exc

    return policy.decide(state)
