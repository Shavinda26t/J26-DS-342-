
"""
Component 4: Safe HDD Mitigation Decision Service

Purpose:
Connect FastAPI request validation to the Component 4
decision engine and independent safety validator.

Execution mode: DRY_RUN_ONLY
No real HDD operations are performed.
"""

from pathlib import Path
import sys

from app.config import settings
from app.schemas.component4_decision import (
    HDDDecisionRequest,
    HDDDecisionResponse,
)

# ---------------------------------------------------------
# 1. Locate the independent Component 4 Python modules
# ---------------------------------------------------------

COMPONENT4_SRC = (
    settings.HDD_PROJECT_ROOT
    / "04_component_4_safe_rl"
    / "src"
).resolve()

required_files = [
    "contracts.py",
    "safety_validator.py",
    "decision_engine.py",
]

for filename in required_files:
    if not (COMPONENT4_SRC / filename).is_file():
        raise RuntimeError(
            f"Missing Component 4 module: {filename}"
        )

# Allow importing source modules from the component folder.
# This is a local development adapter. We can package the
# component formally before production deployment.

if str(COMPONENT4_SRC) not in sys.path:
    sys.path.insert(0, str(COMPONENT4_SRC))

from contracts import DriveState, StorageContext
from decision_engine import HDDDecisionEngine


# ---------------------------------------------------------
# 2. Initialize the decision engine
# ---------------------------------------------------------

decision_engine = HDDDecisionEngine()


# ---------------------------------------------------------
# 3. Convert API input into decision-engine input
# ---------------------------------------------------------

def generate_safe_recommendation(
    request: HDDDecisionRequest,
) -> HDDDecisionResponse:
    """
    Generate an explainable mitigation recommendation.

    Parameters
    ----------
    request:
        Validated FastAPI/Pydantic HDD decision request.

    Returns
    -------
    HDDDecisionResponse:
        Recommended action, safety assessment and explanation.
    """

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

    drive_state = DriveState(
        disk_id=request.disk_id,
        observed_at=request.observed_at.isoformat(),
        failure_probability=request.failure_probability,
        prediction_horizon_days=(
            request.prediction_horizon_days
        ),
        rul_days=request.rul_days,
        storage=storage,
    )

    # Run the policy and its independent safety shield.
    result = decision_engine.decide(drive_state)

    # -----------------------------------------------------
    # 4. Construct a validated response
    # -----------------------------------------------------

    return HDDDecisionResponse(
        disk_id=result.disk_id,
        risk_level=result.risk_level,
        proposed_action=result.proposed_action.value,
        recommended_action=result.recommended_action.value,
        decision_status=result.decision_status,
        explanation=result.explanation,
        policy_name=result.policy_name,
        execution_mode=result.execution_mode,
        safety={
            "proposed_action": (
                result.safety.proposed_action.value
            ),
            "technically_eligible": (
                result.safety.technically_eligible
            ),
            "requires_admin_approval": (
                result.safety.requires_admin_approval
            ),
            "admin_approval_recorded": (
                result.safety.admin_approval_recorded
            ),
            "approval_pending": (
                result.safety.approval_pending
            ),
            "violations": list(result.safety.violations),
            "fallback_action": (
                result.safety.fallback_action.value
            ),
        },
    )
