
"""
Component 4 - Safe HDD Mitigation API

Provides a REST endpoint for safety-checked,
risk-aware HDD mitigation recommendations.

Execution mode: DRY_RUN_ONLY
"""

from fastapi import APIRouter

from app.schemas.component4_decision import (
    HDDDecisionRequest,
    HDDDecisionResponse,
)

from app.services.component4_decision_service import (
    generate_safe_recommendation,
)

router = APIRouter()


@router.post(
    "/decide",
    response_model=HDDDecisionResponse,
    summary="Generate a safe HDD mitigation recommendation",
    description=(
        "Evaluates HDD failure risk, remaining useful life, "
        "and storage safety conditions to produce an "
        "explainable dry-run mitigation recommendation."
    ),
)
def decide_hdd_mitigation(
    request: HDDDecisionRequest,
) -> HDDDecisionResponse:
    """Return an independently safety-checked decision."""

    return generate_safe_recommendation(request)
