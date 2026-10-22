
"""
Component 4 - Safe HDD Mitigation API Contracts

Defines validated request and response schemas for
the risk-aware HDD mitigation decision endpoint.

Research mode: Simulation / Dry Run Only
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class StorageInput(BaseModel):
    """Operational conditions supplied to the decision engine."""

    free_capacity_gb: float = Field(ge=0, allow_inf_nan=False)
    data_to_migrate_gb: float = Field(ge=0, allow_inf_nan=False)
    healthy_redundant_copies: int = Field(ge=0)
    target_healthy: bool
    migration_progress_fraction: float = Field(ge=0, le=1)
    io_utilization: float = Field(ge=0, le=1)
    admin_approved: bool = False


class HDDDecisionRequest(BaseModel):
    """Validated input for Component 4 mitigation decisions."""

    disk_id: str = Field(min_length=1)
    observed_at: datetime
    failure_probability: float = Field(
        ge=0, le=1, allow_inf_nan=False
    )
    prediction_horizon_days: Literal[7] = 7
    rul_days: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False
    )
    storage: StorageInput

    @field_validator("observed_at")
    @classmethod
    def validate_timezone(cls, value: datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(
                "Observation timestamp must include timezone"
            )
        return value


class SafetyResult(BaseModel):
    """Independent safety-validator response."""

    proposed_action: str
    technically_eligible: bool
    requires_admin_approval: bool
    admin_approval_recorded: bool
    approval_pending: bool
    violations: list[str]
    fallback_action: str


class HDDDecisionResponse(BaseModel):
    """Explainable HDD mitigation recommendation."""

    disk_id: str
    risk_level: str
    proposed_action: str
    recommended_action: str
    decision_status: str
    explanation: str
    policy_name: str
    execution_mode: Literal["DRY_RUN_ONLY"]
    safety: SafetyResult
