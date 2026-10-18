
"""
Component 4: HDD Mitigation Decision Engine

Purpose:
Generate explainable, risk-aware mitigation recommendations
using a deterministic baseline and independent safety shield.

Research status:
Prototype baseline for future RL policy comparisons.

IMPORTANT:
- Dry-run only.
- No real storage operations.
- Thresholds are provisional research assumptions.
- Only seven-day risk probabilities are accepted at this stage.
"""

from dataclasses import dataclass
from math import isfinite

from contracts import DriveState, MitigationAction
from safety_validator import SafetyValidator, SafetyAssessment


@dataclass(frozen=True)
class DecisionConfig:
    """Provisional policy thresholds."""

    elevated_risk: float = 0.15
    high_risk: float = 0.50
    critical_risk: float = 0.85

    elevated_rul_days: float = 45.0
    high_rul_days: float = 14.0
    critical_rul_days: float = 3.0

    required_horizon_days: int = 7

    def __post_init__(self):

        risks = [
            self.elevated_risk,
            self.high_risk,
            self.critical_risk
        ]

        if not all(
            isinstance(x, (int, float))
            and not isinstance(x, bool)
            and isfinite(x)
            and 0 <= x <= 1
            for x in risks
        ):
            raise ValueError("Invalid risk thresholds")

        if risks != sorted(risks) or len(set(risks)) != 3:
            raise ValueError(
                "Risk thresholds must be strictly increasing"
            )

        ruls = [
            self.critical_rul_days,
            self.high_rul_days,
            self.elevated_rul_days
        ]

        if not all(
            isinstance(x, (int, float))
            and not isinstance(x, bool)
            and isfinite(x)
            and x >= 0
            for x in ruls
        ):
            raise ValueError("Invalid RUL thresholds")

        if ruls != sorted(ruls):
            raise ValueError("RUL thresholds must be ordered")

        if (
            type(self.required_horizon_days) is not int
            or self.required_horizon_days <= 0
        ):
            raise ValueError("Invalid prediction horizon")


@dataclass(frozen=True)
class DecisionResult:
    """Traceable result of a mitigation decision."""

    disk_id: str
    risk_level: str
    proposed_action: MitigationAction
    recommended_action: MitigationAction
    decision_status: str
    explanation: str
    safety: SafetyAssessment
    policy_name: str = "risk_aware_baseline_v1"
    execution_mode: str = "DRY_RUN_ONLY"


class HDDDecisionEngine:
    """
    Risk-aware baseline mitigation policy.

    The policy proposes an action.

    The safety validator independently determines whether
    the action is technically eligible and approved.

    This class never executes real mitigation actions.
    """

    def __init__(
        self,
        config: DecisionConfig | None = None,
        validator: SafetyValidator | None = None
    ):
        self.config = config or DecisionConfig()
        self.validator = validator or SafetyValidator()

    def decide(self, state: DriveState) -> DecisionResult:

        if not isinstance(state, DriveState):
            raise TypeError("Expected a validated DriveState")

        if (
            state.prediction_horizon_days
            != self.config.required_horizon_days
        ):
            raise ValueError(
                "Prediction horizon mismatch. "
                "This baseline requires a seven-day risk estimate."
            )

        risk = state.failure_probability
        rul = state.rul_days
        config = self.config

        critical = (
            risk >= config.critical_risk
            or (
                rul is not None
                and rul <= config.critical_rul_days
            )
        )

        high = (
            risk >= config.high_risk
            or (
                rul is not None
                and rul <= config.high_rul_days
            )
        )

        elevated = (
            risk >= config.elevated_risk
            or (
                rul is not None
                and rul <= config.elevated_rul_days
            )
        )

        migration_complete = (
            state.storage.migration_progress_fraction >= 1.0
        )

        # ----------------------------------------------------
        # 1. Baseline policy proposes an action
        # ----------------------------------------------------

        if critical:
            risk_level = "CRITICAL"

            if migration_complete:
                proposed = MitigationAction.ISOLATE_DRIVE
            else:
                proposed = MitigationAction.MIGRATE_DATA

        elif high:
            risk_level = "HIGH"

            if migration_complete:
                proposed = MitigationAction.RECHECK_SOON
            else:
                proposed = MitigationAction.MIGRATE_DATA

        elif elevated:
            risk_level = "ELEVATED"
            proposed = MitigationAction.RECHECK_SOON

        else:
            risk_level = "LOW"
            proposed = MitigationAction.NO_OP

        # ----------------------------------------------------
        # 2. Independent safety assessment
        # ----------------------------------------------------

        assessment = self.validator.assess(
            state,
            proposed
        )

        # ----------------------------------------------------
        # 3. Resolve recommendation safely
        # ----------------------------------------------------

        if not assessment.technically_eligible:

            recommended = assessment.fallback_action
            status = "BLOCKED_BY_SAFETY"

            reason = (
                "Proposed action blocked: "
                + "; ".join(assessment.violations)
            )

        elif assessment.approval_pending:

            recommended = assessment.fallback_action
            status = "PENDING_ADMIN_APPROVAL"

            reason = (
                "The proposed action passes current technical "
                "checks but needs administrator approval."
            )

        else:

            recommended = proposed
            status = "ELIGIBLE_DRY_RUN"

            reason = (
                "The proposed action passed the current "
                "prototype safety checks."
            )

        # ----------------------------------------------------
        # 4. Produce a traceable explanation
        # ----------------------------------------------------

        rul_text = (
            "unavailable"
            if rul is None
            else f"{rul:.2f} days"
        )

        explanation = (
            f"Risk level: {risk_level}. "
            f"Seven-day failure probability: {risk:.2%}. "
            f"Estimated RUL: {rul_text}. "
            f"Proposed action: {proposed.value}. "
            f"{reason} "
            "This is a dry-run recommendation only."
        )

        return DecisionResult(
            disk_id=state.disk_id,
            risk_level=risk_level,
            proposed_action=proposed,
            recommended_action=recommended,
            decision_status=status,
            explanation=explanation,
            safety=assessment
        )
