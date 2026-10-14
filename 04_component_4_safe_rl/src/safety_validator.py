
"""
Component 4: Independent HDD Mitigation Safety Validator

Purpose:
Validate mitigation proposals independently of the decision policy.

Research stage:
Local dry-run prototype only. No physical HDD operations.

Important:
Safety thresholds are configurable simulation assumptions,
not validated guarantees for production storage systems.
"""

from dataclasses import dataclass

from contracts import DriveState, MitigationAction


@dataclass(frozen=True)
class SafetyConfig:
    """Initial research safety assumptions."""

    minimum_migration_copies: int = 1
    minimum_isolation_copies: int = 2
    maximum_migration_io: float = 0.80
    migration_capacity_reserve: float = 0.10


@dataclass(frozen=True)
class SafetyAssessment:
    """Result of an independent mitigation safety check."""

    proposed_action: MitigationAction
    technically_eligible: bool
    requires_admin_approval: bool
    admin_approval_recorded: bool
    violations: tuple[str, ...]
    fallback_action: MitigationAction

    @property
    def approval_pending(self) -> bool:
        return (
            self.technically_eligible
            and self.requires_admin_approval
            and not self.admin_approval_recorded
        )


class SafetyValidator:
    """
    Preliminary safety shield.

    Separates technical eligibility from administrator approval.

    This class does NOT execute mitigation actions.
    """

    APPROVAL_REQUIRED = {
        MitigationAction.THROTTLE_BACKGROUND_IO,
        MitigationAction.MIGRATE_DATA,
        MitigationAction.REBALANCE_WORKLOAD,
        MitigationAction.ISOLATE_DRIVE,
    }

    def __init__(self, config: SafetyConfig | None = None):
        self.config = config or SafetyConfig()

    def assess(
        self,
        state: DriveState,
        action: MitigationAction
    ) -> SafetyAssessment:

        if not isinstance(state, DriveState):
            raise TypeError("state must be DriveState")

        if not isinstance(action, MitigationAction):
            raise TypeError("action must be MitigationAction")

        storage = state.storage
        violations = []

        if action == MitigationAction.MIGRATE_DATA:

            remaining_gb = (
                storage.data_to_migrate_gb
                * (1.0 - storage.migration_progress_fraction)
            )

            required_capacity = remaining_gb * (
                1.0 + self.config.migration_capacity_reserve
            )

            if remaining_gb <= 0:
                violations.append(
                    "No remaining data requires migration"
                )

            if not storage.target_healthy:
                violations.append(
                    "Migration destination is unhealthy"
                )

            if storage.free_capacity_gb < required_capacity:
                violations.append(
                    "Insufficient destination capacity"
                )

            if (
                storage.healthy_redundant_copies
                < self.config.minimum_migration_copies
            ):
                violations.append(
                    "Insufficient protected copies for migration"
                )

            if (
                storage.io_utilization
                > self.config.maximum_migration_io
            ):
                violations.append(
                    "I/O utilization exceeds migration budget"
                )

        elif action == MitigationAction.ISOLATE_DRIVE:

            if (
                storage.healthy_redundant_copies
                < self.config.minimum_isolation_copies
            ):
                violations.append(
                    "Insufficient protected copies for isolation"
                )

            if storage.migration_progress_fraction < 1.0:
                violations.append(
                    "Migration is not fully completed"
                )

        elif action == MitigationAction.REBALANCE_WORKLOAD:

            # Workload destination/placement information is not
            # yet present in the input contract.
            violations.append(
                "Workload placement safety cannot yet be verified"
            )

        elif action == MitigationAction.THROTTLE_BACKGROUND_IO:

            # We cannot yet distinguish critical from
            # noncritical I/O in the input contract.
            violations.append(
                "Noncritical workload status is not verified"
            )

        requires_approval = action in self.APPROVAL_REQUIRED

        return SafetyAssessment(
            proposed_action=action,
            technically_eligible=len(violations) == 0,
            requires_admin_approval=requires_approval,
            admin_approval_recorded=storage.admin_approved,
            violations=tuple(violations),
            fallback_action=MitigationAction.RECHECK_SOON
        )
