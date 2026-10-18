
"""
Component 4: Safe HDD Mitigation Decision Intelligence

Module: contracts.py
Purpose: Define validated decision inputs and mitigation actions.

Status: Local research prototype.
No physical disk operations are performed.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from math import isfinite
from typing import Optional


class MitigationAction(str, Enum):
    """Proposed discrete mitigation action space."""

    NO_OP = "NO_OP"
    RECHECK_SOON = "RECHECK_SOON"
    THROTTLE_BACKGROUND_IO = "THROTTLE_BACKGROUND_IO"
    MIGRATE_DATA = "MIGRATE_DATA"
    REBALANCE_WORKLOAD = "REBALANCE_WORKLOAD"
    ISOLATE_DRIVE = "ISOLATE_DRIVE"
    RECOMMEND_REPLACEMENT = "RECOMMEND_REPLACEMENT"


def validate_fraction(name: str, value: float) -> None:
    """Require a finite numeric value between 0 and 1."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(f"{name} must be between 0 and 1")


@dataclass(frozen=True)
class StorageContext:
    """Operational storage conditions for mitigation decisions."""

    free_capacity_gb: float
    data_to_migrate_gb: float
    healthy_redundant_copies: int
    target_healthy: bool
    migration_progress_fraction: float
    io_utilization: float
    admin_approved: bool = False

    def __post_init__(self):

        for name in ("free_capacity_gb", "data_to_migrate_gb"):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative")

        if (
            type(self.healthy_redundant_copies) is not int
            or self.healthy_redundant_copies < 0
        ):
            raise ValueError("Invalid redundant copy count")

        for name in ("target_healthy", "admin_approved"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")

        validate_fraction(
            "migration_progress_fraction",
            self.migration_progress_fraction
        )

        validate_fraction("io_utilization", self.io_utilization)


@dataclass(frozen=True)
class DriveState:
    """HDD risk assessment and associated operational context."""

    disk_id: str
    observed_at: str
    failure_probability: float
    prediction_horizon_days: int
    rul_days: Optional[float]
    storage: StorageContext

    def __post_init__(self):

        if not isinstance(self.disk_id, str) or not self.disk_id.strip():
            raise ValueError("disk_id cannot be empty")

        validate_fraction(
            "failure_probability",
            self.failure_probability
        )

        if (
            type(self.prediction_horizon_days) is not int
            or self.prediction_horizon_days <= 0
        ):
            raise ValueError("Prediction horizon must be positive")

        if self.rul_days is not None:
            if (
                isinstance(self.rul_days, bool)
                or not isinstance(self.rul_days, (int, float))
                or not isfinite(self.rul_days)
                or self.rul_days < 0
            ):
                raise ValueError("Invalid remaining useful life")

        if not isinstance(self.storage, StorageContext):
            raise TypeError("storage must be a StorageContext")

        try:
            timestamp = datetime.fromisoformat(
                self.observed_at.replace("Z", "+00:00")
            )
        except (ValueError, AttributeError) as exc:
            raise ValueError("Invalid observation timestamp") from exc

        if (
            timestamp.tzinfo is None
            or timestamp.utcoffset() is None
        ):
            raise ValueError("Observation timestamp needs a timezone")
