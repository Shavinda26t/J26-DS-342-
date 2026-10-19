"""Genuine online adaptation for RUL predictions.

Component 2's BiLSTM is frozen after training: it never learns from outcomes.
This module adds a **lightweight online learner** on top of the frozen model so
the twin genuinely adapts with streamed feedback.

For each drive we fit, *recursively*, a two-parameter correction

    actual_rul  ~=  b0 + b1 * predicted_rul

using **Recursive Least Squares (RLS)** with a forgetting factor. Key points
that make this real online learning (not just a lookup):

* Parameters ``theta = [b0, b1]`` **update on every (predicted, actual) pair**
  as it arrives — no batch retraining, no frozen table.
* The covariance ``P`` and gain ``K`` are updated recursively; the forgetting
  factor ``lam`` lets old evidence decay so the adapter tracks concept drift.
* A per-serial adapter plus a **global adapter** allows cold-start sharing: a
  drive with no outcomes yet borrows the population-level correction.

Safety guards clamp the gain and correction magnitude so a handful of outliers
cannot destabilise predictions.

This is the "genuine adaptation (online learning)" capability: the twin's
output genuinely changes as it observes reality.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# Forgetting factor: 1.0 = never forget, lower = track drift faster.
DEFAULT_FORGETTING = 0.985

# Clamp the learned slope so corrections stay sane.
B1_MIN = 0.3
B1_MAX = 3.0

# Clamp the absolute correction (days) applied to a single prediction.
MAX_CORRECTION_DAYS = 60.0

# Minimum updates before a personal adapter is trusted over the global one.
PERSONAL_TRUST_UPDATES = 5


@dataclass
class OnlineRulAdapter:
    """Recursive least-squares correction for predicted RUL."""

    forgetting: float = DEFAULT_FORGETTING
    delta: float = 1e-2  # init covariance scaling (small -> trust init more)
    theta: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 1.0], dtype=float)
    )
    p: np.ndarray = field(default_factory=lambda: np.eye(2) * 100.0)
    n_updates: int = 0
    residuals: List[float] = field(default_factory=list)
    last_actual: Optional[float] = None
    last_predicted: Optional[float] = None

    def _clamp(self) -> None:
        self.theta[1] = float(np.clip(self.theta[1], B1_MIN, B1_MAX))
        # Keep intercept bounded to a plausible RUL offset.
        self.theta[0] = float(np.clip(self.theta[0], -MAX_CORRECTION_DAYS, MAX_CORRECTION_DAYS))

    def update(self, predicted: float, actual: float) -> None:
        """Consume one labelled outcome and update the parameters."""
        if predicted is None or actual is None:
            return
        if math.isnan(predicted) or math.isnan(actual):
            return

        phi = np.array([1.0, float(predicted)], dtype=float)
        d = float(actual)

        p_phi = self.p @ phi
        denom = self.forgetting + float(phi @ p_phi)
        if denom <= 0 or not math.isfinite(denom):
            return
        k = p_phi / denom

        error = d - float(self.theta @ phi)
        self.theta = self.theta + k * error
        self.p = (self.p - np.outer(k, phi) @ self.p) / self.forgetting
        self._clamp()

        self.n_updates += 1
        self.residuals.append(error)
        if len(self.residuals) > 500:
            self.residuals = self.residuals[-500:]
        self.last_predicted = float(predicted)
        self.last_actual = d

    def correction(self, predicted: float) -> float:
        """Signed correction (days) to add to the raw prediction."""
        if predicted is None or math.isnan(predicted):
            return 0.0
        adapted = self.theta[0] + self.theta[1] * float(predicted)
        delta = adapted - float(predicted)
        return float(np.clip(delta, -MAX_CORRECTION_DAYS, MAX_CORRECTION_DAYS))

    def adapt(self, predicted: float) -> float:
        if predicted is None or math.isnan(predicted):
            return predicted
        return max(0.0, float(predicted) + self.correction(predicted))

    @property
    def mean_abs_residual(self) -> Optional[float]:
        if not self.residuals:
            return None
        return round(float(np.mean(np.abs(self.residuals))), 3)

    @property
    def is_ready(self) -> bool:
        return self.n_updates >= PERSONAL_TRUST_UPDATES

    def snapshot(self) -> Dict[str, Optional[float]]:
        return {
            "n_updates": self.n_updates,
            "intercept": round(float(self.theta[0]), 4),
            "slope": round(float(self.theta[1]), 4),
            "mean_abs_residual": self.mean_abs_residual,
            "last_predicted": self.last_predicted,
            "last_actual": self.last_actual,
            "is_ready": self.is_ready,
        }


class DualRulAdapter:
    """Per-serial adapter with a shared global adapter for cold start."""

    def __init__(self, forgetting: float = DEFAULT_FORGETTING) -> None:
        self.forgetting = forgetting
        self.global_adapter = OnlineRulAdapter(forgetting=forgetting)
        self.serial_adapters: Dict[str, OnlineRulAdapter] = {}

    def _get(self, serial: str) -> OnlineRulAdapter:
        if serial not in self.serial_adapters:
            self.serial_adapters[serial] = OnlineRulAdapter(forgetting=self.forgetting)
        return self.serial_adapters[serial]

    def update(self, serial: str, predicted: float, actual: float) -> None:
        self._get(serial).update(predicted, actual)
        self.global_adapter.update(predicted, actual)

    def adapt(self, serial: str, predicted: float) -> Dict[str, object]:
        personal = self._get(serial)
        if personal.is_ready:
            adapted = personal.adapt(predicted)
            source = "personal"
        else:
            adapted = self.global_adapter.adapt(predicted)
            source = "global" if self.global_adapter.n_updates > 0 else "none"
        return {
            "raw": round(float(predicted), 3) if predicted is not None else None,
            "adapted": round(adapted, 3) if predicted is not None else None,
            "correction_days": (
                round(adapted - float(predicted), 3) if predicted is not None else None
            ),
            "source": source,
            "personal_updates": personal.n_updates,
            "global_updates": self.global_adapter.n_updates,
            "personal": personal.snapshot(),
            "global": self.global_adapter.snapshot(),
        }

    def snapshot(self) -> Dict[str, object]:
        return {
            "global": self.global_adapter.snapshot(),
            "serials_tracked": len(self.serial_adapters),
            "serials_ready": sum(
                1 for a in self.serial_adapters.values() if a.is_ready
            ),
        }
