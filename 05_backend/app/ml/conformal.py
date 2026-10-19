"""Conformal RUL confidence intervals + data-quality gating.

Component 2's BiLSTM emits a *single* RUL point estimate. Operations teams
cannot act on "14 days" without knowing how trustworthy that number is. This
module turns the point estimate into a **calibrated prediction interval** via
split-conformal prediction, then **gates its width by data quality** so the
twin explicitly reports how much to trust it.

Two ideas combined:

1. **Conformal interval** — built from a calibration set of absolute residuals
   (|actual - predicted|). Distribution-free and gives a target marginal
   coverage guarantee when calibration data is representative.
2. **Quality gate** — the interval is *widened* when the input data is poor
   (few SMART signals present, short history, missing temporal features), and
   *narrowed* when data is rich. Confidence is derived from both the interval
   width and the quality score.

Calibration can be **seeded** from Component 2's held-out residual artifacts
and then **updated online** as real outcomes arrive (feeds Feature 1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Fraction of the interval widened per unit of *lack* of quality.
QUALITY_WIDENING = 0.6

# A completely quality-less input at most doubles the base interval.
QUALITY_MIN = 0.0


def data_quality_score(
    health_features_available: Optional[float] = None,
    history_length: int = 0,
    history_target: int = 30,
    temporal_valid_fraction: Optional[float] = None,
    max_features: int = 5,
) -> float:
    """
    Combine telemetry richness into a single 0..1 data-quality score.

    * ``health_features_available`` — how many of the 5 core SMART signals
      were present on the latest observation.
    * ``history_length`` vs ``history_target`` — how much temporal context the
      twin has (the RUL model wants 30 observations).
    * ``temporal_valid_fraction`` — fraction of the 6 derived temporal features
      that had real (non-missing) values.
    """
    parts: List[float] = []

    if health_features_available is not None:
        parts.append(
            max(0.0, min(float(health_features_available), max_features))
            / max_features
        )

    if history_target:
        parts.append(max(0.0, min(float(history_length) / history_target, 1.0)))

    if temporal_valid_fraction is not None:
        parts.append(max(0.0, min(float(temporal_valid_fraction), 1.0)))

    if not parts:
        return 0.0

    return float(np.mean(parts))


@dataclass
class ConformalCalibrator:
    """Split-conformal calibrator over absolute RUL residuals."""

    alpha: float = 0.1  # target miscoverage -> 90% interval
    max_calibration: int = 500
    _residuals: List[float] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Calibration data
    # ------------------------------------------------------------------
    def add_residual(self, predicted: float, actual: float) -> None:
        if predicted is None or actual is None:
            return
        if math.isnan(predicted) or math.isnan(actual):
            return
        self._residuals.append(abs(float(actual) - float(predicted)))
        if len(self._residuals) > self.max_calibration:
            # Keep the most recent window so calibration tracks drift.
            self._residuals = self._residuals[-self.max_calibration :]

    def seed(self, residuals: Sequence[float]) -> None:
        for value in residuals:
            if value is None or (isinstance(value, float) and math.isnan(value)):
                continue
            self._residuals.append(abs(float(value)))
        if len(self._residuals) > self.max_calibration:
            self._residuals = self._residuals[-self.max_calibration :]

    def seed_from_pairs(self, pairs: Sequence[Tuple[float, float]]) -> None:
        for predicted, actual in pairs:
            self.add_residual(predicted, actual)

    @property
    def n_calibration(self) -> int:
        return len(self._residuals)

    # ------------------------------------------------------------------
    # Interval
    # ------------------------------------------------------------------
    def conformity_quantile(self) -> Optional[float]:
        """Split-conformal quantile of absolute residuals (finite-sample)."""
        n = len(self._residuals)
        if n < 10:
            return None
        # ceil((n+1) * (1 - alpha)) / n, clipped to [0,1].
        level = math.ceil((n + 1) * (1.0 - self.alpha)) / n
        level = min(level, 1.0)
        return float(np.quantile(self._residuals, level))

    def interval(
        self,
        predicted: float,
        quality: float,
    ) -> Dict[str, Optional[float]]:
        """
        Quality-gated conformal interval for a predicted RUL.

        Returns dict with low/high/width/base_half_width/coverage_alpha.
        """
        base_q = self.conformity_quantile()

        quality = max(QUALITY_MIN, min(float(quality), 1.0))

        if base_q is None:
            # No reliable calibration yet: use a conservative relative band
            # that shrinks as quality improves.
            base_half = abs(float(predicted)) * 0.5
            if base_half == 0.0:
                base_half = 10.0
        else:
            base_half = base_q

        widen = 1.0 + (1.0 - quality) * QUALITY_WIDENING
        half = base_half * widen

        low = float(predicted) - half
        high = float(predicted) + half
        # RUL cannot be negative.
        low = max(low, 0.0)

        return {
            "low": round(low, 3),
            "high": round(high, 3),
            "width": round(high - low, 3),
            "base_half_width": round(base_half, 3),
            "quality": round(quality, 3),
            "coverage_alpha": self.alpha,
            "calibration_samples": self.n_calibration,
        }

    # ------------------------------------------------------------------
    # Confidence
    # ------------------------------------------------------------------
    def confidence(
        self,
        predicted: float,
        quality: float,
        interval: Optional[Dict[str, Optional[float]]] = None,
    ) -> float:
        """
        A 0..1 trust score combining interval tightness, data quality and
        calibration sufficiency. Higher = more trustworthy.
        """
        if interval is None:
            interval = self.interval(predicted, quality)

        width = interval.get("width") or 0.0
        scale = abs(float(predicted))
        if scale <= 0:
            scale = max(width, 1.0)

        # Tightness: width relative to the prediction magnitude (0..1).
        relative_width = float(width) / scale
        tightness = 1.0 / (1.0 + relative_width)

        # Calibration sufficiency: saturates once we have a decent set.
        sufficiency = min(self.n_calibration / 100.0, 1.0)
        if self.n_calibration < 10:
            sufficiency = 0.3  # early, not fully trustworthy

        quality = max(0.0, min(float(quality), 1.0))

        score = 0.45 * tightness + 0.35 * quality + 0.20 * sufficiency
        return round(max(0.0, min(score, 1.0)), 3)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, Optional[float]]:
        residuals = self._residuals
        if not residuals:
            return {
                "calibration_samples": 0,
                "median_abs_residual": None,
                "mean_abs_residual": None,
                "conformity_quantile": None,
                "target_coverage": 1.0 - self.alpha,
            }
        return {
            "calibration_samples": len(residuals),
            "median_abs_residual": round(float(np.median(residuals)), 3),
            "mean_abs_residual": round(float(np.mean(residuals)), 3),
            "conformity_quantile": (
                round(self.conformity_quantile(), 3)
                if self.conformity_quantile() is not None
                else None
            ),
            "target_coverage": 1.0 - self.alpha,
        }
