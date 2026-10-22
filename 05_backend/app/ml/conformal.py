"""Conformal RUL confidence intervals + data-quality gating.

Component 2's BiLSTM emits a *single* RUL point estimate. Operations teams
cannot act on "14 days" without knowing how trustworthy that number is. This
module turns the point estimate into a **calibrated prediction interval** via
split-conformal prediction, then **gates its width by data quality** so the
twin explicitly reports how much to trust it.

Protocol (split conformal, absolute-residual scores)
----------------------------------------------------
* **Nonconformity score** — ``s_i = |y_i - yhat_i|`` (absolute RUL residual),
  measured on a *calibration* population of drives.
* **Calibration population** — drives that are **disjoint from the drives used
  for final evaluation**. Loading it is the job of
  :mod:`app.ml.calibration_data`; this module only consumes the scores.
* **Target coverage** — ``1 - alpha`` (the project uses ``alpha = 0.1``).
* **Finite-sample quantile** — with ``n`` calibration scores, ``k =
  ceil((n + 1) * (1 - alpha))`` and ``q = s_(k)``, the *k-th smallest
  observed* score. ``[yhat - q, yhat + q]`` then satisfies
  ``P(y in C(x)) >= 1 - alpha`` whenever calibration and evaluation scores
  are exchangeable. (The old implementation interpolated between two order
  statistics; see ``docs`` in the delivery report — the interpolated value is
  not an observed score, and when ``k > n`` the old code silently clamped to
  the maximum residual and still claimed calibration.)
* **Insufficiency** — if ``n < MIN_CALIBRATION_SAMPLES`` or ``k > n`` no finite
  ``q`` can provide the requested coverage. The calibrator then reports
  ``calibrated=False`` and callers fall back to a documented conservative band
  that carries **no** coverage claim.
* **RUL domain** — the lower bound is clipped at ``RUL_DOMAIN_MIN = 0``
  because a drive cannot have negative remaining life. Clipping only ever
  makes the interval *wider* than ``[yhat - q, yhat + q]``, so it cannot
  reduce coverage. The upper bound is deliberately **not** capped (capping it
  would silently invalidate the guarantee).

Quality gate
------------
The calibrated half-width is multiplied by a factor in
``[1.0, 1.0 + QUALITY_WIDENING]``: poor data quality *widens* the interval, a
perfect quality score keeps the calibrated width. The gate **can only widen,
never narrow** — that is what keeps the reported interval a conservative
superset of the calibrated one. (The previous module docstring claimed the
interval is "narrowed when data is rich"; the code never did that, and
allowing it would break the coverage guarantee for rich inputs, so the
documentation was corrected instead of the behaviour.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Fraction of the interval widened per unit of *lack* of quality.
QUALITY_WIDENING = 0.6

# A completely quality-less input at most doubles the base interval.
QUALITY_MIN = 0.0

# Smallest calibration population for which the finite-sample quantile is
# meaningful.  Below this the calibrator refuses to claim coverage.
MIN_CALIBRATION_SAMPLES = 10

# Default target miscoverage (90% interval), kept for compatibility.
DEFAULT_ALPHA = 0.1

# Retained calibration window.  Must be >= the seeded population, otherwise
# seeding silently truncates it (the old default of 500 truncated a
# 2922-residual seed down to 500 residuals = 14 of 89 drives).
DEFAULT_MAX_CALIBRATION = 5000

# Conservative band used while uncalibrated.  No coverage guarantee.
FALLBACK_RELATIVE_HALF_WIDTH = 0.5
FALLBACK_MIN_HALF_WIDTH = 10.0

# A remaining useful life cannot be negative.
RUL_DOMAIN_MIN = 0.0


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


def _finite(value: Any) -> Optional[float]:
    """Return ``float(value)`` when it is a finite number, else ``None``.

    Rejects ``None``, booleans, non-numeric strings/objects, ``NaN`` and
    ``+/-inf``.  This is the single gate used for every residual and every
    prediction that enters the calibrator.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


@dataclass
class ConformalCalibrator:
    """Split-conformal calibrator over absolute RUL residuals."""

    alpha: float = DEFAULT_ALPHA  # target miscoverage -> 90% interval
    max_calibration: int = DEFAULT_MAX_CALIBRATION
    _residuals: List[float] = field(default_factory=list)
    # Residuals rejected for being missing / non-numeric / non-finite.
    n_rejected: int = 0
    # Provenance of the seeded population (checkpoint, split, drive digest...).
    source: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 < float(self.alpha) < 1.0:
            raise ValueError(
                f"alpha must be strictly between 0 and 1, got {self.alpha!r}"
            )
        if int(self.max_calibration) < MIN_CALIBRATION_SAMPLES:
            raise ValueError(
                "max_calibration must be >= "
                f"{MIN_CALIBRATION_SAMPLES}, got {self.max_calibration!r}"
            )

    # ------------------------------------------------------------------
    # Calibration data
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Drop the calibration population and its provenance."""
        self._residuals.clear()
        self.n_rejected = 0
        self.source = {}

    def add_residual(self, predicted: float, actual: float) -> bool:
        """Append ``|actual - predicted|``; reject invalid entries.

        Returns ``True`` when the residual was accepted.  Missing,
        non-numeric, ``NaN`` and infinite inputs are counted in
        ``n_rejected`` instead of silently poisoning the quantile.
        """
        prediction = _finite(predicted)
        observation = _finite(actual)
        if prediction is None or observation is None:
            self.n_rejected += 1
            return False
        residual = abs(observation - prediction)
        if not math.isfinite(residual):
            self.n_rejected += 1
            return False
        self._residuals.append(residual)
        self._trim()
        return True

    def seed(self, residuals: Sequence[float]) -> int:
        """Seed the calibration population from precomputed residuals.

        Invalid entries are rejected (see :meth:`add_residual`); the number of
        accepted residuals is returned so callers can audit the seed.
        """
        accepted = 0
        for value in residuals:
            number = _finite(value)
            if number is None:
                self.n_rejected += 1
                continue
            self._residuals.append(abs(number))
            accepted += 1
            self._trim()
        return accepted

    def seed_from_pairs(
        self, pairs: Sequence[Tuple[float, float]]
    ) -> int:
        """Seed from ``(predicted, actual)`` pairs; returns accepted count."""
        accepted = 0
        for predicted, actual in pairs:
            if self.add_residual(predicted, actual):
                accepted += 1
        return accepted

    def _trim(self) -> None:
        if len(self._residuals) > self.max_calibration:
            # Keep the most recent window so calibration can track drift.
            self._residuals = self._residuals[-self.max_calibration :]

    @property
    def n_calibration(self) -> int:
        return len(self._residuals)

    @property
    def residuals(self) -> List[float]:
        return list(self._residuals)

    # ------------------------------------------------------------------
    # Quantile
    # ------------------------------------------------------------------
    def quantile_index(self) -> Optional[int]:
        """``k = ceil((n + 1) * (1 - alpha))`` when it is attainable."""
        n = len(self._residuals)
        if n < MIN_CALIBRATION_SAMPLES:
            return None
        k = math.ceil((n + 1) * (1.0 - float(self.alpha)))
        return k if 0 < k <= n else None

    def conformity_quantile(self) -> Optional[float]:
        """Finite-sample split-conformal quantile (k-th smallest residual).

        ``None`` means "not calibrated": either the population is too small or
        ``k > n`` makes the requested coverage unreachable with a finite
        quantile.  Callers must not treat ``None`` as "zero width".
        """
        info = self.quantile_info()
        return info["conformity_quantile"]

    def quantile_info(self) -> Dict[str, Any]:
        """Auditable description of the quantile currently in force."""
        n = len(self._residuals)
        target = 1.0 - float(self.alpha)
        info: Dict[str, Any] = {
            "calibration_samples": n,
            "nominal_coverage": target,
            "quantile_index": None,
            "conformity_quantile": None,
            "calibration_sufficient": False,
            "insufficient_reason": None,
        }
        if n < MIN_CALIBRATION_SAMPLES:
            info["insufficient_reason"] = (
                f"need >= {MIN_CALIBRATION_SAMPLES} residuals, have {n}"
            )
            return info
        k = math.ceil((n + 1) * target)
        if k > n:
            info["insufficient_reason"] = (
                f"quantile index k={k} exceeds n={n}: the finite-sample "
                "guarantee cannot be met with a finite half-width"
            )
            return info
        ordered = np.sort(np.asarray(self._residuals, dtype=float))
        info["quantile_index"] = int(k)
        info["conformity_quantile"] = float(ordered[k - 1])
        info["calibration_sufficient"] = True
        return info

    # ------------------------------------------------------------------
    # Interval
    # ------------------------------------------------------------------
    def interval(
        self,
        predicted: float,
        quality: float,
    ) -> Dict[str, Any]:
        """
        Quality-gated conformal interval for a predicted RUL.

        See the module docstring for the protocol.  Keys:

        ``low``/``high``/``width``/``base_half_width`` are the interval and the
        *calibrated* half-width before the quality gate (the forecaster reads
        ``base_half_width``).  ``calibrated`` is ``False`` while the population
        is insufficient; ``conformity_quantile``/``quantile_index`` expose the
        order statistic actually used.
        """
        prediction = _finite(predicted)
        if prediction is None:
            raise ValueError(
                f"predicted RUL must be a finite number, got {predicted!r}"
            )

        quality_value = _finite(quality)
        if quality_value is None:
            # Unknown quality is treated as the worst case (widest interval).
            quality_value = QUALITY_MIN
        quality_value = max(QUALITY_MIN, min(quality_value, 1.0))

        info = self.quantile_info()
        if info["calibration_sufficient"]:
            base_half = float(info["conformity_quantile"])
            calibrated = True
            fallback = False
        else:
            # No reliable calibration: documented conservative band derived
            # from the prediction magnitude.  No coverage claim.
            base_half = max(
                abs(prediction) * FALLBACK_RELATIVE_HALF_WIDTH,
                FALLBACK_MIN_HALF_WIDTH,
            )
            calibrated = False
            fallback = True

        widen = 1.0 + (1.0 - quality_value) * QUALITY_WIDENING
        half = base_half * widen

        low_raw = prediction - half
        low = max(low_raw, RUL_DOMAIN_MIN)
        high = prediction + half

        return {
            "low": round(low, 3),
            "high": round(high, 3),
            "width": round(high - low, 3),
            "base_half_width": round(base_half, 3),
            "half_width": round(half, 3),
            "quality": round(quality_value, 3),
            "quality_widening_factor": round(widen, 3),
            "coverage_alpha": self.alpha,
            "nominal_coverage": round(1.0 - float(self.alpha), 3),
            "calibration_samples": self.n_calibration,
            "calibrated": calibrated,
            "uncalibrated_fallback": fallback,
            "conformity_quantile": (
                round(base_half, 3) if calibrated else None
            ),
            "quantile_index": info["quantile_index"],
            "lower_bound_clipped": bool(low_raw < RUL_DOMAIN_MIN),
        }

    # ------------------------------------------------------------------
    # Confidence
    # ------------------------------------------------------------------
    def confidence(
        self,
        predicted: float,
        quality: float,
        interval: Optional[Dict[str, Any]] = None,
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
            scale = max(float(width), 1.0)

        # Tightness: width relative to the prediction magnitude (0..1).
        relative_width = float(width) / scale
        tightness = 1.0 / (1.0 + relative_width)

        # Calibration sufficiency: saturates once we have a decent set.
        if not self.quantile_info()["calibration_sufficient"]:
            sufficiency = 0.3  # early / uncalibrated, not fully trustworthy
        else:
            sufficiency = min(self.n_calibration / 100.0, 1.0)

        quality = max(0.0, min(float(quality), 1.0))

        score = 0.45 * tightness + 0.35 * quality + 0.20 * sufficiency
        return round(max(0.0, min(score, 1.0)), 3)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        """Auditable snapshot of the calibration population and quantile."""
        info = self.quantile_info()
        base: Dict[str, Any] = {
            "alpha": self.alpha,
            "target_coverage": 1.0 - float(self.alpha),
            "nominal_coverage": info["nominal_coverage"],
            "calibration_samples": self.n_calibration,
            "max_calibration_window": self.max_calibration,
            "calibration_sufficient": info["calibration_sufficient"],
            "insufficient_reason": info["insufficient_reason"],
            "quantile_index": info["quantile_index"],
            "conformity_quantile": info["conformity_quantile"],
            "rejected_residuals": self.n_rejected,
            "median_abs_residual": None,
            "mean_abs_residual": None,
            "residual_min": None,
            "residual_p25": None,
            "residual_p75": None,
            "residual_p90": None,
            "residual_max": None,
            "interval_width_at_full_quality": None,
            "interval_width_at_zero_quality": None,
            "source": dict(self.source),
        }
        residuals = self._residuals
        if residuals:
            values = np.asarray(residuals, dtype=float)
            base.update(
                {
                    "median_abs_residual": round(float(np.median(values)), 3),
                    "mean_abs_residual": round(float(values.mean()), 3),
                    "residual_min": round(float(values.min()), 3),
                    "residual_p25": round(
                        float(np.percentile(values, 25)), 3
                    ),
                    "residual_p75": round(
                        float(np.percentile(values, 75)), 3
                    ),
                    "residual_p90": round(
                        float(np.percentile(values, 90)), 3
                    ),
                    "residual_max": round(float(values.max()), 3),
                }
            )
        if info["calibration_sufficient"]:
            quantile = float(info["conformity_quantile"])
            base["interval_width_at_full_quality"] = round(2.0 * quantile, 3)
            base["interval_width_at_zero_quality"] = round(
                2.0 * quantile * (1.0 + QUALITY_WIDENING), 3
            )
        return base
