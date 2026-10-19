"""Per-drive adaptive degradation thresholds.

The production detector in Component 2 uses *global* absolute thresholds
(``SMART1_THRESHOLDS`` / ``SMART187_THRESHOLDS``) tuned once offline. Every
drive is judged against the same fixed bounds, so a naturally noisy drive trips
false alarms while a quiet one is under-detected.

This module adds a **personalized, self-calibrating** layer on top. The design
keeps the published global detector as the **floor** so real degradation is
never missed, and only *tightens* it for drives that are demonstrably noisier
than typical:

* Each drive learns its own noise level for the two change signals
  (``smart_1_normalized_change_7d`` / ``smart_187_normalized_change_7d``) as a
  robust scale (MAD) over a rolling baseline window.
* That noise level sets a **personalized strictness factor** (>1 for noisy
  drives). The global bounds are scaled by it, so a chatty drive must show a
  *bigger-than-its-own-noise* drop to count as degradation -> fewer false
  alarms, without ever lowering the bar below the published detector.
* The baseline **evolves** (sliding window), so the thresholds genuinely adapt
  online as new observations stream in.
* Detector operating points (``min_severity`` / ``persistence``) are also
  nudged from the drive's observed qualifying-observation rate.

A separate **acceleration z-score** (deviation from the drive's own recent
trend) is reported as a diagnostic: it flags a drive degrading *faster than it
normally does*, which is an early-warning signal distinct from absolute
severity.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from app.ml._component2_paths import get_detector

# Change-signal column names (same semantics as Component 2: observation diffs).
SIGNAL_CHANGE_7D = (
    "smart_1_normalized_change_7d",
    "smart_187_normalized_change_7d",
)

# A "typical" magnitude of day-to-day change-noise, used as the reference to
# judge whether a specific drive is noisier than average.
REFERENCE_SCALE = 1.0

# Strictness is capped so an extremely noisy drive cannot become undetectable.
MAX_STRICTNESS = 3.0

# Robust z bands for the *acceleration* diagnostic (deviation from own trend).
Z_SEVERITY_BANDS: List[Tuple[float, int]] = [
    (-4.0, 4),
    (-3.0, 3),
    (-2.0, 2),
    (-1.0, 1),
]

_SCALE_FLOOR = 1e-6

# Operating-point adaptation: how strongly the observed qualifying rate pushes
# the persistence away from the global default.
PERSISTENCE_SWING = 4


def _median(values: List[float]) -> float:
    return float(np.median(values)) if values else 0.0


def _mad(values: List[float]) -> float:
    """Median absolute deviation, scaled to be a consistent sigma estimator."""
    if not values:
        return 0.0
    med = float(np.median(values))
    deviations = [abs(v - med) for v in values]
    return float(np.median(deviations)) * 1.4826


def _severity_from_z(z: float) -> int:
    for bound, level in Z_SEVERITY_BANDS:
        if z <= bound:
            return level
    return 0


@dataclass
class AdaptiveThresholds:
    """Adaptive, per-drive degradation threshold state."""

    serial_number: str
    window: int = 60
    baseline_warmup: int = 7
    _changes_1: Deque[float] = field(default_factory=lambda: deque(maxlen=60))
    _changes_187: Deque[float] = field(default_factory=lambda: deque(maxlen=60))
    _qualifying_history: Deque[int] = field(default_factory=lambda: deque(maxlen=60))

    def __post_init__(self) -> None:
        self._changes_1 = deque(self._changes_1, maxlen=self.window)
        self._changes_187 = deque(self._changes_187, maxlen=self.window)
        self._qualifying_history = deque(
            self._qualifying_history, maxlen=self.window
        )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def observe(
        self,
        change_1: Optional[float],
        change_187: Optional[float],
        globally_qualifying: bool,
    ) -> None:
        """Feed one observation's change values into the baseline windows."""
        if change_1 is not None and not math.isnan(change_1):
            self._changes_1.append(float(change_1))
        if change_187 is not None and not math.isnan(change_187):
            self._changes_187.append(float(change_187))
        self._qualifying_history.append(1 if globally_qualifying else 0)

    # ------------------------------------------------------------------
    # Baselines / noise
    # ------------------------------------------------------------------
    def _scale(self, values: Deque[float]) -> float:
        data = list(values)
        if len(data) < self.baseline_warmup:
            return 0.0
        return _mad(data)

    def _center(self, values: Deque[float]) -> float:
        data = list(values)
        if len(data) < self.baseline_warmup:
            return 0.0
        return _median(data)

    def strictness(self, values: Deque[float]) -> float:
        """
        How much stricter than the global bounds this drive needs.

        1.0 = exactly the published detector (quiet / typical drive). Grows
        toward MAX_STRICTNESS for drives whose change signal is noisier than
        the reference, so they must show a larger drop to count.
        """
        scale = self._scale(values)
        if scale <= _SCALE_FLOOR:
            return 1.0
        return float(min(max(scale / REFERENCE_SCALE, 1.0), MAX_STRICTNESS))

    # ------------------------------------------------------------------
    # Severity (global detector as the floor, tightened per-drive)
    # ------------------------------------------------------------------
    @staticmethod
    def _global_bounds(which: str) -> List[Tuple[float, int]]:
        detector = get_detector()
        if which == "smart_1":
            return list(detector.SMART1_THRESHOLDS)
        return list(detector.SMART187_THRESHOLDS)

    @staticmethod
    def _severity_from_bounds(
        value: Optional[float],
        bounds: List[Tuple[float, int]],
        strictness: float,
    ) -> int:
        if value is None or math.isnan(value):
            return 0
        for bound, level in bounds:
            # Bounds are negative (a drop). Scaling by strictness > 1 moves
            # them further from zero -> harder to trip -> fewer false alarms.
            if value <= bound * strictness:
                return level
        return 0

    def severity(
        self,
        change_1: Optional[float],
        change_187: Optional[float],
    ) -> int:
        """Personalized severity (0-4), never looser than the global detector."""
        sev_1 = self._severity_from_bounds(
            change_1, self._global_bounds("smart_1"), self.strictness(self._changes_1)
        )
        sev_187 = self._severity_from_bounds(
            change_187,
            self._global_bounds("smart_187"),
            self.strictness(self._changes_187),
        )
        return max(sev_1, sev_187)

    def acceleration_z(
        self,
        change_1: Optional[float],
        change_187: Optional[float],
    ) -> Optional[float]:
        """
        Diagnostic: how far the current change sits below this drive's own
        recent trend, in robust sigma units. Negative = degrading faster than
        this drive normally does (early-warning signal).
        """
        def z(value, values):
            center = self._center(values)
            scale = self._scale(values)
            if value is None or math.isnan(value):
                return None
            if scale <= _SCALE_FLOOR:
                return None
            return round(float((value - center) / scale), 3)

        zs = [x for x in (z(change_1, self._changes_1), z(change_187, self._changes_187)) if x is not None]
        return min(zs) if zs else None

    # ------------------------------------------------------------------
    # Adaptive operating point
    # ------------------------------------------------------------------
    def observed_qualifying_rate(self) -> Optional[float]:
        if not self._qualifying_history:
            return None
        return float(np.mean(self._qualifying_history))

    def suggested_persistence(self, base_persistence: int = 7) -> int:
        """
        Raise persistence for a chatty drive (high qualifying rate) and lower
        it for a quiet one, so the streak requirement adapts per device.
        """
        rate = self.observed_qualifying_rate()
        if rate is None:
            return base_persistence
        offset = int(round((rate - 0.5) * PERSISTENCE_SWING))
        return max(3, base_persistence + offset)

    def suggested_min_severity(self, base_min_severity: int = 1) -> int:
        """Keep the global floor unless the drive is chronically noisy."""
        rate = self.observed_qualifying_rate()
        if rate is not None and rate > 0.8:
            return base_min_severity + 1
        return base_min_severity

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, object]:
        return {
            "serial_number": self.serial_number,
            "baseline_samples": len(self._changes_1) + len(self._changes_187),
            "smart_1_change_center": round(self._center(self._changes_1), 4),
            "smart_1_change_noise": round(self._scale(self._changes_1), 4),
            "smart_1_strictness": round(self.strictness(self._changes_1), 3),
            "smart_187_change_center": round(self._center(self._changes_187), 4),
            "smart_187_change_noise": round(self._scale(self._changes_187), 4),
            "smart_187_strictness": round(self.strictness(self._changes_187), 3),
            "observed_qualifying_rate": (
                round(self.observed_qualifying_rate(), 4)
                if self.observed_qualifying_rate() is not None
                else None
            ),
            "suggested_persistence": self.suggested_persistence(),
            "suggested_min_severity": self.suggested_min_severity(),
            "is_calibrated": len(self._changes_1) >= self.baseline_warmup,
        }
