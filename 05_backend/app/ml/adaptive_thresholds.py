"""Per-drive adaptive degradation thresholds.

The production detector in Component 2 uses *global* absolute thresholds
(``SMART1_THRESHOLDS`` / ``SMART187_THRESHOLDS``) tuned once offline. Every
drive is judged against the same fixed bounds, so a naturally noisy drive trips
false alarms while a quiet one is under-detected.

This module adds a **personalized, self-calibrating** layer on top. Design:

* Each drive learns its own change-signal noise level as a robust scale (MAD)
  over a rolling window of **baseline** observations - observations that were
  *not* already qualifying under the global detector. Degradation therefore
  never enters its own noise estimate, so the personalization cannot tighten a
  drive's thresholds precisely while it is failing.
* The drive noise is compared with a **population reference scale** fitted on
  training/validation drives (``REFERENCE_SCALES`` below). A drive whose
  baseline noise is at or below the population median keeps exactly the
  published global bounds; a noisier drive must show a *bigger-than-typical*
  drop to count as degradation.
* The global bounds are the strictest the layer can ever be relaxed to: the
  personalization factor is clamped to ``[1, MAX_STRICTNESS]``. It is a
  **ceiling on sensitivity**, never a relaxation.

Semantics and limitations (verified, see the Task 3 report)
----------------------------------------------------------
* ``adaptive severity <= global severity`` for every observation: the layer can
  only *remove* degradation evidence, never add it. This is an intentional
  ceiling, **not** a guarantee that every global warning is preserved - a
  stricter detector necessarily suppresses some warnings the global detector
  would raise. The invariant is enforced by construction and covered by tests;
  it is *not* presented as a warning-preservation guarantee.
* The layer alone therefore does **not** preserve global warnings. If it is
  ever used as the final decision it must be combined through
  ``app.ml.warning_decision.combine_warnings`` with preservation on - which is
  what the deployed twin does. Hard preservation restores the global
  false-positive rate, i.e. it removes this layer's false-alarm benefit; the
  measured trade-off is in the Task 3 follow-up report.
* Only the mild (severity-1) bound determines whether an observation qualifies,
  because a larger drop still trips the (unrelaxed) higher bounds. The whole
  ladder is scaled for consistency, but the observable effect on the
  qualifying decision is the scaled mild bound.
* ``suggested_persistence`` / ``suggested_min_severity`` intentionally return
  the global operating point unchanged. The previous implementation nudged
  them from the drive's *current* qualifying rate, which rises exactly when a
  drive degrades - i.e. it demanded more evidence precisely when evidence was
  strongest, and cancelled global detections. That coupling is removed.
* The ``acceleration_z`` diagnostic is unchanged: it measures deviation from
  the drive's own recent trend using the full rolling window.

The layer is only meant to *reduce false alarms*; it is not a replacement for
the global detector and it cannot detect degradation the global detector
misses.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from app.ml._component2_paths import get_detector

# Change-signal column names (same semantics as Component 2: observation diffs).
SIGNAL_CHANGE_7D = (
    "smart_1_normalized_change_7d",
    "smart_187_normalized_change_7d",
)

# ----------------------------------------------------------------------
# Fitted population reference scales
# ----------------------------------------------------------------------
#
# Definition: for each change signal, the median across the tuning drives
# (train + validation failed drives and the tuning healthy controls) of the
# per-drive robust scale (MAD * 1.4826) of the drive's *baseline* change
# values, i.e. observations that were not yet globally qualifying.
#
# Fitting procedure: scripts/tune_adaptive_thresholds.py (untracked
# reproduction: c2verify/u5_refscale.py). Measured on ~1057 tuning drives:
#
#   smart_1   : median baseline MAD = 1.4826 (42% of drives are exactly 0)
#   smart_187 : median baseline MAD = 0.0000 (100% of drives are exactly 0)
#
# ``None`` disables per-drive tightening for that signal: there is no
# calibration evidence that any drive's baseline noise is above zero, so any
# positive MAD would be "noisier than the entire population" and pin straight
# to the cap. smart_187 is therefore left exactly as the global detector.
#
# These are *tuned constants*, like MIN_SEVERITY / PERSISTENCE in
# Component 2's degradation_detector.py; they must be re-fitted - on training /
# baseline data only - if the SMART pipeline changes.
REFERENCE_SCALES: Dict[str, Optional[float]] = {
    "smart_1": 1.4826,
    "smart_187": None,
}

# Strictness is capped so an extremely noisy drive cannot become undetectable.
MAX_STRICTNESS = 2.0

# Minimum number of baseline observations before a drive-specific scale is
# trusted. Below this the layer defers to the global detector (factor 1.0).
BASELINE_WARMUP = 7

# Robust scale floor: a MAD at or below this is treated as "constant signal"
# (no calibration evidence) and the global detector is used unchanged.
_SCALE_FLOOR = 1e-6

# Robust z bands for the *acceleration* diagnostic (deviation from own trend).
Z_SEVERITY_BANDS: List[Tuple[float, int]] = [
    (-4.0, 4),
    (-3.0, 3),
    (-2.0, 2),
    (-1.0, 1),
]


def _median(values: Sequence[float]) -> float:
    return float(np.median(values)) if len(values) else 0.0


def _mad(values: Sequence[float]) -> float:
    """Median absolute deviation, scaled to be a consistent sigma estimator."""
    if not len(values):
        return 0.0
    med = float(np.median(values))
    deviations = [abs(v - med) for v in values]
    return float(np.median(deviations)) * 1.4826


def _finite(value: Optional[float]) -> Optional[float]:
    """Return ``value`` as a finite float, or ``None`` when it is not usable."""
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def fit_reference_scales(
    per_drive_baselines: Mapping[str, Sequence[Sequence[float]]],
    quantile: float = 0.5,
) -> Dict[str, Optional[float]]:
    """Fit a population reference scale per signal from baseline samples.

    ``per_drive_baselines`` maps a signal name to a sequence of per-drive
    baseline change values. The reference is the ``quantile`` of the per-drive
    MADs. A signal whose fitted scale is not strictly positive (no drive has
    measurable baseline noise) is returned as ``None`` to disable per-drive
    tightening.

    Must be called with training / baseline drives only - never the final
    evaluation cohort.
    """
    fitted: Dict[str, Optional[float]] = {}
    for signal, drives in per_drive_baselines.items():
        mads = [
            _mad(list(values))
            for values in drives
            if len(values) >= BASELINE_WARMUP
        ]
        if not mads:
            fitted[signal] = None
            continue
        scale = float(np.quantile(mads, quantile))
        fitted[signal] = round(scale, 4) if scale > _SCALE_FLOOR else None
    return fitted


def _severity_from_z(z: float) -> int:
    for bound, level in Z_SEVERITY_BANDS:
        if z <= bound:
            return level
    return 0


@dataclass
class AdaptiveThresholds:
    """Adaptive, per-drive degradation threshold state.

    ``reference_scales`` defaults to the fitted ``REFERENCE_SCALES``. Pass an
    explicit mapping (or ``None`` as a value) to disable personalization for a
    signal.
    """

    serial_number: str
    window: int = 60
    baseline_warmup: int = BASELINE_WARMUP
    reference_scales: Optional[Dict[str, Optional[float]]] = None
    max_strictness: float = MAX_STRICTNESS
    _changes_1: Deque[float] = field(default_factory=lambda: deque(maxlen=60))
    _changes_187: Deque[float] = field(default_factory=lambda: deque(maxlen=60))
    _baseline_1: Deque[float] = field(default_factory=lambda: deque(maxlen=60))
    _baseline_187: Deque[float] = field(default_factory=lambda: deque(maxlen=60))
    _qualifying_history: Deque[int] = field(default_factory=lambda: deque(maxlen=60))

    def __post_init__(self) -> None:
        self._changes_1 = deque(self._changes_1, maxlen=self.window)
        self._changes_187 = deque(self._changes_187, maxlen=self.window)
        self._baseline_1 = deque(self._baseline_1, maxlen=self.window)
        self._baseline_187 = deque(self._baseline_187, maxlen=self.window)
        self._qualifying_history = deque(
            self._qualifying_history, maxlen=self.window
        )
        if self.reference_scales is None:
            self.reference_scales = dict(REFERENCE_SCALES)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def observe(
        self,
        change_1: Optional[float],
        change_187: Optional[float],
        globally_qualifying: bool,
    ) -> None:
        """Feed one observation's change values into the baseline windows.

        The full window always receives the observation (used by the
        acceleration diagnostic). The baseline window - which drives the
        personalization factor - only receives observations that are *not*
        already globally qualifying, so a drive's own degradation never
        inflates its noise estimate.
        """
        c1 = _finite(change_1)
        c187 = _finite(change_187)

        if c1 is not None:
            self._changes_1.append(c1)
        if c187 is not None:
            self._changes_187.append(c187)

        if not globally_qualifying:
            if c1 is not None:
                self._baseline_1.append(c1)
            if c187 is not None:
                self._baseline_187.append(c187)

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

    def baseline(self, signal: str) -> Deque[float]:
        """The baseline window for ``"smart_1"`` / ``"smart_187"``."""
        return self._baseline_1 if signal == "smart_1" else self._baseline_187

    def reference_scale(self, signal: str) -> Optional[float]:
        return (self.reference_scales or {}).get(signal)

    def strictness(self, signal: str) -> float:
        """Personalization factor for ``signal``.

        1.0 = exactly the published detector (the drive is no noisier than the
        population, or there is not enough baseline data). Grows toward
        ``MAX_STRICTNESS`` for drives whose baseline change signal is noisier
        than the population reference.
        """
        reference = self.reference_scale(signal)
        if reference is None or reference <= _SCALE_FLOOR:
            return 1.0
        data = list(self.baseline(signal))
        if len(data) < self.baseline_warmup:
            return 1.0
        scale = _mad(data)
        if scale <= _SCALE_FLOOR:
            return 1.0
        cap = max(float(self.max_strictness), 1.0)
        return float(min(max(scale / reference, 1.0), cap))

    # ------------------------------------------------------------------
    # Severity (global detector as the ceiling on sensitivity)
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
        value = _finite(value)
        if value is None:
            return 0
        for bound, level in bounds:
            # Bounds are negative (a drop). Scaling by strictness >= 1 moves
            # them further from zero -> harder to trip -> fewer false alarms.
            if value <= bound * strictness:
                return level
        return 0

    def severity(
        self,
        change_1: Optional[float],
        change_187: Optional[float],
    ) -> int:
        """Personalized severity (0-4); never exceeds the global severity."""
        sev_1 = self._severity_from_bounds(
            change_1, self._global_bounds("smart_1"), self.strictness("smart_1")
        )
        sev_187 = self._severity_from_bounds(
            change_187,
            self._global_bounds("smart_187"),
            self.strictness("smart_187"),
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
            value = _finite(value)
            center = self._center(values)
            scale = self._scale(values)
            if value is None:
                return None
            if scale <= _SCALE_FLOOR:
                return None
            return round(float((value - center) / scale), 3)

        candidates = (
            z(change_1, self._changes_1),
            z(change_187, self._changes_187),
        )
        zs = [x for x in candidates if x is not None]
        return min(zs) if zs else None

    # ------------------------------------------------------------------
    # Operating point
    # ------------------------------------------------------------------
    def observed_qualifying_rate(self) -> Optional[float]:
        if not self._qualifying_history:
            return None
        return float(np.mean(self._qualifying_history))

    def suggested_persistence(self, base_persistence: int = 7) -> int:
        """Return the *global* persistence requirement unchanged.

        Kept as part of the public interface for the twin service. The audit
        found that adapting persistence from the drive's current qualifying
        rate raised the evidence bar exactly when a drive was degrading, which
        cancelled completed global detections. Personalization now lives
        entirely in the (clamped, monotone) threshold factor.
        """
        return int(base_persistence)

    def suggested_min_severity(self, base_min_severity: int = 1) -> int:
        """Return the *global* minimum severity unchanged (see audit notes)."""
        return int(base_min_severity)

    # ------------------------------------------------------------------
    # Persistence (state survives application restarts)
    # ------------------------------------------------------------------
    def to_state(self) -> Dict[str, object]:
        """Serialisable state of the baseline windows (persistence payload).

        Only the *learned* state is exported - the observation windows that
        drive the personalization factor, the full change windows used by the
        acceleration diagnostic and the qualifying history - plus the
        configuration the restore must agree with.  ``snapshot`` (below) stays
        the human-facing diagnostics view.
        """
        reference_scales = {
            k: (None if v is None else float(v))
            for k, v in (self.reference_scales or {}).items()
        }
        return {
            "version": 1,
            "serial_number": self.serial_number,
            "window": int(self.window),
            "baseline_warmup": int(self.baseline_warmup),
            "max_strictness": float(self.max_strictness),
            "reference_scales": reference_scales,
            "changes_1": list(self._changes_1),
            "changes_187": list(self._changes_187),
            "baseline_1": list(self._baseline_1),
            "baseline_187": list(self._baseline_187),
            "qualifying_history": [int(x) for x in self._qualifying_history],
        }

    def load_state(self, state: Mapping[str, object]) -> None:
        """Restore a payload produced by :meth:`to_state`.

        Defensive: unknown / malformed entries are ignored rather than raising,
        so a corrupt or older record cannot take the twin down on startup.
        """
        if not isinstance(state, Mapping):
            return

        def _numbers(values):
            out = []
            for v in values or []:
                f = _finite(v)
                if f is not None:
                    out.append(f)
            return out

        def _ints(values):
            out = []
            for v in values or []:
                try:
                    out.append(1 if int(v) else 0)
                except (TypeError, ValueError):
                    continue
            return out

        self._changes_1 = deque(
            _numbers(state.get("changes_1")), maxlen=self.window
        )
        self._changes_187 = deque(
            _numbers(state.get("changes_187")), maxlen=self.window
        )
        self._baseline_1 = deque(
            _numbers(state.get("baseline_1")), maxlen=self.window
        )
        self._baseline_187 = deque(
            _numbers(state.get("baseline_187")), maxlen=self.window
        )
        self._qualifying_history = deque(
            _ints(state.get("qualifying_history")), maxlen=self.window
        )
        scales = state.get("reference_scales")
        if isinstance(scales, Mapping):
            restored = {}
            for key, value in scales.items():
                restored[str(key)] = (
                    None if value is None else _finite(value)
                )
            self.reference_scales = restored

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, object]:
        return {
            "serial_number": self.serial_number,
            "baseline_samples": len(self._baseline_1) + len(self._baseline_187),
            "window_samples": len(self._changes_1) + len(self._changes_187),
            "noise_source": "baseline (non-degrading observations)",
            "baseline_warmup": int(self.baseline_warmup),
            "smart_1_change_center": round(self._center(self._baseline_1), 4),
            "smart_1_change_noise": round(self._scale(self._baseline_1), 4),
            "smart_1_reference_scale": self.reference_scale("smart_1"),
            "smart_1_strictness": round(self.strictness("smart_1"), 3),
            "smart_187_change_center": round(self._center(self._baseline_187), 4),
            "smart_187_change_noise": round(self._scale(self._baseline_187), 4),
            "smart_187_reference_scale": self.reference_scale("smart_187"),
            "smart_187_strictness": round(self.strictness("smart_187"), 3),
            "strictness_cap": float(self.max_strictness),
            "observed_qualifying_rate": (
                round(self.observed_qualifying_rate(), 4)
                if self.observed_qualifying_rate() is not None
                else None
            ),
            "suggested_persistence": self.suggested_persistence(),
            "suggested_min_severity": self.suggested_min_severity(),
            "is_calibrated": len(self._baseline_1) >= self.baseline_warmup,
        }
