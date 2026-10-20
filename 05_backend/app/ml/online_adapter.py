"""Genuine online adaptation for RUL predictions.

Component 2's BiLSTM is frozen after training: it never learns from outcomes.
This module adds a **lightweight online learner** on top of the frozen model so
the twin genuinely adapts with streamed feedback.

For each drive we fit, *recursively*, a two-parameter correction

    actual_rul  ~=  b0 + b1 * predicted_rul

using **Recursive Least Squares (RLS)** with a forgetting factor.

Repairs applied to the original implementation
----------------------------------------------
1. **Raw and adapted predictions are kept strictly apart.**
   ``update(predicted, actual)`` always receives the *raw* BiLSTM output,
   ``last_predicted`` stores that raw value, and the adapted value is never
   fed back as an observation. ``adapt()`` returns ``raw + correction`` and
   never writes back, so the raw prediction is immutable.

2. **Centred / scaled regressor.**  The regressor is ``phi = [1, z]`` with
   ``z = (x - center) / scale``.  ``center``/``scale`` are fixed when the
   adapter is constructed (fit by :meth:`DualRulAdapter.seed_population` on
   permitted calibration drives only) and are identical for online updates and
   inference, so there is no train/serve skew and evaluation drives are never
   used to fit them.  The prior mean is the *identity* correction
   ``b0 = 0, b1 = 1`` -- expressed in the centred space as
   ``theta_z = [center, scale]`` -- so a cold start reproduces
   ``adapted == raw`` instead of a wild guess.

3. **Covariance-consistent coefficient bounds.**  Coefficients are projected
   onto a feasible box *after* the RLS step.  Whenever a coordinate is
   projected, its prior variance is restored (``P[i,i] >= P0[i,i]``) so the
   estimator can never become permanently locked on a bound with a collapsed
   covariance -- the failure mode that pinned ``theta = [51.6, 0.577]`` for
   170 consecutive updates in the original implementation.

4. **``delta`` is the prior covariance scale:** ``P0 = delta * I`` in the
   centred space.  Smaller ``delta`` trusts the initial (identity) correction
   more; larger ``delta`` lets the first outcomes move the coefficients sooner.

5. **Prequential no-harm guard.**  Every update records the a-priori absolute
   error of the raw prediction *and* of the adapted prediction, both computed
   **before** the new pair moves the parameters (test-then-train).  A per-drive
   correction is only applied (:attr:`OnlineRulAdapter.effective`) when it has
   beaten the raw model on that held-back comparison by at least
   ``MIN_GUARD_GAIN`` over ``MIN_GUARD_UPDATES`` outcomes.  Otherwise
   ``adapt()`` returns the raw prediction unchanged.

6. **A population correction needs a drive-level verdict, not a stream
   verdict.**  Streamed outcomes arrive in per-drive bursts, so an
   in-sample prequential comparison over a handful of outcomes can look
   excellent while the coefficients sit on a bound.  This was measured on the
   real data: 13 streamed outcomes produced a fictitious 65 % guard gain,
   pinned the slope at its lower bound (0.3) and then applied the maximum
   correction to every drive (+20.8 % MAE).  The shared population adapter
   therefore sets :attr:`OnlineRulAdapter.require_validation`, which only
   accepts the drive-level, out-of-sample verdict of
   :meth:`DualRulAdapter.seed_population`; the streaming guard is reserved for
   per-drive adapters, where it genuinely is an out-of-sample test of that
   drive's own remaining windows.  The population fit itself uses plain RLS
   (``forgetting = 1.0``): the population relation is stationary, so every
   calibration outcome should count, whereas forgetting belongs to the
   per-drive tracker.

7. **Numerical safety.**  Missing / non-finite / out-of-domain inputs are
   rejected and counted; the covariance is symmetrised and eigenvalue-floored
   after every step; any numerical failure latches ``failed`` so every later
   call falls back to the raw prediction.
"""

from __future__ import annotations

import hashlib
import math
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Forgetting factor for a *per-drive* tracker: 1.0 = never forget, lower =
# track drift faster.
DEFAULT_FORGETTING = 0.985

# The population (shared) correction is fitted with plain RLS.  The relation
# between predicted and realised RUL across drives is stationary, so every
# calibration outcome should carry equal weight; a forgetting factor
# equivalent to ~2 drives (0.985) makes the population fit track whichever
# drives happened to arrive last instead.
POPULATION_FORGETTING = 1.0

# Prior covariance scale: P0 = delta * I in the centred regressor space.
# Small -> trust the initial (identity) correction more.
DEFAULT_DELTA = 1.0

# Clamp the learned slope so corrections stay sane.
B1_MIN = 0.3
B1_MAX = 3.0

# Hard bound on the correction (days) applied to a single prediction.
# Derived from training evidence: the frozen BiLSTM scores MAE 10.611 d and
# RMSE 12.615 d on its own evaluation split, so a correction larger than
# ~1.4x the model's own MAE cannot be justified by the evidence.  (The
# previous 60 d bound covered the entire RUL label range and therefore
# guaranteed nothing.)
MAX_CORRECTION_DAYS = 15.0

# Relative bound: never move more than 50% of the raw prediction (days).
MAX_CORRECTION_FRACTION = 0.5

# Minimum updates before a personal adapter is trusted over the global one.
PERSONAL_TRUST_UPDATES = 5

# Prequential no-harm guard (see module docstring, item 5).
MIN_GUARD_UPDATES = 10
MIN_GUARD_GAIN = 0.02

# Rolling window (in outcomes) kept for the guard / residual diagnostics.
HISTORY_WINDOW = 500

# RUL label domain observed in this project (labels span 1-59 days).
RUL_LABEL_MAX = 60.0

# Accepted domain for observed (predicted, actual) RUL pairs.  Anything
# outside is treated as an invalid label (e.g. -999 sentinels, 1e6 garbage).
MAX_OBSERVED_RUL = 180.0

# Default regressor scaling used when no calibration data is available.
# Fixed domain constants -- never fitted on evaluation data.
DEFAULT_CENTER = 30.0
DEFAULT_SCALE = 15.0
MIN_SCALE = 1e-3

# Covariance eigenvalue floor (keeps P positive definite).
PSD_FLOOR = 1e-9

# Population seed gate (see DualRulAdapter.seed_population).
SEED_MIN_PAIRS = 20
SEED_FIT_FRACTION = 0.8


def _finite(value) -> Optional[float]:
    """Return ``float(value)`` when it is real and finite, else ``None``."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


class OnlineRulAdapter:
    """Bounded, self-validating RLS correction for predicted RUL."""

    def __init__(
        self,
        forgetting: float = DEFAULT_FORGETTING,
        delta: float = DEFAULT_DELTA,
        center: float = DEFAULT_CENTER,
        scale: float = DEFAULT_SCALE,
        slope_min: float = B1_MIN,
        slope_max: float = B1_MAX,
        max_correction_days: float = MAX_CORRECTION_DAYS,
        max_correction_fraction: float = MAX_CORRECTION_FRACTION,
        enabled: bool = True,
        require_validation: bool = False,
    ) -> None:
        if not (0.0 < float(forgetting) <= 1.0):
            raise ValueError("forgetting must lie in (0, 1]")
        if not (float(delta) > 0.0) or not math.isfinite(float(delta)):
            raise ValueError("delta must be a finite value > 0")

        self.forgetting = float(forgetting)
        self.delta = float(delta)
        self.center = float(center)
        self.scale = float(scale)
        if not math.isfinite(self.scale) or abs(self.scale) < MIN_SCALE:
            self.scale = DEFAULT_SCALE
        if not math.isfinite(self.center):
            self.center = DEFAULT_CENTER

        self.slope_min = float(slope_min)
        self.slope_max = float(slope_max)
        self.max_correction_days = float(max_correction_days)
        self.max_correction_fraction = float(max_correction_fraction)
        self.enabled = bool(enabled)

        # When set, the streaming prequential guard is *not* enough: the
        # correction may only be applied after the drive-level, out-of-sample
        # gate of DualRulAdapter.seed_population has accepted it.  Used for the
        # shared population adapter, whose correction reaches every drive.
        self.require_validation = bool(require_validation)

        # Out-of-sample gate, set by DualRulAdapter.seed_population.
        self.validated = False

        self.n_updates = int(0)
        self.n_rejected = int(0)
        self._failed = bool(False)

        # Signed a-priori (test-then-train) errors.
        self._residuals: deque = deque(maxlen=HISTORY_WINDOW)
        self._raw_residuals: deque = deque(maxlen=HISTORY_WINDOW)

        # Raw BiLSTM output of the most recent consumed outcome (immutable).
        self.last_predicted: Optional[float] = None
        self.last_actual: Optional[float] = None

        # Centred-space coefficients and covariance.
        self._p0: np.ndarray = np.eye(2, dtype=float) * self.delta
        self._theta: np.ndarray = self._identity_theta()
        self._p: np.ndarray = self._p0.copy()

    # ------------------------------------------------------------------
    # Parameterisation helpers
    # ------------------------------------------------------------------
    def _identity_theta(self) -> np.ndarray:
        """Centred coefficients that reproduce the raw prediction exactly."""
        return np.array([self.center, self.scale], dtype=float)

    def _phi(self, x: float) -> np.ndarray:
        """Regressor ``[1, (x - center) / scale]`` (same at fit and inference)."""
        return np.array([1.0, (x - self.center) / self.scale], dtype=float)

    def _model(self, x: float) -> float:
        """Model output in RUL days for raw prediction ``x`` (uncapped)."""
        return float(self._theta @ self._phi(x))

    def _project(
        self, theta_u: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Project centred coefficients onto a well-conditioned feasible box.

        Bounds are applied where they are meaningful:

        * ``a0`` is the predicted *actual* RUL at a raw prediction of
          ``center`` -- a physical quantity, bounded by the label domain plus
          the correction allowance.  (The original code bounded the intercept
          at a raw prediction of 0, i.e. a violent extrapolation far outside
          the observed range, which is ill-conditioned and caused corner
          hopping between bounds.)
        * ``a1`` is the slope ``b1`` expressed in centred units
          (``a1 = b1 * scale``), so the documented ``b1`` limits map onto an
          axis-aligned bound.

        Returns the projected coefficients and a mask of *active* constraints.
        """
        a0, a1 = float(theta_u[0]), float(theta_u[1])
        a0_new = float(
            np.clip(a0, -self.max_correction_days,
                    RUL_LABEL_MAX + self.max_correction_days)
        )
        a1_new = float(
            np.clip(a1, self.slope_min * self.scale, self.slope_max * self.scale)
        )

        active = np.array([a0_new != a0, a1_new != a1], dtype=bool)
        if not active.any():
            return theta_u, active
        return np.array([a0_new, a1_new], dtype=float), active

    def _reopen_active(self, p: np.ndarray, active: np.ndarray) -> np.ndarray:
        """Restore prior variance along the normal of every active constraint.

        Both constraints are axis-aligned in the centred space, so this is a
        diagonal repair.  While a bound is active the estimate is *not* the
        unconstrained optimum, so the posterior variance in that direction must
        not fall below its prior value -- otherwise the gain in the only
        direction that can leave the boundary collapses and the coefficients
        lock in place (the failure the original clamp-only update exhibited).
        """
        for idx in np.flatnonzero(active):
            if p[idx, idx] < self._p0[idx, idx]:
                p[idx, idx] = self._p0[idx, idx]
        return p

    def _symmetrise(self, p: np.ndarray) -> np.ndarray:
        return (p + p.T) / 2.0

    @staticmethod
    def _psd_project(p: np.ndarray) -> np.ndarray:
        """Symmetrise and floor the eigenvalues so P stays positive definite."""
        p = (p + p.T) / 2.0
        if not np.all(np.isfinite(p)):
            return p
        try:
            w, v = np.linalg.eigh(p)
        except np.linalg.LinAlgError:  # pragma: no cover - defensive
            return p
        w = np.maximum(w, PSD_FLOOR)
        return (v * w) @ v.T

    def _fail(self) -> None:
        """Latch a numerical failure: every later call falls back to raw."""
        self._failed = True

    # ------------------------------------------------------------------
    # Public parameter views
    # ------------------------------------------------------------------
    @property
    def theta(self) -> np.ndarray:
        """Coefficients ``[b0, b1]`` of ``actual ~= b0 + b1 * predicted``."""
        a0, a1 = self._theta
        return np.array(
            [a0 - a1 * self.center / self.scale, a1 / self.scale], dtype=float
        )

    @theta.setter
    def theta(self, value) -> None:
        arr = np.asarray(value, dtype=float).ravel()
        if arr.size != 2 or not np.all(np.isfinite(arr)):
            raise ValueError("theta must be two finite values [b0, b1]")
        b0, b1 = float(arr[0]), float(arr[1])
        self._theta = np.array(
            [b0 + b1 * self.center, b1 * self.scale], dtype=float
        )

    @property
    def p(self) -> np.ndarray:
        """Covariance of the *centred* coefficients ``[a0, a1]``."""
        return self._p.copy()

    @p.setter
    def p(self, value) -> None:
        arr = np.asarray(value, dtype=float)
        if arr.shape != (2, 2):
            raise ValueError("p must be a 2x2 matrix")
        self._p = self._psd_project(arr)

    @property
    def residuals(self) -> List[float]:
        """Signed a-priori model errors ``actual - adapted`` (rolling window)."""
        return list(self._residuals)

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------
    @property
    def failed(self) -> bool:
        return self._failed

    @property
    def is_valid(self) -> bool:
        return not self._failed

    @property
    def is_ready(self) -> bool:
        return self.n_updates >= PERSONAL_TRUST_UPDATES

    @property
    def mean_abs_residual(self) -> Optional[float]:
        if not self._residuals:
            return None
        return round(float(np.mean(np.abs(self._residuals))), 3)

    @property
    def raw_mean_abs_error(self) -> Optional[float]:
        if not self._raw_residuals:
            return None
        return float(np.mean(np.abs(self._raw_residuals)))

    @property
    def adapted_mean_abs_error(self) -> Optional[float]:
        if not self._residuals:
            return None
        return float(np.mean(np.abs(self._residuals)))

    @property
    def beneficial(self) -> bool:
        """Did the correction beat the raw prediction on held-back outcomes?"""
        if self.n_updates < MIN_GUARD_UPDATES:
            return False
        raw = self.raw_mean_abs_error
        adapted = self.adapted_mean_abs_error
        if raw is None or adapted is None or raw <= 0.0:
            return False
        return bool(adapted <= raw * (1.0 - MIN_GUARD_GAIN))

    @property
    def effective(self) -> bool:
        """May the learned correction be applied to a prediction?"""
        if self._failed or not self.enabled:
            return False
        if self.require_validation:
            # A correction applied to every drive must clear the drive-level,
            # out-of-sample gate.  The streaming guard is not a substitute: it
            # can be satisfied (with a large fictitious margin) by a short,
            # unrepresentative burst of outcomes.
            return bool(self.validated)
        if self.validated:
            return True
        return self.beneficial

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------
    def update(self, predicted: float, actual: float) -> bool:
        """Consume one labelled outcome.

        ``predicted`` must be the **raw** BiLSTM output for this outcome.
        Returns ``True`` when the pair was accepted and applied.
        """
        if self._failed:
            return False

        x = _finite(predicted)
        d = _finite(actual)
        if x is None or d is None:
            self.n_rejected += 1
            return False
        if not (0.0 <= x <= MAX_OBSERVED_RUL) or not (0.0 <= d <= MAX_OBSERVED_RUL):
            self.n_rejected += 1
            return False

        phi = self._phi(x)
        a_priori = float(self._theta @ phi)
        if not math.isfinite(a_priori):
            self._fail()
            return False

        # Prequential (test-then-train) errors: both scored against the same
        # realisation, *before* this pair moves the parameters.
        self._raw_residuals.append(d - x)
        self._residuals.append(d - a_priori)

        p_phi = self._p @ phi
        denom = self.forgetting + float(phi @ p_phi)
        if not math.isfinite(denom) or denom <= 0.0:
            self._fail()
            return False

        k = p_phi / denom
        theta_unconstrained = self._theta + k * (d - a_priori)
        p_new = (self._p - np.outer(k, phi) @ self._p) / self.forgetting
        p_new = self._symmetrise(p_new)

        theta_projected, active = self._project(theta_unconstrained)
        p_new = self._reopen_active(p_new, active)
        p_new = self._symmetrise(p_new)

        self._theta = theta_projected
        self._p = self._psd_project(p_new)

        if not np.all(np.isfinite(self._theta)) or not np.all(np.isfinite(self._p)):
            self._fail()
            return False

        self.n_updates += 1
        self.last_predicted = x  # raw BiLSTM output, kept immutable
        self.last_actual = d
        return True

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def proposed_correction(self, predicted: float) -> float:
        """Bounded correction the model would like to apply (no guard)."""
        x = _finite(predicted)
        if x is None or self._failed or self.n_updates == 0:
            return 0.0
        value = self._model(x) - x
        if not math.isfinite(value):
            self._fail()
            return 0.0
        cap = min(self.max_correction_days, abs(x) * self.max_correction_fraction)
        return float(np.clip(value, -cap, cap))

    def correction(self, predicted: float) -> float:
        """Correction actually applied (``0.0`` unless the guard allows it)."""
        if not self.effective:
            return 0.0
        return self.proposed_correction(predicted)

    def adapt(self, predicted: float):
        """Return ``max(0, raw + applied_correction)``; raw when not effective."""
        if predicted is None:
            return predicted
        x = _finite(predicted)
        if x is None:
            return predicted
        if not self.effective:
            return float(x)
        return max(0.0, float(x) + self.correction(x))

    def reset(self) -> None:
        """Return to the cold-start (identity) state; keeps scaler/config."""
        self.n_updates = 0
        self.n_rejected = 0
        self._failed = False
        self.validated = False
        self._residuals.clear()
        self._raw_residuals.clear()
        self.last_predicted = None
        self.last_actual = None
        self._theta = self._identity_theta()
        self._p = self._p0.copy()

    def snapshot(self) -> Dict[str, object]:
        theta = self.theta
        raw_mae = self.raw_mean_abs_error
        adapted_mae = self.adapted_mean_abs_error
        return {
            "n_updates": self.n_updates,
            "n_rejected": self.n_rejected,
            "intercept": round(float(theta[0]), 4),
            "slope": round(float(theta[1]), 4),
            "mean_abs_residual": self.mean_abs_residual,
            "raw_mean_abs_error": None if raw_mae is None else round(raw_mae, 4),
            "adapted_mean_abs_error": (
                None if adapted_mae is None else round(adapted_mae, 4)
            ),
            "last_predicted": self.last_predicted,
            "last_actual": self.last_actual,
            "is_ready": self.is_ready,
            "beneficial": self.beneficial,
            "effective": self.effective,
            "validated": self.validated,
            "require_validation": self.require_validation,
            "enabled": self.enabled,
            "failed": self._failed,
            "center": round(self.center, 4),
            "scale": round(self.scale, 4),
            "delta": self.delta,
        }


class DualRulAdapter:
    """Per-serial adapter with a shared global (population) adapter."""

    def __init__(
        self,
        forgetting: float = DEFAULT_FORGETTING,
        delta: float = DEFAULT_DELTA,
        center: float = DEFAULT_CENTER,
        scale: float = DEFAULT_SCALE,
        enabled: bool = True,
    ) -> None:
        self.forgetting = float(forgetting)
        self.delta = float(delta)
        self.center = float(center)
        self.scale = float(scale)
        self.enabled = bool(enabled)
        self._initial_scaler = (self.center, self.scale)
        self.global_adapter = self._new_adapter(population=True)
        self.serial_adapters: Dict[str, OnlineRulAdapter] = {}
        self.seed_diagnostics: Optional[dict] = None

    # ------------------------------------------------------------------
    def _new_adapter(self, population: bool = False) -> OnlineRulAdapter:
        """Build an adapter: the population one, or a per-drive one.

        The population adapter uses plain RLS and only trusts the drive-level
        seed gate; per-drive adapters use the configured forgetting factor and
        the streaming prequential guard.
        """
        return OnlineRulAdapter(
            forgetting=POPULATION_FORGETTING if population else self.forgetting,
            delta=self.delta,
            center=self.center,
            scale=self.scale,
            enabled=self.enabled,
            require_validation=population,
        )

    def _get(self, serial: str) -> OnlineRulAdapter:
        if serial not in self.serial_adapters:
            self.serial_adapters[serial] = self._new_adapter()
        return self.serial_adapters[serial]

    def update(self, serial: str, predicted: float, actual: float) -> bool:
        """Feed one (raw prediction, realised RUL) pair to personal + global."""
        personal_ok = self._get(serial).update(predicted, actual)
        global_ok = self.global_adapter.update(predicted, actual)
        return personal_ok or global_ok

    def adapt(self, serial: str, predicted: float) -> Dict[str, object]:
        """Correct ``predicted`` (raw) for ``serial`` and report the decision."""
        personal = self._get(serial)
        global_adapter = self.global_adapter

        if personal.effective:
            chosen, source, reason = (
                personal, "personal", "personal_prequential_gain"
            )
        elif global_adapter.effective:
            chosen = global_adapter
            source = "global"
            reason = (
                "population_validated"
                if global_adapter.validated
                else "global_prequential_gain"
            )
        elif personal.failed or global_adapter.failed:
            chosen, source, reason = None, "none", "numerical_failure"
        elif not (personal.enabled or global_adapter.enabled):
            chosen, source, reason = None, "none", "disabled"
        elif personal.n_rejected or global_adapter.n_rejected:
            chosen, source, reason = None, "none", "invalid_observations_only"
        elif global_adapter.require_validation and not global_adapter.validated:
            chosen, source, reason = None, "none", "population_not_validated"
        else:
            chosen, source, reason = None, "none", "no_valid_evidence"

        if predicted is None:
            raw_value = None
            adapted_value = None
            correction_days = None
        else:
            raw_value = round(float(predicted), 3)
            adapted_value = (
                round(chosen.adapt(float(predicted)), 3)
                if chosen is not None
                else raw_value
            )
            correction_days = round(adapted_value - raw_value, 3)

        return {
            "raw": raw_value,
            "adapted": adapted_value,
            "correction_days": correction_days,
            "source": source,
            "reason": reason,
            "enabled": chosen is not None,
            "personal_updates": personal.n_updates,
            "global_updates": global_adapter.n_updates,
            "personal": personal.snapshot(),
            "global": global_adapter.snapshot(),
        }

    # ------------------------------------------------------------------
    def seed_population(
        self,
        pairs: Sequence[Tuple[float, float]],
        serials: Optional[Sequence[Optional[str]]] = None,
        min_gain: float = MIN_GUARD_GAIN,
        fit_fraction: float = SEED_FIT_FRACTION,
    ) -> dict:
        """Fit the global correction on permitted calibration drives.

        * Invalid / out-of-domain pairs are dropped and counted.
        * ``center``/``scale`` are fitted on the *fit* rows only.
        * The correction is **activated only if it beats the raw prediction
          out-of-sample** on a held-back, drive-level split.  When it does not,
          the global adapter is reset to the cold-start identity state, which
          leaves ``adapt() == raw`` (safe fallback).

        The caller supplies the pairs; they must come from calibration data
        that is disjoint from any evaluation set.
        """
        pairs = list(pairs)
        if serials is None:
            serial_list: List[Optional[str]] = [None] * len(pairs)
        else:
            serial_list = list(serials)
            if len(serial_list) != len(pairs):
                raise ValueError("serials must match pairs in length")

        valid: List[Tuple[float, float, Optional[str]]] = []
        rejected = 0
        for (pred, actual), serial in zip(pairs, serial_list):
            x = _finite(pred)
            d = _finite(actual)
            if (
                x is None
                or d is None
                or not (0.0 <= x <= MAX_OBSERVED_RUL)
                or not (0.0 <= d <= MAX_OBSERVED_RUL)
            ):
                rejected += 1
                continue
            valid.append((x, d, serial))

        diagnostics: dict = {
            "pairs_supplied": len(pairs),
            "pairs_rejected": rejected,
            "min_gain": min_gain,
            "status": "insufficient_pairs",
        }

        if len(valid) >= SEED_MIN_PAIRS:
            fit_idx, val_idx = self._split(valid, fit_fraction)
        else:
            fit_idx, val_idx = [], []

        if not fit_idx or not val_idx:
            self.global_adapter = self._new_adapter(population=True)
            self.seed_diagnostics = diagnostics
            return diagnostics

        fit = [valid[i] for i in fit_idx]
        holdout = [valid[i] for i in val_idx]

        # Scaler from the fit split only (never from evaluation data).
        predictions = np.array([row[0] for row in fit], dtype=float)
        center = float(predictions.mean())
        scale = float(predictions.std())
        if not math.isfinite(center):
            center = DEFAULT_CENTER
        if not math.isfinite(scale) or scale < MIN_SCALE:
            scale = DEFAULT_SCALE
        self.center, self.scale = center, scale

        adapter = self._new_adapter(population=True)
        for x, d, _ in fit:
            adapter.update(x, d)

        raw_mae = float(np.mean([abs(d - x) for x, d, _ in holdout]))
        adapted_mae = float(
            np.mean(
                [abs(d - max(0.0, x + adapter.proposed_correction(x)))
                 for x, d, _ in holdout]
            )
        )
        gain = (raw_mae - adapted_mae) / raw_mae if raw_mae > 0.0 else 0.0
        validated = bool(adapted_mae <= raw_mae * (1.0 - min_gain))

        if validated:
            adapter.validated = True
            self.global_adapter = adapter
            status = "validated"
        else:
            # Safe fallback: keep the fitted scaler (the identity correction is
            # parameterisation-invariant) but discard the correction itself.
            self.global_adapter = self._new_adapter(population=True)
            status = "rejected_no_gain"

        self.seed_diagnostics = {
            **diagnostics,
            "status": status,
            "fit_pairs": len(fit),
            "holdout_pairs": len(holdout),
            "fit_drives": len({row[2] for row in fit if row[2] is not None}),
            "holdout_drives": len(
                {row[2] for row in holdout if row[2] is not None}
            ),
            "center": round(center, 4),
            "scale": round(scale, 4),
            "holdout_raw_mae": round(raw_mae, 4),
            "holdout_adapted_mae": round(adapted_mae, 4),
            "holdout_gain": round(gain, 4),
        }
        return self.seed_diagnostics

    @staticmethod
    def _split(
        rows: Sequence[Tuple[float, float, Optional[str]]],
        fit_fraction: float,
    ) -> Tuple[List[int], List[int]]:
        """Deterministic drive-level split (serials never cross the boundary)."""
        serials = [row[2] for row in rows]
        if all(s is not None for s in serials) and len(set(serials)) >= 4:
            threshold = int(round(fit_fraction * 100))
            fit_idx = [
                i
                for i, serial in enumerate(serials)
                if int(
                    hashlib.sha256(serial.encode("utf-8")).hexdigest(), 16
                )
                % 100
                < threshold
            ]
            if fit_idx:
                chosen = set(fit_idx)
                val_idx = [i for i in range(len(rows)) if i not in chosen]
                if val_idx:
                    return fit_idx, val_idx

        # Fallback: deterministic interleaved index split.
        threshold = int(round(fit_fraction * 100))
        fit_idx = [i for i in range(len(rows)) if i % 100 < threshold]
        chosen = set(fit_idx)
        val_idx = [i for i in range(len(rows)) if i not in chosen]
        return fit_idx, val_idx

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear all learned state (global + every personal adapter)."""
        self.center, self.scale = self._initial_scaler
        self.global_adapter = self._new_adapter(population=True)
        self.serial_adapters.clear()
        self.seed_diagnostics = None

    def set_enabled(self, enabled: bool) -> None:
        """Kill switch: with ``False`` every correction is switched off.

        The adapters keep learning (so turning it back on resumes from the
        current evidence) but ``adapt()`` returns the raw prediction.
        """
        self.enabled = bool(enabled)
        self.global_adapter.enabled = self.enabled
        for adapter in self.serial_adapters.values():
            adapter.enabled = self.enabled

    def snapshot(self) -> Dict[str, object]:
        return {
            "global": self.global_adapter.snapshot(),
            "serials_tracked": len(self.serial_adapters),
            "serials_ready": sum(
                1 for a in self.serial_adapters.values() if a.is_ready
            ),
            "serials_effective": sum(
                1 for a in self.serial_adapters.values() if a.effective
            ),
            "scaler": {
                "center": round(self.center, 4),
                "scale": round(self.scale, 4),
            },
            "seed": self.seed_diagnostics,
        }
