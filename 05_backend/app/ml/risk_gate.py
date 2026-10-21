"""Risk-gated, degradation-aware adaptation controller (Component 2 novelty).

The other adaptive capabilities answer *how* the twin adapts.  This module
answers *whether it should*.  Every batch of realised RUL outcomes ends in
exactly one action:

    WAIT      not enough evidence yet -- production is unchanged while a
              challenger correction keeps learning in the shadow,
    ADAPT     the evidence test has passed (or adaptation is already
              approved and still safe) -- the update reaches production,
    REJECT    the shadow challenger is provably worse than production ->
              it is discarded and reset to the production version,
    ROLLBACK  live adaptation has become provably worse than the last
              known-good checkpoint -> production is restored to that
              checkpoint and adaptation returns to the shadow.

Three decision factors
----------------------
1. **Degradation-aware safety.**  In an HDD a change in SMART behaviour is
   usually *the failure itself*.  Every window is labelled with the production
   degradation detector (severity >= ``MIN_SEVERITY``).  For a *degrading*
   drive an update may only make the RUL more conservative -- adaptation can
   never postpone a warning -- and degrading windows are excluded from the
   drift score, so the failure signal is never mistaken for drift.

2. **Uncertainty-aware, anytime-valid acceptance.**  Instead of a fixed
   "gain >= x %" rule, every decision is a sequential test with an e-process
   (testing by betting).  For a batch with mean paired loss difference ``d``
   the bound ``C = max_i |prediction_A_i - prediction_B_i|`` is computed from
   the two models' predictions *before* the labels are used (``|d| <= C`` by
   the triangle inequality), so it is predictable and

       E  <-  E * (1 + lambda * d / C),     0 <= lambda < 1.

   The bet ``lambda`` is the aGRAPA plug-in ``mean / (mean^2 + var)`` of the
   last ``BET_WINDOW`` *previous* batches' ratios ``d / C`` (clipped to a
   cap), so it is also
   predictable: with no edge the twin stops betting, with a clear edge it
   bets close to the cap.

   Under the null hypothesis "the update is not better" ``E`` is a
   non-negative supermartingale, so by Ville's inequality

       P( sup_t E_t >= 1/alpha )  <=  alpha.

   The probability of ever accepting a non-beneficial update is bounded by
   ``alpha`` *at every stopping time*: the twin may check after every batch
   without inflating the error.  WAIT means the evidence has not crossed
   ``1/alpha`` yet.  The mirror-image e-processes for REJECT and ROLLBACK make
   the same guarantee about harm.  Low-quality observations are not evidence.

3. **SMART behavioural drift caps the bet.**  A drift score compares the
   *operating* SMART signature (``smart_1`` level and volatility, stable
   windows only) of a batch with the drive-to-drive spread of the reference
   population.  It is computed from features only, before the labels are used,
   so it may set the bet cap without breaking the guarantee: under drift the
   twin may bet harder (adapts sooner), otherwise it bets cautiously.

The unit of evidence is a *batch*: in the offline evaluation one batch is one
failed drive (its RUL labels only exist after the failure), in the live
service one ``learn`` call.
"""

from __future__ import annotations

import copy
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from app.ml.online_adapter import (
    DEFAULT_CENTER,
    DEFAULT_SCALE,
    OnlineRulAdapter,
)

ADAPT = "ADAPT"
WAIT = "WAIT"
REJECT = "REJECT"
ROLLBACK = "ROLLBACK"

SHADOW = "shadow"
ACTIVE = "active"

# Significance level of every sequential test: P(false acceptance) <= ALPHA.
DEFAULT_ALPHA = 0.05

# Bet caps.  Promotion bets cautiously unless operating drift is detected;
# the safety tests (reject / rollback) may always bet hard so protection is fast.
LAMBDA_BASE = 0.5
LAMBDA_DRIFT = 0.9
LAMBDA_SAFETY = 0.9

# Floor on the per-batch bound C (days): below this the two models are
# treated as identical and the batch carries no evidence.
MIN_BOUND_DAYS = 0.25

# Observations with a data-quality score below this are not evidence.
MIN_QUALITY = 0.6

# Operating-signature drift: |batch mean - reference mean| / reference spread.
DRIFT_Z = 2.0
OPERATING_FEATURES = (
    "smart_1_normalized",
    "smart_1_normalized_rolling_std_7",
)

# The challenger tracks the fleet with mild forgetting (~ last 200 windows).
CHALLENGER_FORGETTING = 0.995

# Past batches used to size the next bet.  A rolling window lets the bet
# recover when a challenger that started out harmful has since improved.
BET_WINDOW = 10

DECISION_LOG = 200


@dataclass
class Outcome:
    """One labelled window: raw BiLSTM prediction and its realised RUL."""

    predicted: float
    actual: float
    degrading: bool = False
    quality: float = 1.0
    features: Dict[str, float] = field(default_factory=dict)


def _bet(history: Sequence[float], cap: float) -> float:
    """aGRAPA plug-in bet from *past* ratios only (keeps E a supermartingale)."""
    if not history:
        return 0.5 * cap
    history = history[-BET_WINDOW:]
    mean = float(np.mean(history))
    if mean <= 0.0:
        return 0.0
    return float(min(cap, mean / (mean * mean + float(np.var(history)))))


def _finite(value) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


class RiskGatedAdapter:
    """Shadow / active RUL correction with an ADAPT/WAIT/REJECT/ROLLBACK gate."""

    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA,
        center: float = DEFAULT_CENTER,
        scale: float = DEFAULT_SCALE,
        degradation_aware: bool = True,
        drift_aware: bool = True,
    ) -> None:
        if not 0.0 < float(alpha) < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        self.alpha = float(alpha)
        self.threshold = 1.0 / self.alpha
        self.center = float(center)
        self.scale = float(scale)
        self.degradation_aware = bool(degradation_aware)
        self.drift_aware = bool(drift_aware)
        self.reference: Dict[str, tuple] = {}
        self.reset()

    # ------------------------------------------------------------------
    def _identity(self) -> OnlineRulAdapter:
        adapter = OnlineRulAdapter(
            forgetting=CHALLENGER_FORGETTING,
            center=self.center,
            scale=self.scale,
        )
        # The gate, not the adapter's own guard, decides what is deployed.
        adapter.validated = True
        return adapter

    def reset(self) -> None:
        """Back to the raw model; keeps the drift reference."""
        self.mode = SHADOW
        # Production when the mode is SHADOW, and the rollback target.
        self.checkpoint: OnlineRulAdapter = self._identity()
        self.challenger: OnlineRulAdapter = self._identity()
        self.version = 0
        self.e_promote = 1.0
        self.e_reject = 1.0
        self.e_rollback = 1.0
        self._history: Dict[str, List[float]] = {
            "promote": [], "reject": [], "rollback": []
        }
        self.counts = {ADAPT: 0, WAIT: 0, REJECT: 0, ROLLBACK: 0}
        self.log: deque = deque(maxlen=DECISION_LOG)

    @property
    def production(self) -> OnlineRulAdapter:
        return self.challenger if self.mode == ACTIVE else self.checkpoint

    @property
    def deployed(self) -> bool:
        """True while adaptation is approved and reaching production."""
        return self.mode == ACTIVE or self.version > 0

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def _correction(
        self, adapter: OnlineRulAdapter, raw: float, degrading: bool
    ) -> float:
        value = adapter.proposed_correction(raw)
        # Degradation-aware asymmetry: never let adaptation raise the RUL of a
        # drive that is degrading (it would postpone the warning).
        if self.degradation_aware and degrading and value > 0.0:
            return 0.0
        return value

    def _predict(
        self, adapter: Optional[OnlineRulAdapter], raw: float, degrading: bool
    ) -> float:
        if adapter is None:
            return raw
        return max(0.0, raw + self._correction(adapter, raw, degrading))

    def apply(self, predicted: float, degrading: bool = False) -> Dict[str, object]:
        """Production RUL for a raw prediction (pure read, never trains)."""
        raw = _finite(predicted)
        base = {"mode": self.mode, "version": self.version,
                "deployed": self.deployed}
        if raw is None:
            return {**base, "raw": None, "adapted": None,
                    "correction_days": None, "blocked_optimistic": False}
        adapted = self._predict(self.production, raw, degrading)
        return {
            **base,
            "raw": round(raw, 3),
            "adapted": round(adapted, 3),
            "correction_days": round(adapted - raw, 3),
            "blocked_optimistic": bool(
                self.degradation_aware
                and degrading
                and self.production.proposed_correction(raw) > 0.0
            ),
        }

    # ------------------------------------------------------------------
    # Drift
    # ------------------------------------------------------------------
    def _stable_means(self, outcomes: Sequence[Outcome]) -> Dict[str, float]:
        means = {}
        for name in OPERATING_FEATURES:
            values = [
                o.features[name] for o in outcomes
                if not (self.degradation_aware and o.degrading)
                and _finite(o.features.get(name)) is not None
            ]
            if values:
                means[name] = float(np.mean(values))
        return means

    def set_reference(self, batches: Sequence[Sequence[Outcome]]) -> None:
        """Reference = drive-to-drive spread of the operating signature."""
        per_feature: Dict[str, List[float]] = {n: [] for n in OPERATING_FEATURES}
        for batch in batches:
            for name, value in self._stable_means(batch).items():
                per_feature[name].append(value)
        self.reference = {
            name: (float(np.mean(v)), max(float(np.std(v)), 1e-6))
            for name, v in per_feature.items()
            if len(v) >= 5
        }

    def drift_score(self, outcomes: Sequence[Outcome]) -> Optional[float]:
        """Largest |z| of the batch's operating signature (labels unused)."""
        means = self._stable_means(outcomes)
        scores = [
            abs(means[name] - mean) / std
            for name, (mean, std) in self.reference.items()
            if name in means
        ]
        return max(scores) if scores else None

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------
    def seed(self, batches: Sequence[Sequence[Outcome]]) -> int:
        """Pre-train the shadow challenger on calibration data (not deployed)."""
        self.set_reference(batches)
        return sum(self._train(self.challenger, batch) for batch in batches)

    @staticmethod
    def _usable(outcomes: Sequence[Outcome]) -> List[Outcome]:
        usable = []
        for o in outcomes:
            x, y, q = _finite(o.predicted), _finite(o.actual), _finite(o.quality)
            if x is None or y is None or x < 0.0 or y < 0.0:
                continue
            if q is not None and q < MIN_QUALITY:
                continue
            usable.append(o)
        return usable

    def _train(self, adapter: OnlineRulAdapter, outcomes: Sequence[Outcome]) -> int:
        return sum(
            1 for o in self._usable(outcomes) if adapter.update(o.predicted, o.actual)
        )

    def _gain(self, base, other, outcomes: Sequence[Outcome]) -> Optional[tuple]:
        """``(mean |y - base| - |y - other|, bound C)``; positive: ``other`` better.

        ``C`` is the largest gap between the two models' predictions on the
        batch.  It uses no labels, so it is a predictable bound on the gain.
        """
        if not outcomes:
            return None
        gains, gaps = [], []
        for o in outcomes:
            b = self._predict(base, o.predicted, o.degrading)
            c = self._predict(other, o.predicted, o.degrading)
            gains.append(abs(o.actual - b) - abs(o.actual - c))
            gaps.append(abs(b - c))
        bound = max(gaps)
        if bound < MIN_BOUND_DAYS:
            return None
        return float(np.clip(np.mean(gains), -bound, bound)), float(bound)

    def _reset_tests(self) -> None:
        """Start fresh sequential tests (after any state change)."""
        self.e_promote = self.e_reject = self.e_rollback = 1.0
        for history in self._history.values():
            history.clear()

    def observe(self, outcomes: Sequence[Outcome]) -> Dict[str, object]:
        """Consume one batch of labelled outcomes and return the decision."""
        usable = self._usable(outcomes)
        degrading = sum(1 for o in usable if o.degrading)

        drift = self.drift_score(usable) if self.drift_aware else None
        drift_flag = drift is not None and drift >= DRIFT_Z
        cap = LAMBDA_DRIFT if drift_flag else LAMBDA_BASE

        # Test-then-train: every gain is scored before this batch moves anything,
        # and every bet is sized before the gain is known.
        lam = None
        gain = self._gain(self.checkpoint, self.challenger, usable)
        if gain is not None:
            ratio = gain[0] / gain[1]
            hist = self._history
            if self.mode == SHADOW:
                lam = _bet(hist["promote"], cap)
                self.e_promote *= 1.0 + lam * ratio
                self.e_reject *= 1.0 - _bet(hist["reject"], LAMBDA_SAFETY) * ratio
                hist["promote"].append(ratio)
                hist["reject"].append(-ratio)
            else:
                self.e_rollback *= 1.0 - _bet(hist["rollback"], LAMBDA_SAFETY) * ratio
                hist["rollback"].append(-ratio)
        evidence = {
            "e_promote": round(self.e_promote, 4),
            "e_reject": round(self.e_reject, 4),
            "e_rollback": round(self.e_rollback, 4),
        }

        if not usable:
            action, reason = WAIT, "no_usable_evidence"
        elif self.mode == ACTIVE and self.e_rollback >= self.threshold:
            action, reason = ROLLBACK, "adaptation_worse_than_checkpoint"
            self.mode = SHADOW
            self.challenger = copy.deepcopy(self.checkpoint)
            self._reset_tests()
        elif self.mode == ACTIVE:
            action, reason = ADAPT, "approved_and_monitored"
        elif self.e_reject >= self.threshold:
            action, reason = REJECT, "challenger_harmful"
            self.challenger = copy.deepcopy(self.checkpoint)
            self._reset_tests()
        elif self.e_promote >= self.threshold:
            action, reason = ADAPT, "evidence_threshold_crossed"
            self.mode = ACTIVE
            self.version += 1
            self._reset_tests()
        else:
            action, reason = WAIT, "insufficient_evidence"

        # Rejected / rolled-back batches are quarantined, not learned from.
        trained = 0
        if action not in (REJECT, ROLLBACK):
            trained = self._train(self.challenger, usable)

        self.counts[action] += 1
        decision = {
            "action": action,
            "reason": reason,
            "mode": self.mode,
            "version": self.version,
            "batch_size": len(outcomes),
            "usable": len(usable),
            "degrading": degrading,
            "trained": trained,
            "drift_score": None if drift is None else round(drift, 4),
            "drift_detected": drift_flag,
            "bet_cap": cap,
            "bet": None if lam is None else round(lam, 4),
            "gain_days": None if gain is None else round(gain[0], 4),
            "gain_bound_days": None if gain is None else round(gain[1], 4),
            "evidence_before_decision": evidence,
            "threshold": self.threshold,
        }
        self.log.append(decision)
        return decision

    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, object]:
        return {
            "alpha": self.alpha,
            "threshold": self.threshold,
            "mode": self.mode,
            "version": self.version,
            "deployed": self.deployed,
            "e_promote": round(self.e_promote, 4),
            "e_reject": round(self.e_reject, 4),
            "e_rollback": round(self.e_rollback, 4),
            "counts": dict(self.counts),
            "degradation_aware": self.degradation_aware,
            "drift_aware": self.drift_aware,
            "drift_reference": {
                k: {"mean": round(m, 4), "std": round(s, 4)}
                for k, (m, s) in self.reference.items()
            },
            "checkpoint": self.checkpoint.snapshot(),
            "challenger": self.challenger.snapshot(),
            "last_decision": self.log[-1] if self.log else None,
        }

    def snapshot_brief(self) -> Dict[str, object]:
        """Compact status for the per-drive twin state."""
        last = self.log[-1] if self.log else None
        return {
            "mode": self.mode,
            "version": self.version,
            "deployed": self.deployed,
            "alpha": self.alpha,
            "e_promote": round(self.e_promote, 4),
            "e_reject": round(self.e_reject, 4),
            "e_rollback": round(self.e_rollback, 4),
            "threshold": self.threshold,
            "counts": dict(self.counts),
            "last_action": None if last is None else last["action"],
            "last_reason": None if last is None else last["reason"],
        }

    def decisions(self) -> List[dict]:
        return list(self.log)
