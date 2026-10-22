"""Future health forecasting for the digital twin.

Component 2's model emits a single RUL scalar - a *point* in time.  Operations
need a **trajectory**: how health is expected to evolve, when the twin is
*likely* to escalate, and how uncertain the future is.

The previous implementation mixed four different kinds of statement into one
unqualified output, so a conditional scenario looked like a prediction:

1. ``projected_streak = current_streak + day`` asserted that degradation
   evidence continues for every future day.  A drive with **zero** current
   evidence was therefore "forecast" to reach "Elevated Degradation" at day 7
   merely because time passed (100% of the test drives did).
2. ``escalates_from_current`` presented that assumption as a flat fact.
3. ``current_streak`` counts **observations**, not calendar days, and the live
   API accepts observations at any cadence, so "+7 days" was not seven days.
4. The RUL band was one conformal interval translated by ``-h`` and widened by
   a cosmetic factor, yet advertised per-horizon coverage.

This module now separates *what is observed* from *what is assumed* and never
invents escalation:

* **Current state** - ``current_state`` / ``current_streak`` from the detector
  (``streak`` is explicitly in observations).
* **RUL path** - a deterministic countdown from the frozen BiLSTM estimate
  (``failure_eta_day``).  This is **definitional**: it is the label convention
  (RUL = failure date - sequence end date), *not* evidence that the model
  detected worsening SMART behaviour on its own.
* **Escalation scenarios** - each horizon reports escalation under three
  explicit, labelled assumptions plus an availability flag:

  - ``neutral`` - no new degradation evidence; the streak cannot grow.
  - ``persistence`` - the continuation model: evidence persists, so the streak
    grows by *at most* one qualifying **observation** per observation *and*
    only if at least one observation is expected in the horizon.  Expected
    observations come from the measured cadence; with no cadence evidence the
    continuation forecast is **unsupported** rather than assumed.
  - ``lower_bound`` - strictly adversarial: every future observation breaks
    the run (streak -> 0).  This bounds the range from below and pairs with
    ``persistence`` to give an honest best case.

* **Uncertainty** - the interval is the frozen conformal band translated by
  ``-h`` and widened with horizon.  It is *derived*, not independently
  calibrated per horizon; ``interval_calibration`` says so explicitly.

The output is a pure function of the current state (no hidden model), so
repeated calls on unchanged state are deterministic.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

# Growth applied to the base uncertainty half-width per day of horizon.
# The band is *derived* from the base conformal interval, NOT independently
# calibrated per horizon (see the module docstring and ``interval_calibration``).
HORIZON_UNCERTAINTY_GROWTH = 0.08

# Max horizon we will ever project (days).
MAX_HORIZON = 90

DEFAULT_HORIZONS = (1, 7, 14, 30)

# Observations counted per calendar day when the measured cadence says the
# stream is sampled daily.  Above this a stream is "irregular" and the day
# projection is reported as unsupported.
DAILY_CADENCE_TOLERANCE = 1.5

# Cadence is only trusted with at least this many observations.
MIN_CADENCE_OBSERVATIONS = 3


def escalation_level(streak: int, persistence: int) -> int:
    """Component 2's state-ladder mapping (mirrors the production detector)."""
    if streak >= persistence + 2:
        return 3
    if streak >= persistence:
        return 2
    if streak >= max(1, persistence - 2):
        return 1
    return 0


STATE_NAMES = {
    0: "Normal / Monitor",
    1: "Early Warning",
    2: "Elevated Degradation",
    3: "Strong Persistent Degradation",
}

# Cadence status values (``cadence_status``).
CADENCE_DAILY = "daily"
CADENCE_IRREGULAR = "irregular"
CADENCE_INSUFFICIENT = "insufficient"

# Evidence status values describing whether the *continuation* scenario is
# defensible (``projected_evidence_status``).
EVIDENCE_SUPPORTED = "supported"
EVIDENCE_UNSUPPORTED = "unsupported"
EVIDENCE_NO_TREND = "no_current_trend"


def _estimate_cadence(
    observation_gaps_days: Optional[List[float]],
    observation_count: Optional[int],
) -> Dict[str, object]:
    """Summarise the observed sampling cadence (days between observations).

    ``observation_gaps_days`` are *consecutive* calendar-day gaps from the
    drive's own history.  The result never invents a cadence: with too little
    history it reports ``insufficient`` and the continuation forecast becomes
    unsupported.
    """
    gaps = [
        float(g)
        for g in (observation_gaps_days or [])
        if g is not None and math.isfinite(float(g)) and float(g) > 0.0
    ]
    count = int(observation_count) if observation_count is not None else 0

    if len(gaps) < MIN_CADENCE_OBSERVATIONS - 1:
        return {
            "status": CADENCE_INSUFFICIENT,
            "median_gap_days": None,
            "observations_per_day": None,
            "n_gaps": len(gaps),
            "observations_counted": count,
            "reason": (
                "fewer than "
                f"{MIN_CADENCE_OBSERVATIONS - 1} consecutive observation gaps"
            ),
        }

    ordered = sorted(gaps)
    mid = len(ordered) // 2
    median = (
        ordered[mid]
        if len(ordered) % 2
        else (ordered[mid - 1] + ordered[mid]) / 2.0
    )
    per_day = 1.0 / median if median > 0 else None
    status = CADENCE_DAILY if median <= DAILY_CADENCE_TOLERANCE else CADENCE_IRREGULAR
    return {
        "status": status,
        "median_gap_days": round(median, 4),
        "observations_per_day": round(per_day, 4) if per_day is not None else None,
        "n_gaps": len(gaps),
        "observations_counted": count,
        "reason": None,
    }


def _observations_in_horizon(cadence: Dict[str, object], horizon: int) -> Optional[int]:
    """Expected observations in ``horizon`` **calendar days**.

    The conversion is cadence-aware: it uses the *measured* median gap, so a
    weekly stream maps 7 days to 1 observation instead of 7.  It never assumes
    daily sampling.  ``None`` only when there is no cadence evidence at all
    (the horizon cannot be converted at all).
    """
    median = cadence.get("median_gap_days")
    if median is None:
        return None
    median = float(median)
    if median <= 0:
        return None
    return int(math.floor(int(horizon) / median))


def _scenarios(
    current_streak: int,
    persistence: int,
    horizon: int,
    cadence: Dict[str, object],
) -> Dict[str, object]:
    """Escalation under explicit, labelled assumptions for one horizon."""
    current_level = escalation_level(current_streak, persistence)

    # Neutral: no new evidence.  Future observations are non-qualifying (or
    # absent), so the streak cannot grow.
    neutral_level = escalation_level(current_streak, persistence)

    # Lower bound: strictly adversarial - every future observation breaks the
    # run.  Mild <=1 when history is complete.
    lower_level = escalation_level(0, persistence)

    # Continuation: evidence persists, bounded by how many observations can
    # actually occur within the horizon.
    obs_in_horizon = _observations_in_horizon(cadence, horizon)
    if obs_in_horizon is None:
        evidence_status = EVIDENCE_UNSUPPORTED
        continuation_streak = None
        continuation_level = None
        continuation_reason = (
            "observation cadence is unknown "
            f"({cadence.get('status')}); a day horizon cannot be converted to "
            "an observation count"
        )
    elif obs_in_horizon <= 0:
        evidence_status = EVIDENCE_SUPPORTED
        continuation_streak = current_streak
        continuation_level = neutral_level
        continuation_reason = (
            "fewer than one observation is expected within this horizon"
        )
    elif current_streak <= 0:
        # No *current* evidence: continuation of a trend that does not exist is
        # not a forecast.  Report it as unsupported rather than fabricating
        # escalation.
        evidence_status = EVIDENCE_NO_TREND
        continuation_streak = current_streak
        continuation_level = neutral_level
        continuation_reason = (
            "no current degradation evidence to continue "
            f"(streak={current_streak})"
        )
    else:
        evidence_status = EVIDENCE_SUPPORTED
        continuation_streak = current_streak + obs_in_horizon
        continuation_level = escalation_level(continuation_streak, persistence)
        continuation_reason = None

    neutral_escalates = neutral_level > current_level
    continuation_escalates = (
        continuation_level is not None and continuation_level > current_level
    )
    lower_escalates = lower_level > current_level

    # Range of *possible* escalation under the two extremal assumptions.
    esc_floor = lower_level
    esc_ceiling = (
        continuation_level if continuation_level is not None else current_level
    )
    possible_levels = [
        level
        for level in (lower_level, current_level, neutral_level, esc_ceiling)
        if level is not None
    ]

    all_branch_levels = (
        [neutral_level] if continuation_level is None else [neutral_level, continuation_level]
    )
    # "Unexplained" = at least one scenario escalates beyond the current state,
    # i.e. escalation depends on an assumption rather than on observed evidence.
    unexplained = any(level > current_level for level in all_branch_levels) or (
        lower_level > current_level
    )

    return {
        "current_level": current_level,
        "neutral_level": neutral_level,
        "neutral_streak": current_streak,
        "lower_level": lower_level,
        "lower_streak": 0,
        "continuation_level": continuation_level,
        "continuation_streak": continuation_streak,
        "obs_in_horizon": obs_in_horizon,
        "evidence_status": evidence_status,
        "evidence_reason": continuation_reason,
        "neutral_escalates": neutral_escalates,
        "continuation_escalates": continuation_escalates,
        "lower_escalates": lower_escalates,
        "escalation_floor": esc_floor,
        "escalation_ceiling": esc_ceiling,
        "range": sorted(set(possible_levels)),
        "unexplained": unexplained,
    }


def forecast(
    predicted_rul: Optional[float],
    current_state: int = 0,
    current_streak: int = 0,
    persistence: int = 7,
    base_half_width: float = 0.0,
    horizons: Optional[List[int]] = None,
    observation_gaps_days: Optional[List[float]] = None,
    observation_count: Optional[int] = None,
) -> Dict[str, object]:
    """
    Project twin health across future horizons.

    Parameters mirror the live twin state so the forecast is a pure function of
    what is known now (no hidden model).

    ``observation_gaps_days`` (consecutive calendar-day gaps from the drive's
    history) and ``observation_count`` let the day-based horizons be validated
    against the measured cadence.  When they are missing the continuation
    forecast is reported as unsupported instead of assuming daily sampling.
    """
    horizons = sorted({h for h in (horizons or list(DEFAULT_HORIZONS)) if 0 < h <= MAX_HORIZON})
    if not horizons:
        horizons = list(DEFAULT_HORIZONS)

    rul = None if predicted_rul is None or math.isnan(predicted_rul) else float(predicted_rul)
    rul = max(0.0, rul) if rul is not None else None

    cadence = _estimate_cadence(observation_gaps_days, observation_count)
    cadence_supported = cadence.get("status") == CADENCE_DAILY

    # Deterministic RUL countdown (definitional - see module docstring).
    failure_eta = round(rul) if rul is not None else None

    trajectory: List[Dict[str, object]] = []
    first_escalation: Optional[Dict[str, object]] = None

    for day in horizons:
        # --- RUL path: definitional 1 day of life consumed per day --------
        if rul is None:
            proj_rul = None
            low = high = None
        else:
            proj_rul = max(0.0, rul - day)
            spread = base_half_width * (1.0 + HORIZON_UNCERTAINTY_GROWTH * day)
            low = max(0.0, proj_rul - spread)
            high = proj_rul + spread

        # --- State path: explicit, labelled scenarios ---------------------
        sc = _scenarios(current_streak, persistence, day, cadence)

        entry: Dict[str, object] = {
            "day": day,
            "projected_rul_days": round(proj_rul, 2) if proj_rul is not None else None,
            "rul_low": round(low, 2) if low is not None else None,
            "rul_high": round(high, 2) if high is not None else None,
            # Backwards-compatible "continuation" projection.  May be None when
            # the continuation model is not supported; never invented.
            "projected_streak": sc["continuation_streak"],
            "projected_state": sc["continuation_level"],
            "projected_state_name": (
                STATE_NAMES.get(sc["continuation_level"], "Unknown")
                if sc["continuation_level"] is not None
                else None
            ),
            # A day-based "escalates?" flag is only defensible when the horizon
            # is supportable *and* escalation does not depend on an assumption.
            "escalates_from_current": bool(
                sc["continuation_escalates"] and cadence_supported
            ),
            # --- Explicit scenario separation ----------------------------
            "scenarios": {
                "current": {
                    "state": sc["current_level"],
                    "state_name": STATE_NAMES.get(sc["current_level"], "Unknown"),
                    "streak": int(current_streak),
                },
                "neutral": {
                    "state": sc["neutral_level"],
                    "state_name": STATE_NAMES.get(sc["neutral_level"], "Unknown"),
                    "streak": sc["neutral_streak"],
                    "assumption": "no new degradation evidence",
                },
                "lower_bound": {
                    "state": sc["lower_level"],
                    "state_name": STATE_NAMES.get(sc["lower_level"], "Unknown"),
                    "streak": sc["lower_streak"],
                    "assumption": "every future observation breaks the streak",
                },
                "continuation": {
                    "state": sc["continuation_level"],
                    "state_name": (
                        STATE_NAMES.get(sc["continuation_level"], "Unknown")
                        if sc["continuation_level"] is not None
                        else None
                    ),
                    "streak": sc["continuation_streak"],
                    "available": sc["continuation_level"] is not None,
                    "assumption": "degradation evidence persists at the observed cadence",
                },
            },
            "availability": {
                "continuation_forecast_available": sc["continuation_level"] is not None,
                "cadence_supported": cadence_supported,
                "evidence_status": sc["evidence_status"],
                "reason": sc["evidence_reason"],
            },
            "escalation_range": sc["range"],
            "escalation_floor": sc["escalation_floor"],
            "escalation_ceiling": sc["escalation_ceiling"],
            "unexplained_escalation": bool(sc["unexplained"]),
        }
        trajectory.append(entry)

        # First *supported* continuation escalation (never from a flat
        # assumption, never on an unsupported horizon).
        if (
            first_escalation is None
            and sc["continuation_escalates"]
            and cadence_supported
        ):
            first_escalation = {
                "day": day,
                "to_state": sc["continuation_level"],
                "to_state_name": STATE_NAMES.get(sc["continuation_level"], "Unknown"),
                "assumption": "degradation evidence persists at the observed cadence",
            }

    # An "unconditional" escalation claim is one that no longer depends on the
    # continuation assumption: observed evidence alone reaches a warning.
    warning_state = 2
    current_level = escalation_level(int(current_streak), int(persistence))
    unconditional = current_level >= warning_state

    return {
        "horizons_days": horizons,
        "current_state": int(current_state),
        "current_state_name": STATE_NAMES.get(int(current_state), "Unknown"),
        "current_streak": int(current_streak),
        "current_streak_unit": "observations",
        "persistence_used": int(persistence),
        "predicted_rul_now": round(rul, 2) if rul is not None else None,
        "failure_eta_day": failure_eta,
        "first_escalation": first_escalation,
        "escalation_forecast_status": "observed" if unconditional else "conditional",
        "cadence": cadence,
        "evidence_policy": {
            "continuation_model": (
                "streak grows by at most one qualifying observation per "
                "observation, bounded by the expected observations within the "
                "horizon; requires at least one current qualifying observation"
            ),
            "day_horizon_requires": "daily cadence evidence",
            "neutral_model": "no new degradation evidence (streak held constant)",
            "lower_bound_model": "every future observation breaks the streak",
        },
        "interval_calibration": {
            "type": "derived_from_base",
            "horizon_specific": False,
            "base_alpha": 0.1,
            "note": (
                "each horizon band is the base conformal interval translated by "
                "-h and widened by (1 + 0.08*h); it is NOT independently "
                "calibrated per horizon"
            ),
        },
        "trajectory": trajectory,
        "assumptions": [
            "RUL counts down 1 day per day by label definition "
            "(failure date - sequence end date); this is definitional, not "
            "independently detected degradation",
            "streak is measured in observations, not calendar days",
            "escalation is only forecast under the labelled scenarios; with no "
            "current evidence the continuation forecast is unsupported",
            "uncertainty widens with horizon but is derived from the base "
            "conformal interval, not horizon-calibrated",
        ],
    }
