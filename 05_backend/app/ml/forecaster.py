"""Future health forecasting for the digital twin.

Component 2's model emits a single RUL scalar — a *point* in time. Operations
need a **trajectory**: how health is expected to evolve, when the twin is
likely to escalate, and how uncertainty grows with distance into the future.

This module produces a transparent, multi-horizon forecast from the current
twin state:

* **RUL trajectory** — remaining useful life declines ~1 day per day under
  nominal operation, giving a projected RUL and a failure ETA.
* **State escalation schedule** — if degradation evidence persists at the
  current pace, the persistence streak keeps growing and the adaptive state
  climbs the Component 2 ladder (Normal -> Early Warning -> Elevated ->
  Strong). We project the streak forward and map it through the *same*
  ``escalation_level`` rules the detector uses, so the forecast is consistent
  with the published state machine.
* **Widening uncertainty** — the conformal interval half-width is inflated
  with horizon so far-future claims are honestly less certain.

The forecast is deliberately simple and explainable: it is a *nominal
degradation projection*, not a second black-box model, which makes it auditable
and safe to show to operators.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

# Growth applied to the base uncertainty half-width per day of horizon.
HORIZON_UNCERTAINTY_GROWTH = 0.08

# Max horizon we will ever project (days).
MAX_HORIZON = 90

DEFAULT_HORIZONS = (1, 7, 14, 30)


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


def forecast(
    predicted_rul: Optional[float],
    current_state: int = 0,
    current_streak: int = 0,
    persistence: int = 7,
    base_half_width: float = 0.0,
    horizons: Optional[List[int]] = None,
) -> Dict[str, object]:
    """
    Project twin health across future horizons.

    Parameters mirror the live twin state so the forecast is a pure function of
    what is known now (no hidden model).
    """
    horizons = sorted({h for h in (horizons or list(DEFAULT_HORIZONS)) if 0 < h <= MAX_HORIZON})
    if not horizons:
        horizons = list(DEFAULT_HORIZONS)

    rul = None if predicted_rul is None or math.isnan(predicted_rul) else float(predicted_rul)
    rul = max(0.0, rul) if rul is not None else None

    failure_eta = round(rul) if rul is not None else None

    trajectory: List[Dict[str, object]] = []
    first_escalation: Optional[Dict[str, object]] = None

    for day in horizons:
        # --- RUL path: nominal 1 day of life consumed per day -------------
        if rul is None:
            proj_rul = None
            low = high = None
        else:
            proj_rul = max(0.0, rul - day)
            spread = base_half_width * (1.0 + HORIZON_UNCERTAINTY_GROWTH * day)
            low = max(0.0, proj_rul - spread)
            high = proj_rul + spread

        # --- State path: streak grows ~1/observation-day if degradation
        #     continues at the current pace, then mapped via the ladder. ----
        projected_streak = int(current_streak + day)
        state_level = escalation_level(projected_streak, persistence)

        escalation = state_level > current_state
        entry = {
            "day": day,
            "projected_rul_days": round(proj_rul, 2) if proj_rul is not None else None,
            "rul_low": round(low, 2) if low is not None else None,
            "rul_high": round(high, 2) if high is not None else None,
            "projected_streak": projected_streak,
            "projected_state": state_level,
            "projected_state_name": STATE_NAMES.get(state_level, "Unknown"),
            "escalates_from_current": escalation,
        }
        trajectory.append(entry)

        if escalation and first_escalation is None:
            first_escalation = {
                "day": day,
                "to_state": state_level,
                "to_state_name": STATE_NAMES.get(state_level, "Unknown"),
            }

    return {
        "horizons_days": horizons,
        "current_state": current_state,
        "current_state_name": STATE_NAMES.get(current_state, "Unknown"),
        "current_streak": current_streak,
        "persistence_used": persistence,
        "predicted_rul_now": round(rul, 2) if rul is not None else None,
        "failure_eta_day": failure_eta,
        "first_escalation": first_escalation,
        "trajectory": trajectory,
        "assumptions": [
            "RUL declines 1 day per day under nominal operation",
            "degradation streak grows ~1 observation per day if evidence persists",
            "state escalation uses Component 2's production ladder rules",
            "uncertainty widens with horizon from the conformal interval",
        ],
    }
