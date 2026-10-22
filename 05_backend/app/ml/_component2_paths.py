"""Bootstrap access to Component 2's research scripts.

Component 2 keeps its logic in ``02_component_2_digital_twin/scripts``.
The backend reuses those modules as the **single source of truth** for the
degradation detector and the streaming state manager rather than duplicating
their rules. Only pure-logic modules (no file IO at import time) are imported.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
COMPONENT2_SCRIPTS = (
    REPO_ROOT / "02_component_2_digital_twin" / "scripts"
)

_COMPONENT2_RESULTS = (
    REPO_ROOT / "02_component_2_digital_twin" / "results"
)


def _ensure_scripts_on_path() -> None:
    scripts = str(COMPONENT2_SCRIPTS)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)


def get_detector():
    """Return Component 2's shared ``degradation_detector`` module."""
    _ensure_scripts_on_path()
    import degradation_detector as detector  # noqa: PLC0415

    return detector


def get_state_manager_class():
    """Return Component 2's ``DigitalTwinStateManager`` class."""
    _ensure_scripts_on_path()
    from digital_twin_state_manager import (  # noqa: PLC0415
        DigitalTwinStateManager,
    )

    return DigitalTwinStateManager


def component2_results_dir() -> Path:
    return _COMPONENT2_RESULTS
