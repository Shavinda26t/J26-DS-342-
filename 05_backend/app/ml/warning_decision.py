"""Final warning-decision layer: global-warning preservation.

The production warning source is the **global** degradation detector.  A
*warning* is a completed persistence episode: at least ``PERSISTENCE`` (7)
consecutive observations whose combined severity is at or above
``MIN_SEVERITY`` (1).  In the streaming twin that is exactly
``DigitalTwinStateManager.adaptive_state >= 2`` ("Elevated Degradation"),
which ``app.services.twin_sync_service`` reports as ``adaptive_state`` /
``state_level``.

The per-drive adaptive thresholds (Feature 3, ``adaptive_thresholds.py``) are a
*stricter* detector: their severity never exceeds the global severity, so an
observation the adaptive layer qualifies is always a globally qualifying
observation.  The adaptive persistence streak - and therefore any adaptive
warning - is a **subset** of the global one.  Promoting that layer to the final
decision *without* care would silently suppress global warnings.

This module makes the relationship explicit and enforces the hard safety
requirement at the **decision layer** rather than with a severity-score
comparison::

    final_streak = max(global_streak, adaptive_streak)   # preservation on

A global warning can therefore never be cancelled *or delayed* by adaptation.
With preservation on, the adaptive layer cannot change any warning; the
measured cost of that guarantee is reported in the Task 3 follow-up (hard
preservation restores the global false-positive rate, i.e. it removes the
adaptive layer's false-alarm benefit).

``preserve_global_warnings`` defaults to the module flag
``PRESERVE_GLOBAL_WARNINGS`` (True - the production contract).  Setting it to
``False`` yields the "adaptive-only" decision used as the *before* baseline in
that report.  Nothing here reads test labels or tunes anything.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

from app.ml._component2_paths import get_detector

#: Production contract: every global-detector warning survives the final
#: decision.  See the module docstring for the trade-off this implies.
PRESERVE_GLOBAL_WARNINGS = True

#: Escalation level at or above which a persistence episode is a *warning*
#: (as opposed to the pre-warning "Early Warning" state).  Mirrors
#: ``degradation_detector.escalation_level``: 2 == "Elevated Degradation".
WARNING_STATE = 2


def _min_severity() -> int:
    return int(get_detector().MIN_SEVERITY)


def persistence() -> int:
    """The global persistence requirement (tuned production constant)."""
    return int(get_detector().PERSISTENCE)


def escalation_level(streak: int, persist: Optional[int] = None) -> int:
    """Map a persistence streak to an adaptive state level (0-3).

    Delegates to Component 2's ``degradation_detector.escalation_level`` so the
    decision rule cannot drift from the detector's.
    """
    if persist is None:
        persist = persistence()
    return int(get_detector().escalation_level(int(streak), int(persist)))


def is_warning(state: int) -> bool:
    """True when an adaptive state level counts as a warning."""
    return int(state) >= WARNING_STATE


@dataclass(frozen=True)
class WarningDecision:
    """Result of combining the global and adaptive decision paths."""

    global_streak: int
    adaptive_streak: int
    global_state: int
    adaptive_state: int
    state: int
    preserve_global_warnings: bool

    @property
    def preserved(self) -> bool:
        """True when the final decision keeps the global warning level."""
        return self.state >= self.global_state

    @property
    def suppressed(self) -> bool:
        """True when the final decision would lose global warning evidence."""
        return self.state < self.global_state

    @property
    def source(self) -> str:
        """Which path produced the *final* warning level (``none`` if quiet)."""
        if not is_warning(self.state):
            return "none"
        has_adaptive = is_warning(self.adaptive_state)
        has_global = is_warning(self.global_state)
        if has_global and has_adaptive:
            return "global+adaptive"
        if has_global:
            return "global"
        return "adaptive"


def combine_warnings(
    global_streak: int,
    adaptive_streak: int,
    persist: Optional[int] = None,
    preserve_global_warnings: Optional[bool] = None,
) -> WarningDecision:
    """Combine both decision paths at the final warning-decision layer.

    ``global_streak`` comes from ``DigitalTwinStateManager`` (the source of
    truth), ``adaptive_streak`` from :class:`AdaptiveWarningTracker`.  With
    preservation on the final streak is their maximum, so the global decision
    is a floor: no suppression, no delay.  With preservation off the adaptive
    path alone decides (the historically-evaluated "adaptive detector").
    """
    if persist is None:
        persist = persistence()
    if preserve_global_warnings is None:
        preserve_global_warnings = PRESERVE_GLOBAL_WARNINGS

    global_streak = max(0, int(global_streak))
    adaptive_streak = max(0, int(adaptive_streak))
    final_streak = (
        max(global_streak, adaptive_streak)
        if preserve_global_warnings
        else adaptive_streak
    )

    return WarningDecision(
        global_streak=global_streak,
        adaptive_streak=adaptive_streak,
        global_state=escalation_level(global_streak, persist),
        adaptive_state=escalation_level(adaptive_streak, persist),
        state=escalation_level(final_streak, persist),
        preserve_global_warnings=bool(preserve_global_warnings),
    )


@dataclass
class AdaptiveWarningTracker:
    """Streaming persistence tracker over *adaptive* severities.

    Mirrors the streaming detector's ``_calculate_streak``: the streak is the
    number of trailing consecutive observations whose adaptive severity is at
    or above ``MIN_SEVERITY``.  The window matches the state manager's history
    size so both paths see the same effective horizon.

    ``observe`` is idempotent for a repeated ``observation_key``: a read-only
    state request, or a repeated sync of the same observation, must not extend
    the streak ("repeated observations do not unexpectedly change the state").
    The key is normally the observation's timestamp plus history length.
    """

    drive: str = ""
    window: int = 30
    persist: int = field(default_factory=persistence)
    min_severity: int = field(default_factory=_min_severity)
    _severities: Deque[int] = field(default_factory=deque)
    _last_key: Optional[object] = None

    def __post_init__(self) -> None:
        self._severities = deque(
            self._severities, maxlen=max(1, int(self.window))
        )

    def observe(self, severity, observation_key: Optional[object] = None) -> bool:
        """Record one observation's adaptive severity.

        Returns True when a new observation was recorded, False when it was an
        idempotent repeat.  Invalid severities are coerced to 0 (no evidence).
        """
        if observation_key is not None and observation_key == self._last_key:
            return False
        if observation_key is not None:
            self._last_key = observation_key
        try:
            value = 0 if severity is None else int(severity)
        except (TypeError, ValueError):
            value = 0
        self._severities.append(value)
        return True

    def reset(self) -> None:
        self._severities.clear()
        self._last_key = None

    def streak(self) -> int:
        """Trailing consecutive count of qualifying adaptive observations."""
        count = 0
        for severity in reversed(self._severities):
            if severity >= self.min_severity:
                count += 1
            else:
                break
        return count

    def state(self) -> int:
        """Adaptive warning state level (0-3) for the current streak."""
        return escalation_level(self.streak(), self.persist)

    def snapshot(self) -> dict:
        return {
            "drive": self.drive,
            "window": int(self.window),
            "observed": len(self._severities),
            "streak": self.streak(),
            "state": self.state(),
        }

    def to_state(self) -> dict:
        """Serialisable state (persistence payload)."""
        return {
            "drive": self.drive,
            "window": int(self.window),
            "severities": [int(s) for s in self._severities],
            "last_key": self._last_key,
        }

    def load_state(self, state) -> None:
        """Restore a payload produced by :meth:`to_state` (defensive)."""
        if not isinstance(state, dict):
            return
        values = []
        for v in state.get("severities") or []:
            try:
                values.append(int(v))
            except (TypeError, ValueError):
                continue
        self._severities = deque(values, maxlen=max(1, int(self.window)))
        self._last_key = state.get("last_key")


@dataclass(frozen=True)
class WarningEpisode:
    """One maximal run of warning-state observations on one decision path."""

    drive: str
    path: str
    start_index: int
    end_index: int
    start: object
    end: object
    peak_state: int

    @property
    def length(self) -> int:
        return int(self.end_index) - int(self.start_index) + 1


@dataclass
class WarningEpisodeTracker:
    """Collapse a chronological stream of states into distinct episodes.

    An episode is one maximal run of observations with ``state >=
    warning_state``; a sustained warning is therefore a single episode
    (deduplication).  Repeated reads are ignored through the monotone
    ``index`` guard, so re-reading a stream never fabricates episodes.
    """

    drive: str = ""
    path: str = "global"
    warning_state: int = WARNING_STATE
    _episodes: List[WarningEpisode] = field(default_factory=list)
    _start: Optional[int] = None
    _start_ts: object = None
    _end: Optional[int] = None
    _end_ts: object = None
    _peak: int = 0
    _last_index: int = -1

    def observe(self, index: int, timestamp, state: int) -> None:
        index = int(index)
        if index <= self._last_index:
            return
        self._last_index = index
        state = int(state)
        if state >= self.warning_state:
            if self._start is None:
                self._start = index
                self._start_ts = timestamp
            self._peak = max(self._peak, state)
            self._end = index
            self._end_ts = timestamp
        else:
            self._close()

    def _close(self) -> None:
        if self._start is None:
            return
        self._episodes.append(
            WarningEpisode(
                drive=self.drive,
                path=self.path,
                start_index=int(self._start),
                end_index=int(self._end if self._end is not None else self._start),
                start=self._start_ts,
                end=self._end_ts,
                peak_state=int(self._peak),
            )
        )
        self._start = None
        self._end = None
        self._start_ts = None
        self._end_ts = None
        self._peak = 0

    def close(self) -> None:
        """Flush a warning that is still open at the end of the stream."""
        self._close()

    @property
    def episodes(self) -> List[WarningEpisode]:
        return list(self._episodes)
