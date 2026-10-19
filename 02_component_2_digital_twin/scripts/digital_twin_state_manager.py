import numpy as np
import pandas as pd

import degradation_detector as detector


class DigitalTwinStateManager:
    """
    Maintains the evolving state of one HDD Digital Twin.

    The detector configuration (severity thresholds, MIN_SEVERITY,
    PERSISTENCE and the adaptive-state ladder) comes from
    scripts/degradation_detector.py, the single source of truth shared
    with the batch detector in adaptive_digital_twin.py.
    """

    SMART_FEATURES = [
        "smart_1_normalized",
        "smart_5_normalized",
        "smart_187_normalized",
        "smart_197_normalized",
        "smart_198_normalized",
    ]

    # ========================================================
    # Detector configuration (shared with the production
    # batch detector - see degradation_detector.py)
    # ========================================================

    # Severity 1 (mild) or higher counts as degradation evidence.
    MIN_SEVERITY = detector.MIN_SEVERITY

    # Consecutive qualifying observations before escalation.
    PERSISTENCE = detector.PERSISTENCE

    # Per-attribute change thresholds, from strict to mild.
    # Each entry maps a lower bound (change <= bound) to a
    # severity level.
    SMART1_THRESHOLDS = detector.SMART1_THRESHOLDS

    SMART187_THRESHOLDS = detector.SMART187_THRESHOLDS

    def __init__(self, serial_number, history_size=30):

        self.serial_number = serial_number
        self.history_size = history_size

        self.history = []

        # Adaptive state (recomputed from history, see
        # update_persistence).
        self.moderate_plus_streak = 0
        self.adaptive_state = 0

        self.last_update = None

    def _sort_history_by_date(self):
        """
        Keep stored observations ordered by date.

        The temporal features use groupby-style diffs and rolling
        windows that assume chronological order, so out-of-order
        ingestion must not reach calculate_temporal_features().
        Observations without a parseable date keep their arrival
        order at the end.
        """

        if len(self.history) < 2:
            return

        def sort_key(item):

            index, observation = item

            timestamp = pd.to_datetime(
                observation.get("date"),
                errors="coerce",
            )

            if pd.isna(timestamp):
                return (1, 0, index)

            return (0, timestamp.value, index)

        self.history = [
            observation
            for _, observation in sorted(
                enumerate(self.history),
                key=sort_key,
            )
        ]

    def add_observation(self, observation):
        """
        Add one new SMART observation.
        """

        self.history.append(observation.copy())

        # Sort before trimming so the oldest dates are dropped,
        # not the most recently inserted rows.
        self._sort_history_by_date()

        if len(self.history) > self.history_size:
            self.history = self.history[-self.history_size:]

        if self.history:
            self.last_update = self.history[-1].get("date")

    def get_history_dataframe(self):
        """
        Return stored HDD observations as a DataFrame.
        """

        if not self.history:
            return pd.DataFrame()

        return pd.DataFrame(self.history)

    def get_history_length(self):
        """
        Return number of stored observations.
        """

        return len(self.history)

    def calculate_temporal_features(self):
        """
        Calculate temporal features required by the RUL model.

        NOTE: the "_change_7d" / "_change_14d" features are 7- and
        14-OBSERVATION diffs, not calendar-day diffs (the same
        semantics as build_temporal_features.py and the batch
        detector). With irregular sampling, seven observations can
        span more than seven calendar days.
        """

        df = self.get_history_dataframe()

        if df.empty:
            return None

        result = df.copy()

        for feature in self.SMART_FEATURES:

            if feature in result.columns:

                result[feature] = pd.to_numeric(
                    result[feature],
                    errors="coerce"
                )

        # SMART 1

        result[
            "smart_1_normalized_change_7d"
        ] = result[
            "smart_1_normalized"
        ].diff(7)

        result[
            "smart_1_normalized_change_14d"
        ] = result[
            "smart_1_normalized"
        ].diff(14)

        result[
            "smart_1_normalized_rolling_std_7"
        ] = result[
            "smart_1_normalized"
        ].rolling(
            window=7,
            min_periods=2
        ).std()

        # SMART 187

        result[
            "smart_187_normalized_change_7d"
        ] = result[
            "smart_187_normalized"
        ].diff(7)

        result[
            "smart_187_normalized_change_14d"
        ] = result[
            "smart_187_normalized"
        ].diff(14)

        result[
            "smart_187_normalized_rolling_std_7"
        ] = result[
            "smart_187_normalized"
        ].rolling(
            window=7,
            min_periods=2
        ).std()

        return result

    @staticmethod
    def severity_from_thresholds(change, thresholds):
        """
        Return the production severity level (0-4) for an
        observation-diff change value, given an ordered list of
        (bound, level) pairs.

        A missing change returns np.nan so that "no evidence"
        stays distinguishable from "no degradation" (severity 0).
        """

        return detector.severity_from_thresholds(
            change,
            thresholds,
            missing=np.nan
        )

    @classmethod
    def calculate_smart1_severity(cls, change):
        """
        Production severity levels for SMART 1.

        0 = no degradation
        1 = mild
        2 = moderate
        3 = strong
        4 = severe
        """

        return cls.severity_from_thresholds(
            change,
            cls.SMART1_THRESHOLDS
        )

    @classmethod
    def calculate_smart187_severity(cls, change):
        """
        Production severity levels for SMART 187.

        0 = no degradation
        1 = mild
        2 = moderate
        3 = strong
        4 = severe
        """

        return cls.severity_from_thresholds(
            change,
            cls.SMART187_THRESHOLDS
        )

    @classmethod
    def _row_severity(cls, row):
        """
        Combined severity for one temporal row.

        Returns np.nan when neither degradation signal has
        evidence on that row.
        """

        return detector.combined_severity(
            row.get("smart_1_normalized_change_7d"),
            row.get("smart_187_normalized_change_7d"),
            missing=np.nan
        )

    def calculate_degradation_state(self):
        """
        Degradation severities for the most recent observation.
        """

        temporal = self.calculate_temporal_features()

        if temporal is None or temporal.empty:
            return None

        latest = temporal.iloc[-1]

        smart1_severity = self.calculate_smart1_severity(
            latest.get("smart_1_normalized_change_7d")
        )

        smart187_severity = self.calculate_smart187_severity(
            latest.get("smart_187_normalized_change_7d")
        )

        combined_severity = self._row_severity(latest)

        return {
            "smart_1_severity": smart1_severity,
            "smart_187_severity": smart187_severity,
            "combined_severity": combined_severity
        }

    def _calculate_streak(self):
        """
        Recompute the consecutive qualifying-observation streak
        from the stored history.

        Returns (streak, evidence_available). The result is a pure
        function of the stored observations, so reading the state
        any number of times never double-counts.
        """

        temporal = self.calculate_temporal_features()

        if temporal is None or temporal.empty:
            return 0, False

        streak = 0

        for _, row in temporal.iterrows():

            severity = self._row_severity(row)

            if detector.is_qualifying(severity, self.MIN_SEVERITY):

                streak += 1

            else:

                # A non-degrading observation breaks the run.
                # A row with no evidence breaks it as well (the
                # batch detector does the same), but callers can
                # tell the two apart through evidence_available,
                # so missing telemetry is never reported as an
                # observed healthy value.
                streak = 0

        latest_severity = self._row_severity(temporal.iloc[-1])

        return streak, not pd.isna(latest_severity)

    def update_persistence(self):
        """
        Recompute the persistence state from the stored history.

        Any degradation evidence at or above MIN_SEVERITY
        (mild or worse) extends the consecutive streak; any other
        observation ends it.

        Adaptive states (escalation mirrors the production
        detector, where p = PERSISTENCE):
            0 = Normal / Monitor              (streak < max(1, p-2))
            1 = Early Warning                 (streak >= max(1, p-2))
            2 = Elevated Degradation          (streak >= p)
            3 = Strong Persistent Degradation (streak >= p+2)

        Missing degradation evidence does NOT mean healthy: a row
        without evidence resets the run (as in the batch detector)
        and is reported via evidence_available = False, so callers
        can distinguish "observed non-degradation" from "no data".

        This method is idempotent: it derives the state from the
        history instead of incrementing counters, so calling it
        repeatedly without adding observations returns the same
        values.
        """

        self.moderate_plus_streak, evidence_available = (
            self._calculate_streak()
        )

        self.adaptive_state = detector.escalation_level(
            self.moderate_plus_streak,
            self.PERSISTENCE
        )

        return {
            "moderate_plus_streak": self.moderate_plus_streak,
            "adaptive_state": self.adaptive_state,
            "evidence_available": evidence_available
        }

    def get_adaptive_state_name(self):
        """
        Convert the numerical adaptive state into a readable label.
        """

        return detector.adaptive_state_name(
            self.adaptive_state
        )

    def get_current_state(self):
        """
        Return the current Digital Twin state.

        Safe to call any number of times: reading the state does
        not mutate the streak.
        """

        degradation = self.calculate_degradation_state()

        persistence = self.update_persistence()

        return {
            "serial_number": self.serial_number,
            "last_update": self.last_update,
            "history_length": self.get_history_length(),

            "smart_1_severity": (
                degradation["smart_1_severity"]
                if degradation is not None
                else np.nan
            ),

            "smart_187_severity": (
                degradation["smart_187_severity"]
                if degradation is not None
                else np.nan
            ),

            "combined_severity": (
                degradation["combined_severity"]
                if degradation is not None
                else np.nan
            ),

            "moderate_plus_streak":
                persistence["moderate_plus_streak"],

            "adaptive_state":
                persistence["adaptive_state"],

            "adaptive_state_name":
                self.get_adaptive_state_name(),

            "degradation_evidence_available":
                persistence["evidence_available"]
        }
