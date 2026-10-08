import numpy as np
import pandas as pd


class DigitalTwinStateManager:
    """
    Maintains the evolving state of one HDD Digital Twin.
    """

    SMART_FEATURES = [
        "smart_1_normalized",
        "smart_5_normalized",
        "smart_187_normalized",
        "smart_197_normalized",
        "smart_198_normalized",
    ]

    def __init__(self, serial_number, history_size=30):

        self.serial_number = serial_number
        self.history_size = history_size

        self.history = []

        # Adaptive state
        self.moderate_plus_streak = 0
        self.adaptive_state = 0

        self.last_update = None

    def add_observation(self, observation):
        """
        Add one new SMART observation.
        """

        self.history.append(observation.copy())

        if len(self.history) > self.history_size:
            self.history = self.history[-self.history_size:]

        self.last_update = observation.get("date")

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
    def calculate_smart1_severity(change):
        """
        Experimental candidate severity levels for SMART 1.

        0 = no degradation
        1 = mild
        2 = moderate
        3 = strong
        4 = severe
        """

        if pd.isna(change):
            return np.nan

        if change >= 0:
            return 0

        if change >= -2:
            return 1

        if change >= -7:
            return 2

        if change >= -14:
            return 3

        return 4

    @staticmethod
    def calculate_smart187_severity(change):
        """
        Experimental candidate severity levels for SMART 187.

        0 = no degradation
        1 = mild
        2 = moderate
        3 = strong
        4 = severe
        """

        if pd.isna(change):
            return np.nan

        if change >= 0:
            return 0

        if change >= -2:
            return 1

        if change >= -8:
            return 2

        if change >= -22:
            return 3

        return 4

    def calculate_degradation_state(self):

        temporal = self.calculate_temporal_features()

        if temporal is None or temporal.empty:
            return None

        latest = temporal.iloc[-1]

        smart1_severity = self.calculate_smart1_severity(
            latest["smart_1_normalized_change_7d"]
        )

        smart187_severity = self.calculate_smart187_severity(
            latest["smart_187_normalized_change_7d"]
        )

        severities = [
            value
            for value in [
                smart1_severity,
                smart187_severity
            ]
            if not pd.isna(value)
        ]

        if not severities:
            combined_severity = np.nan
        else:
            combined_severity = max(severities)

        return {
            "smart_1_severity": smart1_severity,
            "smart_187_severity": smart187_severity,
            "combined_severity": combined_severity
        }

    def update_persistence(self):
        """
        Update the persistence state using the latest degradation severity.

        Moderate or stronger degradation is defined as severity >= 2.

        Adaptive states:
            0 = Normal / Monitoring
            1 = Early Warning
            2 = Elevated Degradation
            3 = Strong Persistent Degradation

        Missing degradation evidence does NOT mean healthy.
        When no degradation evidence is available, the previous
        persistence state is retained.
        """

        degradation = self.calculate_degradation_state()

        if degradation is None:
            return {
                "moderate_plus_streak": self.moderate_plus_streak,
                "adaptive_state": self.adaptive_state,
                "evidence_available": False
            }

        combined_severity = degradation["combined_severity"]

        # ----------------------------------------------------
        # No usable degradation evidence
        # ----------------------------------------------------

        if pd.isna(combined_severity):

            return {
                "moderate_plus_streak": self.moderate_plus_streak,
                "adaptive_state": self.adaptive_state,
                "evidence_available": False
            }

        # ----------------------------------------------------
        # Moderate or stronger degradation
        # ----------------------------------------------------

        if combined_severity >= 2:

            self.moderate_plus_streak += 1

        else:

            # A non-moderate observation breaks the
            # consecutive degradation sequence.
            self.moderate_plus_streak = 0

        # ----------------------------------------------------
        # Adaptive state
        # ----------------------------------------------------

        if self.moderate_plus_streak >= 7:

            self.adaptive_state = 3

        elif self.moderate_plus_streak >= 5:

            self.adaptive_state = 2

        elif self.moderate_plus_streak >= 3:

            self.adaptive_state = 1

        else:

            self.adaptive_state = 0

        return {
            "moderate_plus_streak": self.moderate_plus_streak,
            "adaptive_state": self.adaptive_state,
            "evidence_available": True
        }

    def get_adaptive_state_name(self):
        """
        Convert the numerical adaptive state into a readable label.
        """

        state_names = {
            0: "Normal / Monitoring",
            1: "Early Warning",
            2: "Elevated Degradation",
            3: "Strong Persistent Degradation"
        }

        return state_names.get(
            self.adaptive_state,
            "Unknown"
        )

    def get_current_state(self):
        """
        Return the current Digital Twin state.
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