import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import argparse

import degradation_detector as detector

# ============================================================
# CONFIGURATION
# ============================================================

V5_STATE_FILE = r".\results\digital_twin_state_v5.csv"

MODEL_FILE = (
    r".\results\rul_sequences\sequence_length_experiments"
    r"\seq_30\bilstm\best_bilstm_model.pt"
)

PREPROCESSING_FILE = (
    r".\results\rul_sequences\sequence_length_experiments"
    r"\seq_30\preprocessed\preprocessing_parameters.json"
)

OUTPUT_DIR = r".\results\digital_twin_rul"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "adaptive_digital_twin_predictions.csv"
)

SEQUENCE_LENGTH = 30


# ============================================================
# BILSTM MODEL
# ============================================================

class RULBiLSTM(nn.Module):

    def __init__(
        self,
        input_size=22,
        hidden_size=64,
        num_layers=2,
        dropout=0.2
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        last_output = output[:, -1, :]

        return self.fc(last_output).squeeze(1)


# ============================================================
# ADAPTIVE DIGITAL TWIN
# ============================================================

class AdaptiveDigitalTwin:

    # ========================================================
    # Degradation detector configuration
    # ========================================================
    #
    # The values themselves live in scripts/degradation_detector.py,
    # the single source of truth shared with the streaming state
    # manager (digital_twin_state_manager.py) and the validators.
    # Summary:
    #
    #   MIN_SEVERITY = 1   mild-or-higher counts as degradation
    #   PERSISTENCE  = 7   consecutive qualifying observations
    #
    # Selected by scripts/tune_detection_thresholds.py over 40
    # configurations on the TRAIN + VALIDATION HDD split (657 failed
    # HDDs + 400 never-failed controls), then evaluated once on the
    # untouched held-out test set:
    #
    #   results/digital_twin_rul/threshold_sweep_results.csv (tuning)
    #   results/digital_twin_rul/threshold_sweep_final_test.csv (test)
    #
    # Requiring Moderate-or-higher severity was too strict: counting
    # Mild-or-higher signals that persist for 7 consecutive
    # observations roughly doubles detection and doubles the warning
    # lead time, at a modest false-alarm cost.
    #
    # NOTE: "7 consecutive observations" is an observation count, not
    # a calendar-week guarantee - the change features are
    # observation diffs and sampling gaps are common (see
    # results/hdd_observation_gap_analysis.csv).
    # ========================================================

    # None keeps the production thresholds; a factor < 1 would
    # make them less strict (bounds are negative).
    RELAX_THRESHOLDS = None

    MIN_SEVERITY = detector.MIN_SEVERITY

    PERSISTENCE = detector.PERSISTENCE

    # Per-attribute change thresholds, from strict to mild.
    # Each entry maps a lower bound (change <= bound) to a
    # severity level. When RELAX_THRESHOLDS is set, the bounds are
    # scaled by that factor (bounds are negative, so a factor < 1
    # makes them less strict).
    SMART1_THRESHOLDS = detector.SMART1_THRESHOLDS

    SMART187_THRESHOLDS = detector.SMART187_THRESHOLDS

    def __init__(self):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print("Device:", self.device)

        # Scale the severity thresholds once at construction time.
        factor = self.RELAX_THRESHOLDS

        self.smart1_thresholds = detector.scaled_thresholds(
            self.SMART1_THRESHOLDS,
            factor,
        )

        self.smart187_thresholds = detector.scaled_thresholds(
            self.SMART187_THRESHOLDS,
            factor,
        )

        # ----------------------------------------------------
        # Load preprocessing parameters
        # ----------------------------------------------------

        with open(
            PREPROCESSING_FILE,
            "r"
        ) as f:

            self.preprocessing = json.load(f)

        self.base_features = (
            self.preprocessing["base_features"]
        )

        self.temporal_features = (
            self.preprocessing["temporal_features"]
        )

        self.medians = np.array(
            self.preprocessing["medians"],
            dtype=np.float32
        )

        self.means = np.array(
            self.preprocessing["means"],
            dtype=np.float32
        )

        self.stds = np.array(
            self.preprocessing["stds"],
            dtype=np.float32
        )

        # Prevent division by zero
        self.stds[
            self.stds == 0
        ] = 1.0

        # ----------------------------------------------------
        # Load trained BiLSTM
        # ----------------------------------------------------

        self.model = RULBiLSTM(
            input_size=22,
            hidden_size=64,
            num_layers=2,
            dropout=0.2
        )

        checkpoint = torch.load(
            MODEL_FILE,
            map_location=self.device
        )

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

            self.model.load_state_dict(
                checkpoint["model_state_dict"]
            )

        else:

            self.model.load_state_dict(
                checkpoint
            )

        self.model.to(self.device)

        self.model.eval()

        print(
            "BiLSTM RUL model loaded successfully."
        )

    # ========================================================
    # TEMPORAL VALIDITY
    # ========================================================

    def add_validity_features(self, df):

        df = df.copy()

        for feature in self.temporal_features:

            validity_name = (
                feature + "_valid"
            )

            df[validity_name] = (
                df[feature].notna().astype(np.float32)
            )

        return df

    # ========================================================
    # PREPROCESS SEQUENCE
    # ========================================================

    def preprocess(self, sequence):

        df = sequence.copy()

        # Add validity indicators BEFORE imputation
        df = self.add_validity_features(df)

        # ----------------------------------------------------
        # Base features
        # ----------------------------------------------------

        base_values = (
            df[self.base_features]
            .astype(np.float32)
            .to_numpy()
        )

        # Median imputation
        for i in range(
            base_values.shape[1]
        ):

            missing = np.isnan(
                base_values[:, i]
            )

            base_values[
                missing,
                i
            ] = self.medians[i]

        # Standardisation
        base_values = (
            base_values - self.means
        ) / self.stds

        # ----------------------------------------------------
        # Validity features
        # ----------------------------------------------------

        validity_features = [
            feature + "_valid"
            for feature in self.temporal_features
        ]

        validity_values = (
            df[validity_features]
            .astype(np.float32)
            .to_numpy()
        )

        # ----------------------------------------------------
        # Combine
        # ----------------------------------------------------

        processed = np.concatenate(
            [
                base_values,
                validity_values
            ],
            axis=1
        )

        # Safety check
        if processed.shape[1] != 22:

            raise ValueError(
                f"Expected 22 features, "
                f"got {processed.shape[1]}"
            )

        return processed

    # ========================================================
    # RUL PREDICTION
    # ========================================================

    def predict_rul(self, sequence):

        if len(sequence) < SEQUENCE_LENGTH:

            return np.nan

        sequence = (
            sequence
            .sort_values("date")
            .tail(SEQUENCE_LENGTH)
            .copy()
        )

        processed = self.preprocess(
            sequence
        )

        X = torch.tensor(
            processed,
            dtype=torch.float32
        ).unsqueeze(0)

        X = X.to(self.device)

        with torch.no_grad():

            prediction = (
                self.model(X)
                .item()
            )

        return max(
            0.0,
            float(prediction)
        )

    # ========================================================
    # SEVERITY
    # ========================================================

    @staticmethod
    def severity_from_thresholds(value, thresholds):
        """
        Return the severity level (0-4) for a change value given
        an ordered list of (bound, level) pairs. Missing values
        return 0 (no evidence, which never qualifies).

        Delegates to the shared rules in degradation_detector.py.
        """

        return detector.severity_from_thresholds(
            value,
            thresholds,
            missing=0,
        )

    # ========================================================
    # ADAPTIVE DEGRADATION
    # ========================================================

    def calculate_degradation_state(
        self,
        sequence
    ):

        sequence = (
            sequence
            .sort_values("date")
            .copy()
        )

        # ----------------------------------------------------
        # Determine whether each observation qualifies as
        # degradation (severity >= MIN_SEVERITY)
        # ----------------------------------------------------

        qualifying_flags = []

        for _, row in sequence.iterrows():

            smart1 = row[
                "smart_1_normalized_change_7d"
            ]

            smart187 = row[
                "smart_187_normalized_change_7d"
            ]

            smart1_severity = self.severity_from_thresholds(
                smart1,
                self.smart1_thresholds,
            )

            smart187_severity = self.severity_from_thresholds(
                smart187,
                self.smart187_thresholds,
            )

            combined_severity = max(
                smart1_severity,
                smart187_severity
            )

            qualifying_flags.append(
                combined_severity >= self.MIN_SEVERITY
            )

        # ----------------------------------------------------
        # Longest consecutive qualifying run
        # ----------------------------------------------------

        longest_run = 0
        current_run = 0

        for flag in qualifying_flags:

            if flag:

                current_run += 1

                longest_run = max(
                    longest_run,
                    current_run
                )

            else:

                current_run = 0

        # ----------------------------------------------------
        # Current consecutive run
        # ----------------------------------------------------

        current_run = 0

        for flag in reversed(
            qualifying_flags
        ):

            if flag:

                current_run += 1

            else:

                break

        # ----------------------------------------------------
        # Adaptive state
        #
        # Persistence is measured in consecutive qualifying
        # observations (severity >= MIN_SEVERITY, mild or
        # worse). The confirmation threshold is PERSISTENCE;
        # longer runs escalate the warning level. The ladder
        # itself lives in degradation_detector.py so every
        # detector escalates identically.
        # ----------------------------------------------------

        adaptive_state = detector.adaptive_state_name(
            detector.escalation_level(
                current_run,
                self.PERSISTENCE,
            )
        )

        return {
            "current_persistence": current_run,
            "longest_persistence": longest_run,
            "adaptive_state": adaptive_state
        }

    # ========================================================
    # PROCESS ONE HDD
    # ========================================================

    def process_hdd(
        self,
        serial_number,
        sequence
    ):

        sequence = (
            sequence
            .sort_values("date")
            .reset_index(drop=True)
        )

        latest = sequence.iloc[-1]

        rul = self.predict_rul(
            sequence
        )

        degradation = (
            self.calculate_degradation_state(
                sequence
            )
        )

                # Input variation warning
        if len(sequence) < SEQUENCE_LENGTH:
            input_variation_status = "INSUFFICIENT_HISTORY"
        else:
            processed = self.preprocess(
                sequence.sort_values("date").tail(SEQUENCE_LENGTH)
            )
            if np.all(np.std(processed, axis=0) < 1e-6):
                input_variation_status = "NO_INPUT_VARIATION"
            else:
                input_variation_status = "INPUT_VARIATION_PRESENT"

        # ----------------------------------------------------
        # Health feature availability
        # ----------------------------------------------------

        availability_columns = [
            "smart_1_available",
            "smart_5_available",
            "smart_187_available",
            "smart_197_available",
            "smart_198_available"
        ]

        available_count = int(
            latest[
                availability_columns
            ].sum()
        )

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        return {
            "serial_number": serial_number,
            "latest_date": latest["date"],
            "observations_used": min(
                len(sequence),
                SEQUENCE_LENGTH
            ),
            "health_features_available":
                available_count,

            "current_degradation_persistence":
                degradation[
                    "current_persistence"
                ],

            "longest_degradation_persistence":
                degradation[
                    "longest_persistence"
                ],

            "adaptive_state":
                degradation[
                    "adaptive_state"
                ],

                        "input_variation_status":
                input_variation_status,

            "predicted_rul_days":
                rul
        }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("ADAPTIVE DIGITAL TWIN")
    print("=" * 80)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load V5 state
    # --------------------------------------------------------

    print(
        "\nLoading Digital Twin V5 state..."
    )

    df = pd.read_csv(
        V5_STATE_FILE,
        parse_dates=["date"]
    )

    print(
        "Rows:",
        len(df)
    )

    print(
        "HDDs:",
        df["serial_number"].nunique()
    )

    # --------------------------------------------------------
    # Initialise Twin
    # --------------------------------------------------------

    twin = AdaptiveDigitalTwin()

    # --------------------------------------------------------
    # Process HDDs
    # --------------------------------------------------------

    results = []

    serial_numbers = (
        df["serial_number"]
        .dropna()
        .unique()
    )

    print(
        "\nProcessing HDDs..."
    )

    for index, serial in enumerate(
        serial_numbers,
        start=1
    ):

        hdd = df[
            df["serial_number"] == serial
        ].sort_values("date")

        if len(hdd) < SEQUENCE_LENGTH:

            continue

        result = twin.process_hdd(
            serial,
            hdd
        )

        results.append(
            result
        )

        if index % 10000 == 0:

            print(
                f"Processed {index} HDDs"
            )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "ADAPTIVE DIGITAL TWIN COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        "HDDs evaluated:",
        len(results_df)
    )

    print(
        "\nAdaptive state distribution:"
    )

    print(
        results_df[
            "adaptive_state"
        ].value_counts()
    )

    print(
        "\nRUL summary:"
    )

    print(
        results_df[
            "predicted_rul_days"
        ].describe()
    )

    print(
        "\nSaved:"
    )

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":

    main()