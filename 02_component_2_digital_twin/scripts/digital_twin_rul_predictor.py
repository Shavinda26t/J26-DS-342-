import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# Paths
# ============================================================

MODEL_PATH = (
    r".\results\rul_sequences\sequence_length_experiments"
    r"\seq_30\bilstm\best_bilstm_model.pt"
)

PREPROCESSING_PATH = (
    r".\results\rul_sequences\sequence_length_experiments"
    r"\seq_30\preprocessed\preprocessing_parameters.json"
)


# ============================================================
# Device
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Exact BiLSTM architecture used during training
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

        return self.fc(
            last_output
        ).squeeze(1)


# ============================================================
# Digital Twin RUL Predictor
# ============================================================

class DigitalTwinRULPredictor:
    """
    Uses the trained 30-step BiLSTM to estimate HDD RUL.

    Input:
        30 chronological observations with the 16 base
        Digital Twin features.

    Output:
        RUL estimate in observations/days represented by
        the training target.
    """

    def __init__(
        self,
        model_path=MODEL_PATH,
        preprocessing_path=PREPROCESSING_PATH
    ):

        self.model_path = model_path
        self.preprocessing_path = preprocessing_path

        # ----------------------------------------------------
        # Load preprocessing parameters
        # ----------------------------------------------------

        with open(
            self.preprocessing_path,
            "r",
            encoding="utf-8"
        ) as file:

            self.preprocessing = json.load(file)

        # ----------------------------------------------------
        # Load model
        # ----------------------------------------------------

        self.model = RULBiLSTM()

        state_dict = torch.load(
            self.model_path,
            map_location=DEVICE
        )

        self.model.load_state_dict(
            state_dict
        )

        self.model.to(DEVICE)

        self.model.eval()

        print(
            "RUL BiLSTM loaded successfully."
        )

        print(
            "Device:",
            DEVICE
        )

    # ========================================================
    # Feature definitions
    # ========================================================

    @property
    def base_features(self):

        return self.preprocessing[
            "base_features"
        ]

    @property
    def temporal_features(self):

        return self.preprocessing[
            "temporal_features"
        ]

    # ========================================================
    # Build validity features
    # ========================================================

    def add_validity_features(self, dataframe):
        """
        Create the six temporal validity indicators.

        1 = temporal feature was available
        0 = temporal feature was unavailable

        Missing temporal values are not treated as healthy.
        """

        df = dataframe.copy()

        for feature in self.temporal_features:

            validity_name = (
                feature + "_valid"
            )

            df[validity_name] = (
                df[feature]
                .notna()
                .astype(np.float32)
            )

        return df

    # ========================================================
    # Preprocess sequence
    # ========================================================

    def preprocess_sequence(
        self,
        dataframe
    ):
        """
        Apply the same train-only preprocessing used for
        the 30-step BiLSTM.

        Steps:
        1. Preserve temporal validity indicators.
        2. Median-impute the 16 base features.
        3. Standardize the 16 base features using the
           training-only mean and standard deviation.
        4. Append the six validity features.
        """

        df = dataframe.copy()

        # ----------------------------------------------------
        # Add validity indicators BEFORE imputation
        # ----------------------------------------------------

        df = self.add_validity_features(df)

        # ----------------------------------------------------
        # Process 16 base features
        # ----------------------------------------------------

        processed = []

        medians = self.preprocessing["medians"]
        means = self.preprocessing["means"]
        stds = self.preprocessing["stds"]

        for index, feature in enumerate(
            self.base_features
        ):

            values = pd.to_numeric(
                df[feature],
                errors="coerce"
            ).to_numpy(
                dtype=np.float32
            )

            # Median imputation using training parameter.
            values = np.where(
                np.isnan(values),
                medians[index],
                values
            )

            # Standardisation using training parameters.
            std = stds[index]

            if std < 1e-8:
                std = 1.0

            values = (
                values - means[index]
            ) / std

            processed.append(values)

        base_array = np.column_stack(
            processed
        )

        # ----------------------------------------------------
        # Append six validity features
        # ----------------------------------------------------

        validity_arrays = []

        for feature in self.temporal_features:

            validity_name = (
                feature + "_valid"
            )

            validity_arrays.append(
                df[validity_name]
                .to_numpy(
                    dtype=np.float32
                )
            )

        validity_array = np.column_stack(
            validity_arrays
        )

        # ----------------------------------------------------
        # Final 22-feature matrix
        # ----------------------------------------------------

        final_array = np.concatenate(
            [
                base_array,
                validity_array
            ],
            axis=1
        )

        if final_array.shape[1] != 22:

            raise ValueError(
                "Expected 22 features, got "
                f"{final_array.shape[1]}"
            )

        return final_array.astype(
            np.float32
        )

    # ========================================================
    # Predict RUL
    # ========================================================

    def predict(
        self,
        dataframe
    ):
        """
        Predict RUL from the most recent 30 observations.
        """

        if len(dataframe) < 30:

            raise ValueError(
                "At least 30 observations are "
                "required for the BiLSTM."
            )

        # Keep chronological order.
        df = dataframe.copy()

        if "date" in df.columns:

            df = df.sort_values(
                "date"
            )

        # Use the latest 30 observations.
        df = df.tail(30)

        processed = self.preprocess_sequence(
            df
        )

        # Shape:
        # (1, 30, 22)

        tensor = torch.tensor(
            processed,
            dtype=torch.float32
        ).unsqueeze(0)

        tensor = tensor.to(
            DEVICE
        )

        # ----------------------------------------------------
        # Model prediction
        # ----------------------------------------------------

        with torch.no_grad():

            prediction = self.model(
                tensor
            )

        rul = float(
            prediction
            .cpu()
            .item()
        )

        return rul


# ============================================================
# Simple test
# ============================================================

if __name__ == "__main__":

    predictor = DigitalTwinRULPredictor()

    print(
        "\nPredictor ready."
    )

    print(
        "Expected input: "
        "30 observations × 22 processed features."
    )