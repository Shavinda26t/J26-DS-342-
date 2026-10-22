import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# Paths (absolute, relative to repo)
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = (
    REPO_ROOT
    / "02_component_2_digital_twin"
    / "results"
    / "rul_sequences"
    / "sequence_length_experiments"
    / "seq_30"
    / "bilstm"
    / "best_bilstm_model.pt"
)
PREPROCESSING_PATH = (
    REPO_ROOT
    / "02_component_2_digital_twin"
    / "results"
    / "rul_sequences"
    / "sequence_length_experiments"
    / "seq_30"
    / "preprocessed"
    / "preprocessing_parameters.json"
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

class Component2RULPredictor:
    """
    Uses the trained 30-step BiLSTM to estimate HDD RUL.
    """

    def __init__(
        self,
        model_path=MODEL_PATH,
        preprocessing_path=PREPROCESSING_PATH
    ):

        self.model_path = Path(model_path)
        self.preprocessing_path = Path(preprocessing_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}"
            )
        if not self.preprocessing_path.exists():
            raise FileNotFoundError(
                f"Preprocessing file not found: {self.preprocessing_path}"
            )

        # Load preprocessing parameters
        with open(
            self.preprocessing_path,
            "r",
            encoding="utf-8"
        ) as file:
            self.preprocessing = json.load(file)

        # Load model
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

    def add_validity_features(self, dataframe):
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

    def preprocess_sequence(
        self,
        dataframe
    ):
        df = dataframe.copy()

        df = self.add_validity_features(df)

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

            values = np.where(
                np.isnan(values),
                medians[index],
                values
            )

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

    def predict(
        self,
        dataframe
    ):
        if len(dataframe) < 30:
            raise ValueError(
                "At least 30 observations are "
                "required for the BiLSTM."
            )

        df = dataframe.copy()

        if "date" in df.columns:
            df = df.sort_values(
                "date"
            )

        df = df.tail(30)

        processed = self.preprocess_sequence(
            df
        )

        tensor = torch.tensor(
            processed,
            dtype=torch.float32
        ).unsqueeze(0)

        tensor = tensor.to(
            DEVICE
        )

        with torch.no_grad():
            prediction = self.model(
                tensor
            )

        rul = float(
            prediction
            .cpu()
            .item()
        )

        return max(0.0, rul)


# Alias for compatibility
DigitalTwinRULPredictor = Component2RULPredictor
