import json
import os

import numpy as np
import pandas as pd
import torch

from adaptive_digital_twin import AdaptiveDigitalTwin


# ============================================================
# FINAL ADAPTIVE DIGITAL TWIN SYSTEM
# ============================================================

class AdaptiveDigitalTwinSystem:

    def __init__(self):

        print("=" * 80)
        print("ADAPTIVE DIGITAL TWIN SYSTEM")
        print("=" * 80)

        # ----------------------------------------------------
        # Existing validated Digital Twin components
        # ----------------------------------------------------

        self.twin = AdaptiveDigitalTwin()

        print(
            "\nDigital Twin system initialized."
        )

    # ========================================================
    # UPDATE DIGITAL TWIN
    # ========================================================

    def update_state(
        self,
        hdd_history
    ):

        if hdd_history.empty:

            raise ValueError(
                "HDD history is empty."
            )

        history = (
            hdd_history
            .copy()
            .sort_values("date")
            .reset_index(drop=True)
        )

        # ----------------------------------------------------
        # Latest state
        # ----------------------------------------------------

        latest = history.iloc[-1]

        # ----------------------------------------------------
        # Feature availability
        # ----------------------------------------------------

        availability_columns = [
            "smart_1_available",
            "smart_5_available",
            "smart_187_available",
            "smart_197_available",
            "smart_198_available"
        ]

        available_features = int(
            latest[
                availability_columns
            ].sum()
        )

        # ----------------------------------------------------
        # Adaptive degradation
        # ----------------------------------------------------

        degradation = (
            self.twin.calculate_degradation_state(
                history
            )
        )

        # ----------------------------------------------------
        # RUL prediction
        # ----------------------------------------------------

        predicted_rul = (
            self.twin.predict_rul(
                history
            )
        )

        # ----------------------------------------------------
        # Final Digital Twin state
        # ----------------------------------------------------

        state = {

            "serial_number":
                str(latest["serial_number"]),

            "latest_date":
                latest["date"],

            "observations_available":
                len(history),

            "health_features_available":
                available_features,

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

            "predicted_rul_days":
                predicted_rul
        }

        return state

    # ========================================================
    # GET HUMAN-READABLE STATUS
    # ========================================================

    def get_status(
        self,
        state
    ):

        return (
            f"HDD: {state['serial_number']}\n"
            f"Latest observation: "
            f"{state['latest_date']}\n"
            f"Health features available: "
            f"{state['health_features_available']}/5\n"
            f"Current persistence: "
            f"{state['current_degradation_persistence']}\n"
            f"Longest persistence: "
            f"{state['longest_degradation_persistence']}\n"
            f"Adaptive state: "
            f"{state['adaptive_state']}\n"
            f"Predicted RUL: "
            f"{state['predicted_rul_days']:.2f} days"
        )


# ============================================================
# MAIN TEST
# ============================================================

def main():

    TEST_SERIAL = "Z302SZP7"

    V5_STATE_FILE = (
        r".\results\digital_twin_state_v5.csv"
    )

    print(
        "\nLoading test HDD..."
    )

    chunks = []

    for chunk in pd.read_csv(
        V5_STATE_FILE,
        parse_dates=["date"],
        chunksize=100000
    ):

        chunk["serial_number"] = (
            chunk["serial_number"]
            .astype(str)
            .str.strip()
        )

        matched = chunk[
            chunk["serial_number"]
            == TEST_SERIAL
        ]

        if not matched.empty:

            chunks.append(matched)

    if not chunks:

        raise ValueError(
            f"HDD {TEST_SERIAL} not found."
        )

    hdd = pd.concat(
        chunks,
        ignore_index=True
    )

    hdd = (
        hdd
        .sort_values("date")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Remove failure-day observation
    # --------------------------------------------------------

    pre_failure = hdd[
        hdd["failure"] == 0
    ].copy()

    pre_failure = (
        pre_failure
        .sort_values("date")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Create system
    # --------------------------------------------------------

    system = (
        AdaptiveDigitalTwinSystem()
    )

    # --------------------------------------------------------
    # Update Twin
    # --------------------------------------------------------

    print(
        "\nUpdating Digital Twin..."
    )

    state = system.update_state(
        pre_failure
    )

    # --------------------------------------------------------
    # Display final state
    # --------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "FINAL DIGITAL TWIN STATE"
    )

    print(
        "=" * 80
    )

    print(
        system.get_status(
            state
        )
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":

    main()