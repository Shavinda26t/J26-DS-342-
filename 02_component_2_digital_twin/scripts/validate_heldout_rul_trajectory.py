import os
import pandas as pd
import numpy as np

from adaptive_digital_twin import AdaptiveDigitalTwin


# ============================================================
# CONFIGURATION
# ============================================================

V5_STATE_FILE = (
    r".\results\digital_twin_state_v5.csv"
)

TEST_HDD_FILE = (
    r".\results\rul_sequences\test_hdds.csv"
)

OUTPUT_DIR = (
    r".\results\digital_twin_rul"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "heldout_rul_trajectory_results.csv"
)

CHECKPOINTS = [30, 14, 7, 1]

SEQUENCE_LENGTH = 30


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("HELD-OUT RUL TRAJECTORY VALIDATION")
    print("=" * 80)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load original 117 test HDDs
    # --------------------------------------------------------

    print(
        "\nLoading original held-out test HDDs..."
    )

    test_hdds = pd.read_csv(
        TEST_HDD_FILE
    )

    test_hdds["serial_number"] = (
        test_hdds["serial_number"]
        .astype(str)
        .str.strip()
    )

    test_serials = set(
        test_hdds["serial_number"]
    )

    print(
        f"Test HDDs: {len(test_serials)}"
    )

    # --------------------------------------------------------
    # Load V5 state for test HDDs
    # --------------------------------------------------------

    print(
        "\nLoading V5 Digital Twin state..."
    )

    selected_chunks = []

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
            .isin(test_serials)
        ]

        if not matched.empty:

            selected_chunks.append(
                matched
            )

    if not selected_chunks:

        raise ValueError(
            "No test HDDs found in V5 state."
        )

    test_state = pd.concat(
        selected_chunks,
        ignore_index=True
    )

    test_state = (
        test_state
        .sort_values(
            ["serial_number", "date"]
        )
        .reset_index(drop=True)
    )

    print(
        f"Loaded observations: "
        f"{len(test_state)}"
    )

    print(
        f"HDDs found: "
        f"{test_state['serial_number'].nunique()}"
    )

    # --------------------------------------------------------
    # Load Adaptive Digital Twin
    # --------------------------------------------------------

    print(
        "\nLoading Digital Twin..."
    )

    twin = AdaptiveDigitalTwin()

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results = []

    # --------------------------------------------------------
    # Process HDDs
    # --------------------------------------------------------

    for index, serial in enumerate(
        sorted(test_serials),
        start=1
    ):

        hdd = test_state[
            test_state["serial_number"]
            == serial
        ].copy()

        if hdd.empty:

            continue

        hdd = (
            hdd
            .sort_values("date")
            .reset_index(drop=True)
        )

        # ----------------------------------------------------
        # Failure date
        # ----------------------------------------------------

        failure_dates = hdd[
            hdd["failure"] == 1
        ]["date"]

        if failure_dates.empty:

            continue

        failure_date = (
            failure_dates.iloc[0]
        )

        # ----------------------------------------------------
        # Pre-failure observations only
        # ----------------------------------------------------

        pre_failure = hdd[
            hdd["failure"] == 0
        ].copy()

        pre_failure = (
            pre_failure
            .sort_values("date")
            .reset_index(drop=True)
        )

        n_pre_failure = len(
            pre_failure
        )

        # ----------------------------------------------------
        # Each checkpoint is measured backward
        # from the latest pre-failure observation.
        #
        # checkpoint = 1
        # -> final pre-failure prediction
        #
        # checkpoint = 7
        # -> seven observations earlier
        #
        # etc.
        # ----------------------------------------------------

        for checkpoint in CHECKPOINTS:

            endpoint_index = (
                n_pre_failure - checkpoint
            )

            if endpoint_index < (
                SEQUENCE_LENGTH - 1
            ):

                continue

            # Include observations through
            # the selected endpoint.
            sequence = pre_failure[
                :endpoint_index + 1
            ].copy()

            sequence = sequence.tail(
                SEQUENCE_LENGTH
            )

            sequence_end_date = (
                sequence["date"].iloc[-1]
            )

            actual_rul = (
                failure_date -
                sequence_end_date
            ).days

            # ------------------------------------------------
            # RUL prediction
            # ------------------------------------------------

            predicted_rul = twin.predict_rul(
                sequence
            )

            absolute_error = abs(
                predicted_rul -
                actual_rul
            )

            signed_error = (
                predicted_rul -
                actual_rul
            )

            # ------------------------------------------------
            # Adaptive state at same checkpoint
            # ------------------------------------------------

            degradation = (
                twin.calculate_degradation_state(
                    sequence
                )
            )

            # ------------------------------------------------
            # Health availability
            # ------------------------------------------------

            latest = sequence.iloc[-1]

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

            # ------------------------------------------------
            # Save result
            # ------------------------------------------------

            results.append({

                "serial_number":
                    serial,

                "checkpoint_observations":
                    checkpoint,

                "sequence_start_date":
                    sequence["date"].iloc[0],

                "sequence_end_date":
                    sequence_end_date,

                "failure_date":
                    failure_date,

                "actual_rul_days":
                    actual_rul,

                "predicted_rul_days":
                    predicted_rul,

                "error_days":
                    signed_error,

                "absolute_error_days":
                    absolute_error,

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

                "health_features_available":
                    available_count
            })

        if index % 20 == 0:

            print(
                f"Processed "
                f"{index}/{len(test_serials)} HDDs"
            )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "HELD-OUT RUL TRAJECTORY RESULTS"
    )

    print(
        "=" * 80
    )

    print(
        f"Total predictions: "
        f"{len(results_df)}"
    )

    print(
        f"Unique HDDs evaluated: "
        f"{results_df['serial_number'].nunique()}"
    )

    # --------------------------------------------------------
    # Metrics by checkpoint
    # --------------------------------------------------------

    print(
        "\nRUL performance by checkpoint:"
    )

    summary_rows = []

    for checkpoint in CHECKPOINTS:

        subset = results_df[
            results_df[
                "checkpoint_observations"
            ] == checkpoint
        ]

        if subset.empty:

            continue

        mae = (
            subset[
                "absolute_error_days"
            ].mean()
        )

        rmse = np.sqrt(
            np.mean(
                subset[
                    "error_days"
                ] ** 2
            )
        )

        bias = (
            subset[
                "error_days"
            ].mean()
        )

        summary_rows.append({

            "checkpoint_observations":
                checkpoint,

            "n_predictions":
                len(subset),

            "n_hdds":
                subset[
                    "serial_number"
                ].nunique(),

            "mean_actual_rul":
                subset[
                    "actual_rul_days"
                ].mean(),

            "mean_predicted_rul":
                subset[
                    "predicted_rul_days"
                ].mean(),

            "MAE":
                mae,

            "RMSE":
                rmse,

            "bias":
                bias
        })

        print(
            f"{checkpoint:>2} observations before final: "
            f"n={len(subset):3d}, "
            f"MAE={mae:.3f}, "
            f"RMSE={rmse:.3f}, "
            f"Bias={bias:+.3f}"
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_file = os.path.join(
        OUTPUT_DIR,
        "heldout_rul_trajectory_metrics.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False
    )

    # --------------------------------------------------------
    # Adaptive state by checkpoint
    # --------------------------------------------------------

    print(
        "\nAdaptive state by checkpoint:"
    )

    for checkpoint in CHECKPOINTS:

        subset = results_df[
            results_df[
                "checkpoint_observations"
            ] == checkpoint
        ]

        if subset.empty:

            continue

        print(
            f"\nCheckpoint "
            f"{checkpoint}:"
        )

        print(
            subset[
                "adaptive_state"
            ].value_counts()
        )

    # --------------------------------------------------------
    # Persistence >=7
    # --------------------------------------------------------

    print(
        "\nStrong persistence (>=7) by checkpoint:"
    )

    for checkpoint in CHECKPOINTS:

        subset = results_df[
            results_df[
                "checkpoint_observations"
            ] == checkpoint
        ]

        if subset.empty:

            continue

        count = (
            subset[
                "current_degradation_persistence"
            ] >= 7
        ).sum()

        print(
            f"Checkpoint {checkpoint}: "
            f"{count}/{len(subset)}"
        )

    print(
        "\nSaved:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        summary_file
    )

    print(
        "=" * 80
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()