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
    "heldout_test_adaptive_digital_twin_results.csv"
)

SEQUENCE_LENGTH = 30


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("HELD-OUT TEST SET - ADAPTIVE DIGITAL TWIN VALIDATION")
    print("=" * 80)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load exact original test HDD IDs
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
        f"Original test HDDs: {len(test_serials)}"
    )

    # --------------------------------------------------------
    # Load V5 state for ONLY the 117 test HDDs
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
            "No held-out test HDDs were found "
            "in the V5 Digital Twin state."
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
        f"Loaded observations: {len(test_state)}"
    )

    print(
        f"HDDs found in V5: "
        f"{test_state['serial_number'].nunique()}"
    )

    # --------------------------------------------------------
    # Load Adaptive Digital Twin
    # --------------------------------------------------------

    print(
        "\nLoading Adaptive Digital Twin..."
    )

    twin = AdaptiveDigitalTwin()

    # --------------------------------------------------------
    # Process each held-out HDD
    # --------------------------------------------------------

    results = []

    print(
        "\nProcessing held-out HDDs..."
    )

    for index, serial in enumerate(
        sorted(test_serials),
        start=1
    ):

        hdd = test_state[
            test_state["serial_number"]
            == serial
        ].copy()

        if hdd.empty:

            print(
                f"WARNING: {serial} not found"
            )

            continue

        hdd = (
            hdd
            .sort_values("date")
            .reset_index(drop=True)
        )

        # ----------------------------------------------------
        # Find failure date
        # ----------------------------------------------------

        failure_dates = hdd[
            hdd["failure"] == 1
        ]["date"]

        if failure_dates.empty:

            print(
                f"WARNING: {serial} has no failure record"
            )

            continue

        failure_date = (
            failure_dates.iloc[0]
        )

        # ----------------------------------------------------
        # Remove failure-day observation
        # ----------------------------------------------------

        pre_failure = hdd[
            hdd["failure"] == 0
        ].copy()

        pre_failure = (
            pre_failure
            .sort_values("date")
            .reset_index(drop=True)
        )

        pre_failure_count = len(
            pre_failure
        )

        # ----------------------------------------------------
        # Need at least 30 observations
        # ----------------------------------------------------

        if pre_failure_count < SEQUENCE_LENGTH:

            results.append({
                "serial_number": serial,
                "failure_date": failure_date,
                "pre_failure_observations":
                    pre_failure_count,
                "latest_pre_failure_date": (
                    pre_failure["date"].iloc[-1]
                    if pre_failure_count > 0
                    else pd.NaT
                ),
                "actual_rul_days": np.nan,
                "predicted_rul_days": np.nan,
                "absolute_error_days": np.nan,
                "current_degradation_persistence":
                    np.nan,
                "longest_degradation_persistence":
                    np.nan,
                "adaptive_state":
                    "Insufficient History",
                "health_features_available":
                    np.nan,
                "evaluated": False
            })

            continue

        # ----------------------------------------------------
        # Latest pre-failure state
        # ----------------------------------------------------

        latest = pre_failure.iloc[-1]

        actual_rul = (
            failure_date -
            latest["date"]
        ).days

        # ----------------------------------------------------
        # RUL prediction
        # ----------------------------------------------------

        predicted_rul = twin.predict_rul(
            pre_failure
        )

        # ----------------------------------------------------
        # Adaptive degradation state
        # ----------------------------------------------------

        degradation = (
            twin.calculate_degradation_state(
                pre_failure
            )
        )

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
        # Error
        # ----------------------------------------------------

        absolute_error = abs(
            predicted_rul -
            actual_rul
        )

        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        results.append({

            "serial_number":
                serial,

            "failure_date":
                failure_date,

            "pre_failure_observations":
                pre_failure_count,

            "latest_pre_failure_date":
                latest["date"],

            "actual_rul_days":
                actual_rul,

            "predicted_rul_days":
                predicted_rul,

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
                available_count,

            "evaluated":
                True
        })

        if index % 20 == 0:

            print(
                f"Processed {index}/"
                f"{len(test_serials)} HDDs"
            )

    # ========================================================
    # RESULTS DATAFRAME
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Save per-HDD results
    # --------------------------------------------------------

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    evaluated = results_df[
        results_df["evaluated"] == True
    ].copy()

    print(
        "\n" + "=" * 80
    )

    print(
        "HELD-OUT ADAPTIVE DIGITAL TWIN RESULTS"
    )

    print(
        "=" * 80
    )

    print(
        f"Original test HDDs: "
        f"{len(test_serials)}"
    )

    print(
        f"Evaluated HDDs: "
        f"{len(evaluated)}"
    )

    print(
        f"Not evaluated: "
        f"{len(test_serials) - len(evaluated)}"
    )

    # --------------------------------------------------------
    # RUL metrics
    # --------------------------------------------------------

    if not evaluated.empty:

        mae = (
            evaluated[
                "absolute_error_days"
            ].mean()
        )

        rmse = np.sqrt(
            np.mean(
                (
                    evaluated[
                        "predicted_rul_days"
                    ]
                    -
                    evaluated[
                        "actual_rul_days"
                    ]
                ) ** 2
            )
        )

        bias = (
            evaluated[
                "predicted_rul_days"
            ]
            -
            evaluated[
                "actual_rul_days"
            ]
        ).mean()

        print(
            "\nRUL performance:"
        )

        print(
            f"MAE:  {mae:.3f} days"
        )

        print(
            f"RMSE: {rmse:.3f} days"
        )

        print(
            f"Bias: {bias:+.3f} days"
        )

        # ----------------------------------------------------
        # RUL range
        # ----------------------------------------------------

        def rul_range(value):

            if value <= 7:
                return "1-7"

            elif value <= 14:
                return "8-14"

            elif value <= 30:
                return "15-30"

            elif value <= 60:
                return "31-60"

            else:
                return ">60"

        evaluated["actual_rul_range"] = (
            evaluated[
                "actual_rul_days"
            ].apply(rul_range)
        )

        print(
            "\nRUL performance by actual RUL range:"
        )

        for range_name in [
            "1-7",
            "8-14",
            "15-30",
            "31-60",
            ">60"
        ]:

            subset = evaluated[
                evaluated[
                    "actual_rul_range"
                ] == range_name
            ]

            if subset.empty:
                continue

            range_mae = (
                subset[
                    "absolute_error_days"
                ].mean()
            )

            print(
                f"{range_name:>5} days: "
                f"n={len(subset):3d}, "
                f"MAE={range_mae:.3f}"
            )

        # ----------------------------------------------------
        # Adaptive state distribution
        # ----------------------------------------------------

        print(
            "\nAdaptive state distribution:"
        )

        print(
            evaluated[
                "adaptive_state"
            ].value_counts()
        )

        # ----------------------------------------------------
        # Persistence statistics
        # ----------------------------------------------------

        print(
            "\nPersistence statistics:"
        )

        print(
            "Current persistence >=3:",
            (
                evaluated[
                    "current_degradation_persistence"
                ] >= 3
            ).sum()
        )

        print(
            "Current persistence >=5:",
            (
                evaluated[
                    "current_degradation_persistence"
                ] >= 5
            ).sum()
        )

        print(
            "Current persistence >=7:",
            (
                evaluated[
                    "current_degradation_persistence"
                ] >= 7
            ).sum()
        )

        print(
            "Longest persistence >=3:",
            (
                evaluated[
                    "longest_degradation_persistence"
                ] >= 3
            ).sum()
        )

        print(
            "Longest persistence >=5:",
            (
                evaluated[
                    "longest_degradation_persistence"
                ] >= 5
            ).sum()
        )

        print(
            "Longest persistence >=7:",
            (
                evaluated[
                    "longest_degradation_persistence"
                ] >= 7
            ).sum()
        )

        # ----------------------------------------------------
        # SMART187 availability
        # ----------------------------------------------------

        print(
            "\nSMART187 availability:"
        )

        smart187_available = (
            evaluated[
                "health_features_available"
            ] == 5
        )

        print(
            f"All 5 health features available: "
            f"{smart187_available.sum()}"
        )

        print(
            f"SMART187 unavailable: "
            f"{(~smart187_available).sum()}"
        )

    # --------------------------------------------------------
    # Save final results
    # --------------------------------------------------------

    print(
        "\nSaved:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":

    main()