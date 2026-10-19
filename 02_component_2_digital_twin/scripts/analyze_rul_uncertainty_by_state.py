import os
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

RESULT_FILE = (
    r".\results\digital_twin_rul"
    r"\heldout_rul_trajectory_results.csv"
)

OUTPUT_FILE = (
    r".\results\digital_twin_rul"
    r"\heldout_rul_uncertainty_by_state.csv"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("HELD-OUT RUL UNCERTAINTY BY ADAPTIVE STATE")
    print("=" * 80)

    # --------------------------------------------------------
    # Load held-out trajectory results
    # --------------------------------------------------------

    df = pd.read_csv(
        RESULT_FILE
    )

    print(
        f"\nLoaded predictions: {len(df)}"
    )

    print(
        f"Unique HDDs: "
        f"{df['serial_number'].nunique()}"
    )

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    required_columns = [
        "serial_number",
        "adaptive_state",
        "actual_rul_days",
        "predicted_rul_days",
        "absolute_error_days"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    # --------------------------------------------------------
    # Recalculate signed error
    # --------------------------------------------------------

    df["signed_error_days"] = (
        df["predicted_rul_days"]
        -
        df["actual_rul_days"]
    )

    # --------------------------------------------------------
    # Analyse each adaptive state
    # --------------------------------------------------------

    results = []

    state_order = [
        "Normal / Monitor",
        "Early Warning",
        "Elevated Degradation",
        "Strong Persistent Degradation"
    ]

    print(
        "\nRUL performance by adaptive state:"
    )

    for state in state_order:

        subset = df[
            df["adaptive_state"] == state
        ].copy()

        if subset.empty:

            print(
                f"\n{state}: no observations"
            )

            continue

        errors = (
            subset[
                "absolute_error_days"
            ]
            .dropna()
            .to_numpy()
        )

        signed_errors = (
            subset[
                "signed_error_days"
            ]
            .dropna()
            .to_numpy()
        )

        # ----------------------------------------------------
        # Error statistics
        # ----------------------------------------------------

        mae = np.mean(
            errors
        )

        rmse = np.sqrt(
            np.mean(
                signed_errors ** 2
            )
        )

        bias = np.mean(
            signed_errors
        )

        median_error = np.percentile(
            errors,
            50
        )

        error_75 = np.percentile(
            errors,
            75
        )

        error_90 = np.percentile(
            errors,
            90
        )

        error_95 = np.percentile(
            errors,
            95
        )

        error_99 = np.percentile(
            errors,
            99
        )

        results.append({

            "adaptive_state":
                state,

            "n_predictions":
                len(subset),

            "n_hdds":
                subset[
                    "serial_number"
                ].nunique(),

            "mae_days":
                mae,

            "rmse_days":
                rmse,

            "bias_days":
                bias,

            "median_absolute_error_days":
                median_error,

            "75_percentile_error_days":
                error_75,

            "90_percentile_error_days":
                error_90,

            "95_percentile_error_days":
                error_95,

            "99_percentile_error_days":
                error_99,

            "mean_actual_rul_days":
                subset[
                    "actual_rul_days"
                ].mean(),

            "mean_predicted_rul_days":
                subset[
                    "predicted_rul_days"
                ].mean()
        })

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        print(
            f"\n{state}"
        )

        print(
            f"  Predictions: "
            f"{len(subset)}"
        )

        print(
            f"  HDDs: "
            f"{subset['serial_number'].nunique()}"
        )

        print(
            f"  MAE: "
            f"{mae:.3f} days"
        )

        print(
            f"  RMSE: "
            f"{rmse:.3f} days"
        )

        print(
            f"  Bias: "
            f"{bias:+.3f} days"
        )

        print(
            f"  Median error: "
            f"{median_error:.3f} days"
        )

        print(
            f"  90% error: "
            f"{error_90:.3f} days"
        )

        print(
            f"  95% error: "
            f"{error_95:.3f} days"
        )

        print(
            f"  99% error: "
            f"{error_99:.3f} days"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    summary = pd.DataFrame(
        results
    )

    summary.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "STATE-BASED UNCERTAINTY ANALYSIS COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        "Saved:"
    )

    print(
        OUTPUT_FILE
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()