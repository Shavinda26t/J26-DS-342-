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
    r"\heldout_rul_uncertainty_summary.csv"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("HELD-OUT RUL UNCERTAINTY ANALYSIS")
    print("=" * 80)

    # --------------------------------------------------------
    # Load held-out trajectory predictions
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
        "checkpoint_observations",
        "actual_rul_days",
        "predicted_rul_days",
        "absolute_error_days"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    # --------------------------------------------------------
    # Recalculate signed error
    #
    # Positive = RUL overestimation
    # Negative = RUL underestimation
    # --------------------------------------------------------

    df["signed_error_days"] = (
        df["predicted_rul_days"]
        -
        df["actual_rul_days"]
    )

    # --------------------------------------------------------
    # Analyse each checkpoint
    # --------------------------------------------------------

    rows = []

    checkpoints = sorted(
        df[
            "checkpoint_observations"
        ].unique(),
        reverse=True
    )

    for checkpoint in checkpoints:

        subset = df[
            df[
                "checkpoint_observations"
            ] == checkpoint
        ].copy()

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

        if len(errors) == 0:

            continue

        # ----------------------------------------------------
        # Empirical error percentiles
        # ----------------------------------------------------

        q50 = np.percentile(
            errors,
            50
        )

        q75 = np.percentile(
            errors,
            75
        )

        q90 = np.percentile(
            errors,
            90
        )

        q95 = np.percentile(
            errors,
            95
        )

        q99 = np.percentile(
            errors,
            99
        )

        mae = np.mean(
            errors
        )

        bias = np.mean(
            signed_errors
        )

        # ----------------------------------------------------
        # Save summary
        # ----------------------------------------------------

        rows.append({

            "checkpoint_observations":
                checkpoint,

            "n_predictions":
                len(errors),

            "n_hdds":
                subset[
                    "serial_number"
                ].nunique(),

            "mae_days":
                mae,

            "bias_days":
                bias,

            "median_absolute_error_days":
                q50,

            "75_percentile_error_days":
                q75,

            "90_percentile_error_days":
                q90,

            "95_percentile_error_days":
                q95,

            "99_percentile_error_days":
                q99
        })

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        print(
            f"\nCheckpoint: "
            f"{checkpoint} observations before failure"
        )

        print(
            f"  Predictions: "
            f"{len(errors)}"
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
            f"  Bias: "
            f"{bias:+.3f} days"
        )

        print(
            f"  Median error: "
            f"{q50:.3f} days"
        )

        print(
            f"  75% error: "
            f"{q75:.3f} days"
        )

        print(
            f"  90% error: "
            f"{q90:.3f} days"
        )

        print(
            f"  95% error: "
            f"{q95:.3f} days"
        )

        print(
            f"  99% error: "
            f"{q99:.3f} days"
        )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary = pd.DataFrame(
        rows
    )

    summary.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "UNCERTAINTY ANALYSIS COMPLETE"
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