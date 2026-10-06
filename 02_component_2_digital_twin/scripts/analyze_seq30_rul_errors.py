import os
import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

BASE = r".\results\rul_sequences\sequence_length_experiments"

PREDICTIONS = os.path.join(
    BASE,
    "seq_30",
    "lstm",
    "test_predictions.csv"
)


# ============================================================
# Load predictions
# ============================================================

df = pd.read_csv(PREDICTIONS)

print("Total test predictions:", len(df))


# ============================================================
# Calculate error
# ============================================================

df["error"] = (
    df["predicted_rul"] -
    df["actual_rul"]
)

df["absolute_error"] = np.abs(
    df["error"]
)


# ============================================================
# Define RUL groups
# ============================================================

def rul_group(rul):

    if rul <= 7:
        return "1-7 days"

    elif rul <= 14:
        return "8-14 days"

    elif rul <= 30:
        return "15-30 days"

    elif rul <= 60:
        return "31-60 days"

    else:
        return ">60 days"


df["rul_group"] = df[
    "actual_rul"
].apply(rul_group)


# ============================================================
# Analyse each group
# ============================================================

results = []


groups = [
    "1-7 days",
    "8-14 days",
    "15-30 days",
    "31-60 days",
    ">60 days"
]


for group in groups:

    subset = df[
        df["rul_group"] == group
    ]

    if len(subset) == 0:

        results.append({
            "RUL_group": group,
            "samples": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "Mean_Error": np.nan,
            "Median_Error": np.nan,
            "Mean_Actual_RUL": np.nan,
            "Mean_Predicted_RUL": np.nan
        })

        continue


    mae = np.mean(
        np.abs(
            subset["actual_rul"] -
            subset["predicted_rul"]
        )
    )

    rmse = np.sqrt(
        np.mean(
            (
                subset["actual_rul"] -
                subset["predicted_rul"]
            ) ** 2
        )
    )

    mean_error = subset[
        "error"
    ].mean()

    median_error = subset[
        "error"
    ].median()

    mean_actual = subset[
        "actual_rul"
    ].mean()

    mean_predicted = subset[
        "predicted_rul"
    ].mean()


    results.append({
        "RUL_group": group,
        "samples": len(subset),
        "MAE": mae,
        "RMSE": rmse,
        "Mean_Error": mean_error,
        "Median_Error": median_error,
        "Mean_Actual_RUL": mean_actual,
        "Mean_Predicted_RUL": mean_predicted
    })


# ============================================================
# Create summary
# ============================================================

summary = pd.DataFrame(results)


# ============================================================
# Print results
# ============================================================

print("\n" + "=" * 80)
print("30-STEP LSTM — RUL ERROR ANALYSIS")
print("=" * 80)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)


# ============================================================
# Overall metrics
# ============================================================

overall_mae = df[
    "absolute_error"
].mean()

overall_rmse = np.sqrt(
    np.mean(
        df["error"] ** 2
    )
)

overall_mean_error = df[
    "error"
].mean()

overall_median_error = df[
    "error"
].median()


print("\n" + "=" * 80)
print("OVERALL")
print("=" * 80)

print(
    f"MAE: {overall_mae:.3f} days"
)

print(
    f"RMSE: {overall_rmse:.3f} days"
)

print(
    f"Mean Error: {overall_mean_error:.3f} days"
)

print(
    f"Median Error: {overall_median_error:.3f} days"
)


# ============================================================
# Prediction distribution
# ============================================================

print("\n" + "=" * 80)
print("PREDICTION DISTRIBUTION")
print("=" * 80)

print(
    df["predicted_rul"].describe()
)

print("\nActual RUL distribution:")

print(
    df["actual_rul"].describe()
)


# ============================================================
# Save results
# ============================================================

output_file = os.path.join(
    BASE,
    "seq_30",
    "lstm",
    "rul_error_by_range.csv"
)

summary.to_csv(
    output_file,
    index=False
)

print(
    f"\nSaved: {output_file}"
)