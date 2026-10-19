from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

HORIZONS = [7, 14, 30]

DATASET_ID = "ST12000NM0008_Q3_Q4"

FEATURE_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "feature_leakage_audit"
    / "model_feature_manifest.csv"
)


# ==========================================================
# LOAD FEATURE MANIFEST
# ==========================================================

manifest = pd.read_csv(FEATURE_MANIFEST)

FEATURES = (
    manifest.loc[
        manifest["approved_for_reference_model"] == True,
        "feature",
    ]
    .tolist()
)

print("Approved features:", len(FEATURES))


# ==========================================================
# METRIC HELPERS
# ==========================================================


def month_metrics(y_true, y_prob, threshold):

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    rows = len(y_true)
    positives = int((y_true == 1).sum())

    result = {
        "rows": rows,
        "positives": positives,
        "prevalence": positives / rows if rows > 0 else np.nan,
        "mean_predicted_probability": float(np.mean(y_prob)) if rows > 0 else np.nan,
    }

    if rows == 0:
        return result

    result["pr_auc"] = float(
        average_precision_score(y_true, y_prob)
    )

    if positives == 0 or positives == rows:
        result["roc_auc"] = np.nan
    else:
        result["roc_auc"] = float(
            roc_auc_score(y_true, y_prob)
        )

    predictions = (y_prob >= threshold).astype(int)

    result["precision"] = float(
        precision_score(y_true, predictions, zero_division=0)
    )

    result["recall"] = float(
        recall_score(y_true, predictions, zero_division=0)
    )

    result["f1"] = float(
        f1_score(y_true, predictions, zero_division=0)
    )

    tn, fp, fn, tp = (
        confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    )

    result["false_alarm_rate"] = (
        fp / (fp + tn) if (fp + tn) > 0 else 0.0
    )

    result["true_positive"] = tp
    result["false_positive"] = fp
    result["true_negative"] = tn
    result["false_negative"] = fn

    return result


# ==========================================================
# MAIN LOOP OVER HORIZONS
# ==========================================================

SPLITS = ["train", "validation", "test"]

summary_rows = []

for horizon in HORIZONS:

    target = f"failure_within_{horizon}d"

    model_file = (
        PROJECT_ROOT
        / "models"
        / "reference_model"
        / f"{horizon}d"
        / "xgboost_reference.joblib"
    )

    result_dir = (
        PROJECT_ROOT
        / "results"
        / "reference_model"
        / f"{horizon}d"
    )

    metrics_file = result_dir / "metrics.csv"

    print(f"\n{'=' * 60}")
    print(f"Horizon: {horizon}d")
    print(f"{'=' * 60}")

    # ------------------------------------------------------
    # LOAD THRESHOLD FROM ORIGINAL TRAINING RUN (VALIDATION)
    # ------------------------------------------------------

    original_metrics = pd.read_csv(metrics_file)

    validation_row = original_metrics[
        original_metrics["split"] == "validation"
    ].iloc[0]

    threshold = float(validation_row["threshold"])

    test_row = original_metrics[
        original_metrics["split"] == "test"
    ].iloc[0]

    print("Validation threshold:", round(threshold, 6))

    print(
        "Reported test PR-AUC:",
        round(float(test_row["pr_auc"]), 4),
    )

    # ------------------------------------------------------
    # SCORE EVERY SPLIT, GROUPED BY MONTH
    # ------------------------------------------------------

    model = joblib.load(model_file)

    all_monthly_rows = []

    for split_name in SPLITS:

        split_file = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "splits"
            / DATASET_ID
            / f"{horizon}d"
            / f"{split_name}.parquet"
        )

        print(f"\n--- Split: {split_name} ---")

        print("Loading:", split_file.name)

        df = pd.read_parquet(
            split_file,
            columns=FEATURES + [target, "date"],
        )

        print("Rows:", len(df))

        df["date"] = pd.to_datetime(df["date"])

        X = df[FEATURES].astype("float32")

        y = df[target].astype("int8")

        probability = model.predict_proba(X)[:, 1]

        df = df[["date"]].copy()

        df["actual"] = y.to_numpy()

        df["probability"] = probability

        df["month"] = df["date"].dt.to_period("M").astype(str)

        pooled = month_metrics(
            df["actual"].to_numpy(),
            df["probability"].to_numpy(),
            threshold,
        )

        pooled["month"] = "ALL"

        pooled["split"] = split_name

        if split_name == "test":
            print(
                "Recomputed pooled test PR-AUC:",
                round(pooled["pr_auc"], 4),
            )

        monthly_rows = [pooled]

        for month, group in df.groupby("month"):

            metrics = month_metrics(
                group["actual"].to_numpy(),
                group["probability"].to_numpy(),
                threshold,
            )

            metrics["month"] = month

            metrics["split"] = split_name

            monthly_rows.append(metrics)

        all_monthly_rows.extend(monthly_rows)

        # --------------------------------------------------
        # CONSOLE TABLE PER SPLIT
        # --------------------------------------------------

        display = pd.DataFrame(monthly_rows)

        for column in [
            "prevalence",
            "mean_predicted_probability",
            "pr_auc",
            "roc_auc",
            "precision",
            "recall",
            "f1",
            "false_alarm_rate",
        ]:
            display[column] = display[column].map(
                lambda value: f"{value:.4f}" if pd.notna(value) else "NA"
            )

        print()
        print(display.to_string(index=False))

    # ------------------------------------------------------
    # SAVE COMBINED MONTHLY METRICS
    # ------------------------------------------------------

    monthly_df = pd.DataFrame(all_monthly_rows)

    ordered_columns = [
        "split",
        "month",
        "rows",
        "positives",
        "prevalence",
        "mean_predicted_probability",
        "pr_auc",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "false_alarm_rate",
        "true_positive",
        "false_positive",
        "true_negative",
        "false_negative",
    ]

    monthly_df = monthly_df[ordered_columns]

    output_file = result_dir / "metrics_by_month.csv"

    monthly_df.to_csv(output_file, index=False)

    print("\nSaved:", output_file)

    # ------------------------------------------------------
    # PER-SPLIT TREND AND SUMMARY ROWS
    # ------------------------------------------------------

    for split_name in ["validation", "test"]:

        month_only = monthly_df[
            (monthly_df["split"] == split_name)
            & (monthly_df["month"] != "ALL")
        ]

        if len(month_only) == 0:
            continue

        split_pooled = monthly_df[
            (monthly_df["split"] == split_name)
            & (monthly_df["month"] == "ALL")
        ].iloc[0]

        reported_pr_auc = float(
            original_metrics.loc[
                original_metrics["split"] == split_name,
                "pr_auc",
            ].iloc[0]
        )

        summary = {
            "horizon_days": horizon,
            "split": split_name,
            "threshold": threshold,
            "pooled_pr_auc": split_pooled["pr_auc"],
            "reported_pr_auc": reported_pr_auc,
            "prevalence": split_pooled["prevalence"],
            "mean_predicted_probability": split_pooled[
                "mean_predicted_probability"
            ],
            "months": len(month_only),
            "worst_month_pr_auc": float(month_only["pr_auc"].min()),
            "worst_month": month_only.loc[
                month_only["pr_auc"].idxmin(), "month",
            ],
            "best_month_pr_auc": float(month_only["pr_auc"].max()),
            "best_month": month_only.loc[
                month_only["pr_auc"].idxmax(), "month",
            ],
        }

        if month_only["pr_auc"].notna().sum() >= 2:

            month_positions = np.arange(len(month_only))

            trend = np.polyfit(
                month_positions,
                month_only["pr_auc"]
                .fillna(month_only["pr_auc"].mean())
                .to_numpy(),
                1,
            )[0]

            print(
                f"\n{split_name} PR-AUC monthly slope: {trend:+.5f}"
            )

            summary["pr_auc_monthly_slope"] = float(trend)

        summary_rows.append(summary)


# ==========================================================
# CROSS-HORIZON SUMMARY
# ==========================================================

if summary_rows:

    summary_df = pd.DataFrame(summary_rows)

    print(f"\n{'=' * 60}")
    print("CROSS-HORIZON SUMMARY")
    print(f"{'=' * 60}")

    print(summary_df.to_string(index=False))

    summary_file = (
        PROJECT_ROOT
        / "results"
        / "reference_model"
        / "metrics_by_month_summary.csv"
    )

    summary_df.to_csv(summary_file, index=False)

    print("\nSaved:", summary_file)
