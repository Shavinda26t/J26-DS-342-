from pathlib import Path
import json
import time

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
    precision_recall_curve,
    brier_score_loss,
)

from xgboost import XGBClassifier


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

BASELINE_RESULTS = (
    PROJECT_ROOT
    / "results"
    / "reference_model"
)

RETRAIN_MODEL_ROOT = (
    PROJECT_ROOT
    / "models"
    / "reference_model_retrain"
)

RETRAIN_RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "reference_model_retrain"
)


# ==========================================================
# CONFIGS
#
# Same family as 09B, plus early stopping on validation
# AUCPR. Two capacity variants; best picked per horizon
# by validation PR-AUC only (test stays untouched for
# final reporting, matching original methodology).
# ==========================================================

EARLY_STOPPING_ROUNDS = 50

CONFIGS = {

    "original_es": {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 5,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    },

    "shallow_es": {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 3,
        "min_child_weight": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    },
}


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
# HELPERS
# ==========================================================


def pooled_metrics(y_true, y_prob, threshold):

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    positives = int((y_true == 1).sum())

    predictions = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = (
        confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    )

    return {
        "rows": len(y_true),
        "positives": positives,
        "prevalence": positives / len(y_true),
        "mean_predicted_probability": float(np.mean(y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if 0 < positives < len(y_true) else np.nan,
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "false_alarm_rate": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
    }


def best_f1_threshold(y_true, y_prob):

    precision_values, recall_values, thresholds = (
        precision_recall_curve(y_true, y_prob)
    )

    f1_values = (
        2
        * precision_values[:-1]
        * recall_values[:-1]
        / (precision_values[:-1] + recall_values[:-1] + 1e-12)
    )

    return float(thresholds[int(np.nanargmax(f1_values))])


# ==========================================================
# MAIN
# ==========================================================

summary_rows = []

for horizon in HORIZONS:

    target = f"failure_within_{horizon}d"

    split_dir = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "splits"
        / DATASET_ID
        / f"{horizon}d"
    )

    model_dir = RETRAIN_MODEL_ROOT / f"{horizon}d"
    result_dir = RETRAIN_RESULT_ROOT / f"{horizon}d"

    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Horizon: {horizon}d")
    print(f"{'=' * 60}")

    # ------------------------------------------------------
    # BASELINE FROM ORIGINAL 09B RUN
    # ------------------------------------------------------

    baseline_metrics = pd.read_csv(
        BASELINE_RESULTS / f"{horizon}d" / "metrics.csv"
    )

    baseline_val = baseline_metrics[
        baseline_metrics["split"] == "validation"
    ].iloc[0]

    baseline_test = baseline_metrics[
        baseline_metrics["split"] == "test"
    ].iloc[0]

    print(
        "Baseline val PR-AUC:",
        round(float(baseline_val["pr_auc"]), 4),
        "| Baseline test PR-AUC:",
        round(float(baseline_test["pr_auc"]), 4),
    )

    # ------------------------------------------------------
    # LOAD SPLITS
    # ------------------------------------------------------

    def load_split(split_name):

        df = pd.read_parquet(
            split_dir / f"{split_name}.parquet",
            columns=FEATURES + [target, "date"],
        )

        return (
            df[FEATURES].astype("float32"),
            df[target].astype("int8"),
            pd.to_datetime(df["date"]),
        )

    X_train, y_train, _ = load_split("train")
    X_val, y_val, val_dates = load_split("validation")
    X_test, y_test, test_dates = load_split("test")

    print(
        "Train / Val / Test:",
        len(X_train),
        len(X_val),
        len(X_test),
    )

    positive_train = int((y_train == 1).sum())
    negative_train = int((y_train == 0).sum())
    scale_pos_weight = negative_train / positive_train

    # ------------------------------------------------------
    # TRAIN EVERY CONFIG
    # ------------------------------------------------------

    config_rows = []

    fitted = {}

    for config_name, params in CONFIGS.items():

        print(f"\n--- Config: {config_name} ---")

        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=scale_pos_weight,
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            **params,
        )

        start_time = time.time()

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=100,
        )

        train_seconds = time.time() - start_time

        val_probability = model.predict_proba(X_val)[:, 1]

        val_pr_auc = float(
            average_precision_score(y_val, val_probability)
        )

        best_iteration = int(getattr(model, "best_iteration", -1)) + 1

        print(
            f"{config_name}: val PR-AUC={val_pr_auc:.4f}, "
            f"trees used={best_iteration}, "
            f"time={train_seconds:.1f}s"
        )

        config_rows.append(
            {
                "config": config_name,
                "val_pr_auc": val_pr_auc,
                "trees_used": best_iteration,
                "train_seconds": round(train_seconds, 1),
            }
        )

        fitted[config_name] = (model, val_pr_auc)

    # ------------------------------------------------------
    # PICK BEST CONFIG BY VALIDATION PR-AUC
    # ------------------------------------------------------

    best_name, (model, best_val_pr_auc) = max(
        fitted.items(),
        key=lambda item: item[1][1],
    )

    print(
        f"\nSelected config: {best_name} "
        f"(val PR-AUC {best_val_pr_auc:.4f})"
    )

    # ------------------------------------------------------
    # THRESHOLD FROM VALIDATION (ORIGINAL METHODOLOGY)
    # ------------------------------------------------------

    val_probability = model.predict_proba(X_val)[:, 1]
    test_probability = model.predict_proba(X_test)[:, 1]

    threshold = best_f1_threshold(y_val, val_probability)

    print("Validation best-F1 threshold:", round(threshold, 6))

    # ------------------------------------------------------
    # POOLED METRICS
    # ------------------------------------------------------

    val_result = pooled_metrics(y_val, val_probability, threshold)
    val_result["split"] = "validation"

    test_result = pooled_metrics(y_test, test_probability, threshold)
    test_result["split"] = "test"

    print(
        "Retrained val PR-AUC:",
        round(val_result["pr_auc"], 4),
        "| Retrained test PR-AUC:",
        round(test_result["pr_auc"], 4),
    )
    print(
        "Retrained val ROC-AUC:",
        round(val_result["roc_auc"], 4),
        "| Retrained test ROC-AUC:",
        round(test_result["roc_auc"], 4),
    )
    print(
        "Test Brier:",
        round(test_result["brier_score"], 5),
        "(baseline not comparable directly)",
    )

    results_df = pd.DataFrame([val_result, test_result])

    results_df.to_csv(
        result_dir / "metrics.csv",
        index=False,
    )

    # ------------------------------------------------------
    # PER-MONTH TEST METRICS
    # ------------------------------------------------------

    monthly_rows = []

    for split_name, dates, y_true, probabilities in [
        ("validation", val_dates, y_val, val_probability),
        ("test", test_dates, y_test, test_probability),
    ]:

        frame = pd.DataFrame(
            {
                "date": dates.to_numpy(),
                "actual": y_true.to_numpy(),
                "probability": probabilities,
            }
        )

        frame["month"] = (
            pd.to_datetime(frame["date"])
            .dt.to_period("M")
            .astype(str)
        )

        for month, group in frame.groupby("month"):

            row = pooled_metrics(
                group["actual"].to_numpy(),
                group["probability"].to_numpy(),
                threshold,
            )

            row["split"] = split_name
            row["month"] = month

            monthly_rows.append(row)

    monthly_df = pd.DataFrame(monthly_rows)

    monthly_df.to_csv(
        result_dir / "metrics_by_month.csv",
        index=False,
    )

    # ------------------------------------------------------
    # SAVE MODEL + METADATA
    # ------------------------------------------------------

    joblib.dump(
        model,
        model_dir / "xgboost_reference.joblib",
    )

    model.save_model(
        model_dir / "xgboost_reference.json"
    )

    metadata = {
        "horizon_days": horizon,
        "target": target,
        "config": best_name,
        "config_params": CONFIGS[best_name],
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "trees_used": int(getattr(model, "best_iteration", -1)) + 1,
        "number_of_features": len(FEATURES),
        "scale_pos_weight": scale_pos_weight,
        "validation_threshold": threshold,
        "baseline_val_pr_auc": float(baseline_val["pr_auc"]),
        "baseline_test_pr_auc": float(baseline_test["pr_auc"]),
        "retrained_val_pr_auc": val_result["pr_auc"],
        "retrained_test_pr_auc": test_result["pr_auc"],
        "selection_rule": (
            "best config by validation PR-AUC; "
            "threshold = validation best-F1"
        ),
    }

    with open(
        model_dir / "metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=2)

    config_df = pd.DataFrame(config_rows)

    config_df.to_csv(
        result_dir / "config_comparison.csv",
        index=False,
    )

    summary_rows.append(
        {
            "horizon_days": horizon,
            "selected_config": best_name,
            "trees_used": metadata["trees_used"],
            "baseline_val_pr_auc": float(baseline_val["pr_auc"]),
            "retrained_val_pr_auc": val_result["pr_auc"],
            "baseline_test_pr_auc": float(baseline_test["pr_auc"]),
            "retrained_test_pr_auc": test_result["pr_auc"],
            "baseline_test_roc_auc": float(baseline_test["roc_auc"]),
            "retrained_test_roc_auc": test_result["roc_auc"],
            "retrained_test_precision": test_result["precision"],
            "retrained_test_recall": test_result["recall"],
            "retrained_test_false_alarm_rate": test_result[
                "false_alarm_rate"
            ],
        }
    )


# ==========================================================
# CROSS-HORIZON COMPARISON
# ==========================================================

summary_df = pd.DataFrame(summary_rows)

print(f"\n{'=' * 60}")
print("BASELINE (09B) VS RETRAINED (EARLY STOPPING)")
print(f"{'=' * 60}")

print(summary_df.to_string(index=False))

summary_file = RETRAIN_RESULT_ROOT / "retrain_summary.csv"

summary_df.to_csv(summary_file, index=False)

print("\nSaved:", summary_file)
