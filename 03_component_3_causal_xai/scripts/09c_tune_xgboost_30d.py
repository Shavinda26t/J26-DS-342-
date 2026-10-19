from pathlib import Path
import json
import time
import logging

import joblib
import numpy as np
import pandas as pd

import optuna
from optuna.samplers import TPESampler

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    precision_recall_curve,
)

from xgboost import XGBClassifier


logging.basicConfig(level=logging.WARNING)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ID = "ST12000NM0008_Q3_Q4"

HORIZON = 30

TARGET = f"failure_within_{HORIZON}d"

FEATURE_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "feature_leakage_audit"
    / "model_feature_manifest.csv"
)

SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
    / DATASET_ID
    / f"{HORIZON}d"
)

BASELINE_MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "reference_model"
    / f"{HORIZON}d"
)

BASELINE_RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "reference_model"
    / f"{HORIZON}d"
)

TUNED_MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "tuned_reference_model"
    / f"{HORIZON}d"
)

TUNED_RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "tuned_reference_model"
    / f"{HORIZON}d"
)

TUNED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
TUNED_RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# TUNING CONFIG
# ==========================================================

N_TRIALS = 40

N_ESTIMATORS = 2500

EARLY_STOPPING_ROUNDS = 50

FIXED_SCALE_POS_WEIGHT = 597.9631384939441

RANDOM_STATE = 42


# ==========================================================
# LOAD FEATURES + VALIDATE AGAINST EXISTING MODEL
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

assert len(FEATURES) == 176, (
    f"Expected 176 approved features, got {len(FEATURES)}"
)

baseline_model = joblib.load(
    BASELINE_MODEL_DIR / "xgboost_reference.joblib"
)

baseline_feature_names = list(
    getattr(baseline_model, "feature_names_in_", [])
)

if baseline_feature_names:
    assert FEATURES == baseline_feature_names, (
        "Feature order does not match the existing reference model."
    )
    print(
        "Feature order validated against existing reference model: OK"
    )
else:
    print(
        "WARNING: existing model has no stored feature names; "
        "validating against manifest order only."
    )


# ==========================================================
# LOAD SPLITS (TEST REMAINS UNTOUCHED UNTIL FINAL EVAL)
# ==========================================================


def load_split(split_name):

    df = pd.read_parquet(
        SPLIT_DIR / f"{split_name}.parquet",
        columns=FEATURES + [TARGET],
    )

    return (
        df[FEATURES].astype("float32"),
        df[TARGET].astype("int8"),
    )


X_train, y_train = load_split("train")
X_val, y_val = load_split("validation")

print(
    "Train / Validation:",
    len(X_train),
    len(X_val),
)

positive_train = int((y_train == 1).sum())
negative_train = int((y_train == 0).sum())

computed_spw = negative_train / positive_train

assert abs(computed_spw - FIXED_SCALE_POS_WEIGHT) < 1e-6, (
    f"Computed scale_pos_weight {computed_spw} does not match "
    f"existing {FIXED_SCALE_POS_WEIGHT}"
)

print(
    "scale_pos_weight fixed at:",
    FIXED_SCALE_POS_WEIGHT,
    "(validated against train split)",
)


# ==========================================================
# HELPERS
# ==========================================================


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


def full_metrics(y_true, y_prob, threshold):

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    predictions = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = (
        confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    )

    positives = int((y_true == 1).sum())

    return {
        "threshold": threshold,
        "rows": len(y_true),
        "positives": positives,
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "false_alarm_rate": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
    }


def build_model(params):

    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=N_ESTIMATORS,
        scale_pos_weight=FIXED_SCALE_POS_WEIGHT,
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        **params,
    )


# ==========================================================
# OPTUNA OBJECTIVE (VALIDATION PR-AUC ONLY)
# ==========================================================


def objective(trial):

    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.10, log=True
        ),
        "min_child_weight": trial.suggest_float(
            "min_child_weight", 1.0, 20.0, log=True
        ),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.6, 1.0
        ),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda": trial.suggest_float(
            "reg_lambda", 0.5, 10.0, log=True
        ),
    }

    model = build_model(params)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    val_probability = model.predict_proba(X_val)[:, 1]

    val_pr_auc = float(
        average_precision_score(y_val, val_probability)
    )

    trial.set_user_attr(
        "best_iteration",
        int(getattr(model, "best_iteration", -1)) + 1,
    )

    return val_pr_auc


# ==========================================================
# RUN TUNING
# ==========================================================

print(f"\nStarting Optuna study: {N_TRIALS} trials")
print(
    f"(n_estimators={N_ESTIMATORS} upper bound, "
    f"early stopping={EARLY_STOPPING_ROUNDS})\n"
)

study = optuna.create_study(
    direction="maximize",
    sampler=TPESampler(seed=RANDOM_STATE),
    study_name="xgb_30d_reference_tuning",
)

study_start = time.time()

study.optimize(
    objective,
    n_trials=N_TRIALS,
    gc_after_trial=True,
)

study_seconds = time.time() - study_start

best_trial = study.best_trial

print(f"\nStudy completed in {study_seconds:.1f}s")
print(
    f"Best trial #{best_trial.number}: "
    f"val PR-AUC={best_trial.value:.4f}"
)

for key, value in best_trial.params.items():
    print(f"  {key}: {value}")

print(
    "  best boosting rounds:",
    best_trial.user_attrs.get("best_iteration"),
)


# ==========================================================
# REFIT BEST CONFIG ON TRAIN (VALIDATION = EARLY STOPPING)
# ==========================================================

best_params = dict(best_trial.params)

best_model = build_model(best_params)

fit_start = time.time()

best_model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=50,
)

fit_seconds = time.time() - fit_start

best_iteration = int(getattr(best_model, "best_iteration", -1)) + 1

val_probability = best_model.predict_proba(X_val)[:, 1]

threshold = best_f1_threshold(y_val, val_probability)

print(
    f"\nRefit complete in {fit_seconds:.1f}s, "
    f"trees used: {best_iteration}"
)
print("Validation best-F1 threshold:", threshold)


# ==========================================================
# FINAL TEST EVALUATION (ONCE, AFTER SELECTION)
# ==========================================================

print("\nLoading untouched test set for final evaluation...")

X_test, y_test = load_split("test")

test_probability = best_model.predict_proba(X_test)[:, 1]

tuned_val_metrics = full_metrics(
    y_val, val_probability, threshold
)
tuned_val_metrics["split"] = "validation"

tuned_test_metrics = full_metrics(
    y_test, test_probability, threshold
)
tuned_test_metrics["split"] = "test"

print(
    "\nTuned validation PR-AUC:",
    round(tuned_val_metrics["pr_auc"], 4),
)
print(
    "Tuned test PR-AUC:",
    round(tuned_test_metrics["pr_auc"], 4),
)
print(
    "Tuned test ROC-AUC:",
    round(tuned_test_metrics["roc_auc"], 4),
)


# ==========================================================
# BASELINE (EXISTING REFERENCE MODEL) FROM REPORTED METRICS
# ==========================================================

baseline_metrics = pd.read_csv(
    BASELINE_RESULT_DIR / "metrics.csv"
)

baseline_val = baseline_metrics[
    baseline_metrics["split"] == "validation"
].iloc[0]

baseline_test = baseline_metrics[
    baseline_metrics["split"] == "test"
].iloc[0]


# ==========================================================
# MODEL COMPARISON
# ==========================================================

comparison_rows = [
    {
        "metric": "pr_auc",
        "primary": True,
        "baseline_validation": float(baseline_val["pr_auc"]),
        "tuned_validation": tuned_val_metrics["pr_auc"],
        "baseline_test": float(baseline_test["pr_auc"]),
        "tuned_test": tuned_test_metrics["pr_auc"],
    },
    {
        "metric": "roc_auc",
        "primary": False,
        "baseline_validation": float(baseline_val["roc_auc"]),
        "tuned_validation": tuned_val_metrics["roc_auc"],
        "baseline_test": float(baseline_test["roc_auc"]),
        "tuned_test": tuned_test_metrics["roc_auc"],
    },
    {
        "metric": "precision",
        "primary": False,
        "baseline_validation": float(baseline_val["precision"]),
        "tuned_validation": tuned_val_metrics["precision"],
        "baseline_test": float(baseline_test["precision"]),
        "tuned_test": tuned_test_metrics["precision"],
    },
    {
        "metric": "recall",
        "primary": False,
        "baseline_validation": float(baseline_val["recall"]),
        "tuned_validation": tuned_val_metrics["recall"],
        "baseline_test": float(baseline_test["recall"]),
        "tuned_test": tuned_test_metrics["recall"],
    },
    {
        "metric": "f1",
        "primary": False,
        "baseline_validation": float(baseline_val["f1"]),
        "tuned_validation": tuned_val_metrics["f1"],
        "baseline_test": float(baseline_test["f1"]),
        "tuned_test": tuned_test_metrics["f1"],
    },
    {
        "metric": "false_alarm_rate",
        "primary": False,
        "baseline_validation": float(
            baseline_val["false_alarm_rate"]
        ),
        "tuned_validation": tuned_val_metrics["false_alarm_rate"],
        "baseline_test": float(baseline_test["false_alarm_rate"]),
        "tuned_test": tuned_test_metrics["false_alarm_rate"],
    },
    {
        "metric": "threshold",
        "primary": False,
        "baseline_validation": float(baseline_val["threshold"]),
        "tuned_validation": tuned_val_metrics["threshold"],
        "baseline_test": float(baseline_val["threshold"]),
        "tuned_test": tuned_test_metrics["threshold"],
    },
]

comparison_df = pd.DataFrame(comparison_rows)

comparison_df["test_delta"] = (
    comparison_df["tuned_test"] - comparison_df["baseline_test"]
)

comparison_file = (
    TUNED_RESULT_DIR / "model_comparison.csv"
)

comparison_df.to_csv(comparison_file, index=False)


def pct_change(metric_name):

    baseline_value = float(
        comparison_df.loc[
            comparison_df["metric"] == metric_name,
            "baseline_test",
        ].iloc[0]
    )

    tuned_value = float(
        comparison_df.loc[
            comparison_df["metric"] == metric_name,
            "tuned_test",
        ].iloc[0]
    )

    if baseline_value == 0:
        return np.nan

    return (tuned_value - baseline_value) / baseline_value * 100


pr_auc_pct = pct_change("pr_auc")


# ==========================================================
# SAVE TUNED ARTIFACTS
# ==========================================================

best_model.save_model(
    TUNED_MODEL_DIR / "xgboost_tuned.json"
)

joblib.dump(
    best_model,
    TUNED_MODEL_DIR / "xgboost_tuned.joblib",
)

with open(
    TUNED_MODEL_DIR / "best_params.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        {
            "best_trial_number": best_trial.number,
            "best_validation_pr_auc": best_trial.value,
            "params": best_params,
            "best_boosting_rounds": best_iteration,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "n_estimators_upper_bound": N_ESTIMATORS,
            "scale_pos_weight": FIXED_SCALE_POS_WEIGHT,
        },
        f,
        indent=2,
    )

tuned_metadata = {
    "horizon_days": HORIZON,
    "target": TARGET,
    "number_of_features": len(FEATURES),
    "scale_pos_weight": FIXED_SCALE_POS_WEIGHT,
    "selected_threshold": threshold,
    "best_boosting_rounds": best_iteration,
    "best_trial_number": best_trial.number,
    "best_validation_pr_auc": best_trial.value,
    "n_trials": N_TRIALS,
    "tuning_objective": "validation PR-AUC (test untouched)",
    "threshold_rule": "validation best-F1 (same as 09B)",
    "training_seconds": fit_seconds,
    "tuning_seconds": study_seconds,
    "random_state": RANDOM_STATE,
}

with open(
    TUNED_MODEL_DIR / "metadata.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(tuned_metadata, f, indent=2)

trials_df = study.trials_dataframe()

trials_file = TUNED_RESULT_DIR / "optuna_trials.csv"

trials_df.to_csv(trials_file, index=False)

tuned_results_df = pd.DataFrame(
    [tuned_val_metrics, tuned_test_metrics]
)

tuned_results_df.to_csv(
    TUNED_RESULT_DIR / "metrics.csv",
    index=False,
)

tuning_summary = {
    "horizon_days": HORIZON,
    "n_trials": N_TRIALS,
    "study_seconds": study_seconds,
    "best_trial_number": best_trial.number,
    "best_validation_pr_auc": best_trial.value,
    "best_params": best_params,
    "best_boosting_rounds": best_iteration,
    "tuned_threshold": threshold,
    "baseline_test_pr_auc": float(baseline_test["pr_auc"]),
    "tuned_test_pr_auc": tuned_test_metrics["pr_auc"],
    "pr_auc_absolute_improvement": tuned_test_metrics["pr_auc"]
    - float(baseline_test["pr_auc"]),
    "pr_auc_percent_improvement": pr_auc_pct,
    "recall_baseline_test": float(baseline_test["recall"]),
    "recall_tuned_test": tuned_test_metrics["recall"],
    "f1_baseline_test": float(baseline_test["f1"]),
    "f1_tuned_test": tuned_test_metrics["f1"],
    "false_alarm_rate_baseline_test": float(
        baseline_test["false_alarm_rate"]
    ),
    "false_alarm_rate_tuned_test": tuned_test_metrics[
        "false_alarm_rate"
    ],
    "promotion_decision": "NOT_PROMOTED - awaiting manual review",
    "note": (
        "Accuracy is reported for completeness only; the dataset "
        "is highly imbalanced (test prevalence ~0.16%), so "
        "accuracy is not a meaningful primary metric."
    ),
}

with open(
    TUNED_RESULT_DIR / "tuning_summary.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(tuning_summary, f, indent=2)


# ==========================================================
# FINAL REPORT
# ==========================================================

print(f"\n{'=' * 64}")
print("TUNING COMPLETE - COMPARISON (TEST SET)")
print(f"{'=' * 64}")

display = comparison_df[
    [
        "metric",
        "baseline_test",
        "tuned_test",
        "test_delta",
    ]
].copy()

for column in ["baseline_test", "tuned_test", "test_delta"]:
    display[column] = display[column].map(
        lambda value: f"{value:+.6f}"
        if abs(value) < 1
        else f"{value:.6f}"
    )

print(display.to_string(index=False))

print(
    f"\n1. Current model test PR-AUC : "
    f"{float(baseline_test['pr_auc']):.6f}"
)
print(
    f"2. Tuned model test PR-AUC   : "
    f"{tuned_test_metrics['pr_auc']:.6f}"
)
print(
    f"3. PR-AUC improvement        : "
    f"{tuned_test_metrics['pr_auc'] - float(baseline_test['pr_auc']):+.6f} "
    f"({pr_auc_pct:+.2f}%)"
)
print(
    f"4. Recall improvement        : "
    f"{float(baseline_test['recall']):.4f} -> "
    f"{tuned_test_metrics['recall']:.4f}"
)
print(
    f"5. F1 improvement            : "
    f"{float(baseline_test['f1']):.4f} -> "
    f"{tuned_test_metrics['f1']:.4f}"
)
print(
    f"6. False alarm rate          : "
    f"{float(baseline_test['false_alarm_rate']):.6f} -> "
    f"{tuned_test_metrics['false_alarm_rate']:.6f}"
)
print(f"7. Best hyperparameters      : {best_params}")
print(f"8. Best boosting iteration   : {best_iteration}")
print(f"9. New validation threshold  : {threshold}")
print(
    "10. Promotion                : NOT performed - review "
    "comparison above and decide manually."
)

print(f"\nSaved model artifacts to: {TUNED_MODEL_DIR}")
print(f"Saved evaluation outputs to: {TUNED_RESULT_DIR}")
print(
    "\nNOTE: accuracy = "
    f"{tuned_test_metrics['accuracy']:.4f} on test, reported for "
    "completeness only; class imbalance (~0.16% positives) makes "
    "accuracy uninformative."
)
