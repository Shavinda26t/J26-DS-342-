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
)

from xgboost import XGBClassifier


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

HORIZON = 7

SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
    / "ST12000NM0008_Q3_Q4"
    / f"{HORIZON}d"
)

FEATURE_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "feature_leakage_audit"
    / "model_feature_manifest.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "reference_model"
    / f"{HORIZON}d"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "reference_model"
    / f"{HORIZON}d"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# LOAD FEATURE MANIFEST
# ==========================================================

manifest = pd.read_csv(
    FEATURE_MANIFEST
)

FEATURES = (
    manifest.loc[
        manifest["approved_for_reference_model"] == True,
        "feature"
    ]
    .tolist()
)

TARGET = (
    f"failure_within_{HORIZON}d"
)


print(
    "Approved features:",
    len(FEATURES)
)

print(
    "Target:",
    TARGET
)


# ==========================================================
# LOAD SPLITS
# ==========================================================

def load_split(name):

    file = (
        SPLIT_DIR
        / f"{name}.parquet"
    )

    print(
        f"Loading {name}:",
        file
    )

    df = pd.read_parquet(
        file,
        columns=FEATURES + [TARGET]
    )

    X = df[
        FEATURES
    ].astype(
        "float32"
    )

    y = (
        df[TARGET]
        .astype("int8")
    )

    return X, y


X_train, y_train = load_split(
    "train"
)

X_val, y_val = load_split(
    "validation"
)

X_test, y_test = load_split(
    "test"
)


print("\nDataset sizes:")

print(
    "Train:",
    X_train.shape
)

print(
    "Validation:",
    X_val.shape
)

print(
    "Test:",
    X_test.shape
)


# ==========================================================
# CLASS IMBALANCE
# ==========================================================

positive_train = int(
    (y_train == 1).sum()
)

negative_train = int(
    (y_train == 0).sum()
)

scale_pos_weight = (
    negative_train
    / positive_train
)


print(
    "\nTrain positives:",
    positive_train
)

print(
    "Train negatives:",
    negative_train
)

print(
    "scale_pos_weight:",
    scale_pos_weight
)


# ==========================================================
# MODEL
# ==========================================================

model = XGBClassifier(

    objective="binary:logistic",

    eval_metric="aucpr",

    n_estimators=1000,

    learning_rate=0.05,

    max_depth=5,

    min_child_weight=5,

    subsample=0.8,

    colsample_bytree=0.8,

    reg_alpha=0.1,

    reg_lambda=1.0,

    scale_pos_weight=scale_pos_weight,

    tree_method="hist",

    random_state=42,

    n_jobs=-1,
)


# ==========================================================
# TRAIN
# ==========================================================

print("\nTraining XGBoost...")

start_time = time.time()


model.fit(

    X_train,
    y_train,

    eval_set=[
        (X_train, y_train),
        (X_val, y_val)
    ],

    verbose=50,
)


training_seconds = (
    time.time()
    - start_time
)


print(
    "\nTraining completed in",
    round(
        training_seconds,
        2
    ),
    "seconds"
)


# ==========================================================
# VALIDATION PROBABILITIES
# ==========================================================

val_probability = (
    model.predict_proba(
        X_val
    )[:, 1]
)


# ==========================================================
# CHOOSE THRESHOLD USING VALIDATION ONLY
# ==========================================================

precision_values, recall_values, thresholds = (
    precision_recall_curve(
        y_val,
        val_probability
    )
)


f1_values = (
    2
    * precision_values[:-1]
    * recall_values[:-1]
    /
    (
        precision_values[:-1]
        + recall_values[:-1]
        + 1e-12
    )
)


best_index = int(
    np.nanargmax(
        f1_values
    )
)


best_threshold = float(
    thresholds[
        best_index
    ]
)


print(
    "\nBest validation threshold:",
    best_threshold
)

print(
    "Best validation F1:",
    f1_values[
        best_index
    ]
)


# ==========================================================
# EVALUATION FUNCTION
# ==========================================================

def evaluate(
    split_name,
    X,
    y,
    threshold
):

    probabilities = (
        model.predict_proba(
            X
        )[:, 1]
    )

    predictions = (
        probabilities
        >= threshold
    ).astype(
        "int8"
    )


    pr_auc = (
        average_precision_score(
            y,
            probabilities
        )
    )


    roc_auc = (
        roc_auc_score(
            y,
            probabilities
        )
    )


    precision = (
        precision_score(
            y,
            predictions,
            zero_division=0
        )
    )


    recall = (
        recall_score(
            y,
            predictions,
            zero_division=0
        )
    )


    f1 = (
        f1_score(
            y,
            predictions,
            zero_division=0
        )
    )


    tn, fp, fn, tp = (
        confusion_matrix(
            y,
            predictions,
            labels=[0, 1]
        )
        .ravel()
    )


    false_alarm_rate = (
        fp
        / (
            fp + tn
        )
        if (
            fp + tn
        ) > 0
        else 0
    )


    result = {

        "split":
            split_name,

        "threshold":
            threshold,

        "rows":
            len(y),

        "positives":
            int(
                (y == 1).sum()
            ),

        "negatives":
            int(
                (y == 0).sum()
            ),

        "pr_auc":
            pr_auc,

        "roc_auc":
            roc_auc,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "false_alarm_rate":
            false_alarm_rate,

        "true_positive":
            int(tp),

        "false_positive":
            int(fp),

        "true_negative":
            int(tn),

        "false_negative":
            int(fn),
    }


    prediction_df = pd.DataFrame(
        {
            "actual":
                y.to_numpy(),

            "probability":
                probabilities,

            "prediction":
                predictions,
        }
    )


    prediction_df.to_parquet(
        RESULT_DIR
        / f"{split_name}_predictions.parquet",
        index=False
    )


    return result


# ==========================================================
# EVALUATE VALIDATION AND TEST
# ==========================================================

validation_result = evaluate(
    "validation",
    X_val,
    y_val,
    best_threshold
)

test_result = evaluate(
    "test",
    X_test,
    y_test,
    best_threshold
)


results_df = pd.DataFrame(
    [
        validation_result,
        test_result
    ]
)


# ==========================================================
# SAVE RESULTS
# ==========================================================

results_file = (
    RESULT_DIR
    / "metrics.csv"
)

results_df.to_csv(
    results_file,
    index=False
)


# ==========================================================
# SAVE FEATURE IMPORTANCE
# ==========================================================

importance_df = pd.DataFrame(
    {
        "feature":
            FEATURES,

        "importance":
            model.feature_importances_
    }
)

importance_df = (
    importance_df
    .sort_values(
        "importance",
        ascending=False
    )
)


importance_file = (
    RESULT_DIR
    / "xgboost_feature_importance.csv"
)

importance_df.to_csv(
    importance_file,
    index=False
)


# ==========================================================
# SAVE MODEL
# ==========================================================

model_file = (
    MODEL_DIR
    / "xgboost_reference.joblib"
)

joblib.dump(
    model,
    model_file
)


model.save_model(
    MODEL_DIR
    / "xgboost_reference.json"
)


# ==========================================================
# SAVE METADATA
# ==========================================================

metadata = {

    "horizon_days":
        HORIZON,

    "target":
        TARGET,

    "number_of_features":
        len(FEATURES),

    "training_rows":
        len(y_train),

    "training_positive_rows":
        positive_train,

    "training_negative_rows":
        negative_train,

    "scale_pos_weight":
        scale_pos_weight,

    "selected_threshold":
        best_threshold,

    "training_seconds":
        training_seconds,

    "random_state":
        42,
}


with open(
    MODEL_DIR
    / "metadata.json",
    "w"
) as file:

    json.dump(
        metadata,
        file,
        indent=4
    )


# ==========================================================
# PRINT RESULTS
# ==========================================================

print("\n")
print("=" * 70)
print(
    f"{HORIZON}-DAY XGBOOST REFERENCE MODEL COMPLETE"
)
print("=" * 70)


print(
    "\nMetrics:\n"
)

print(
    results_df.to_string(
        index=False
    )
)


print(
    "\nTop 20 XGBoost feature importances:\n"
)

print(
    importance_df
    .head(20)
    .to_string(
        index=False
    )
)


print(
    "\nModel saved to:"
)

print(
    model_file
)


print(
    "\nResults saved to:"
)

print(
    RESULT_DIR
)