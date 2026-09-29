from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

HORIZON = 30

TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
    / "ST12000NM0008_Q3_Q4"
    / f"{HORIZON}d"
    / "test.parquet"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "reference_model"
    / f"{HORIZON}d"
)

MODEL_FILE = (
    MODEL_DIR
    / "xgboost_reference.joblib"
)

METADATA_FILE = (
    MODEL_DIR
    / "metadata.json"
)

FEATURE_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "feature_leakage_audit"
    / "model_feature_manifest.csv"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "shap"
    / f"{HORIZON}d"
    / "local"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "figures"
    / "shap"
    / f"{HORIZON}d"
    / "local"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# SETTINGS
# ==========================================================

TARGET = "failure_within_30d"

TOP_FEATURES = 15


# ==========================================================
# LOAD FEATURE MANIFEST
# ==========================================================

manifest = pd.read_csv(
    FEATURE_MANIFEST
)

FEATURES = (
    manifest.loc[
        manifest[
            "approved_for_reference_model"
        ] == True,
        "feature"
    ]
    .tolist()
)

print(
    "Approved features:",
    len(FEATURES)
)


# ==========================================================
# LOAD MODEL + THRESHOLD
# ==========================================================

model = joblib.load(
    MODEL_FILE
)

with open(
    METADATA_FILE,
    "r"
) as f:

    metadata = json.load(f)

threshold = float(
    metadata[
        "selected_threshold"
    ]
)

print(
    "Decision threshold:",
    threshold
)


# ==========================================================
# LOAD TEST DATA
# ==========================================================

columns = (
    [
        "date",
        "serial_number",
        TARGET
    ]
    + FEATURES
)

test = pd.read_parquet(
    TEST_FILE,
    columns=columns
)

test["date"] = pd.to_datetime(
    test["date"]
)

X_test = (
    test[
        FEATURES
    ]
    .astype(
        "float32"
    )
)

y_test = (
    test[
        TARGET
    ]
    .astype(
        "int8"
    )
)


# ==========================================================
# MODEL PREDICTIONS
# ==========================================================

print(
    "Generating test probabilities..."
)

probabilities = (
    model.predict_proba(
        X_test
    )[:, 1]
)

predictions = (
    probabilities
    >= threshold
).astype(
    "int8"
)


test[
    "predicted_probability"
] = probabilities

test[
    "predicted_label"
] = predictions


# ==========================================================
# ASSIGN CONFUSION CATEGORY
# ==========================================================

conditions = [

    (
        (y_test == 1)
        & (predictions == 1)
    ),

    (
        (y_test == 0)
        & (predictions == 1)
    ),

    (
        (y_test == 1)
        & (predictions == 0)
    ),

    (
        (y_test == 0)
        & (predictions == 0)
    ),
]

labels = [
    "TP",
    "FP",
    "FN",
    "TN",
]


test[
    "case_type"
] = np.select(
    conditions,
    labels,
    default="UNKNOWN"
)


# ==========================================================
# PRINT CATEGORY COUNTS
# ==========================================================

print(
    "\nConfusion categories:\n"
)

print(
    test[
        "case_type"
    ]
    .value_counts()
)


# ==========================================================
# SELECT REPRESENTATIVE CASE
# ==========================================================

def select_median_case(
    df,
    case_type
):

    subset = (
        df[
            df[
                "case_type"
            ] == case_type
        ]
        .copy()
    )

    if subset.empty:

        return None

    median_probability = (
        subset[
            "predicted_probability"
        ]
        .median()
    )

    subset[
        "distance_from_median"
    ] = np.abs(
        subset[
            "predicted_probability"
        ]
        -
        median_probability
    )

    selected = (
        subset
        .sort_values(
            "distance_from_median"
        )
        .iloc[0]
    )

    return selected


selected_cases = []


for case_type in labels:

    selected = (
        select_median_case(
            test,
            case_type
        )
    )

    if selected is not None:

        selected_cases.append(
            {
                "case_type":
                    case_type,

                "serial_number":
                    selected[
                        "serial_number"
                    ],

                "date":
                    selected[
                        "date"
                    ],

                "actual":
                    int(
                        selected[
                            TARGET
                        ]
                    ),

                "predicted_label":
                    int(
                        selected[
                            "predicted_label"
                        ]
                    ),

                "predicted_probability":
                    float(
                        selected[
                            "predicted_probability"
                        ]
                    ),

                "threshold":
                    threshold,

                "original_index":
                    selected.name,
            }
        )


selected_df = pd.DataFrame(
    selected_cases
)


selected_file = (
    RESULT_DIR
    / "selected_local_cases.csv"
)

selected_df.to_csv(
    selected_file,
    index=False
)


print(
    "\nSelected representative cases:\n"
)

print(
    selected_df.to_string(
        index=False
    )
)


# ==========================================================
# SHAP EXPLAINER
# ==========================================================

print(
    "\nCreating SHAP explainer..."
)

explainer = shap.TreeExplainer(
    model
)


# ==========================================================
# EXPLAIN EACH CASE
# ==========================================================

local_summary_rows = []


for _, case in selected_df.iterrows():

    case_type = (
        case[
            "case_type"
        ]
    )

    row_index = int(
        case[
            "original_index"
        ]
    )


    X_case = (
        X_test.loc[
            [row_index]
        ]
    )


    shap_values = (
        explainer(
            X_case
        )
    )


    values = (
        shap_values.values[0]
    )


    feature_values = (
        X_case.iloc[0]
        .to_numpy()
    )


    local_df = pd.DataFrame(
        {
            "feature":
                FEATURES,

            "feature_value":
                feature_values,

            "shap_value":
                values,

            "abs_shap":
                np.abs(
                    values
                ),
        }
    )


    local_df[
        "direction"
    ] = np.where(
        local_df[
            "shap_value"
        ] > 0,
        "increases_model_failure_output",
        "decreases_model_failure_output"
    )


    local_df = (
        local_df
        .sort_values(
            "abs_shap",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


    local_df[
        "rank"
    ] = (
        np.arange(
            1,
            len(
                local_df
            ) + 1
        )
    )


    # ------------------------------------------------------
    # SAVE FEATURE EXPLANATION
    # ------------------------------------------------------

    local_file = (
        RESULT_DIR
        / f"{case_type}_local_shap.csv"
    )

    local_df.to_csv(
        local_file,
        index=False
    )


    # ------------------------------------------------------
    # SAVE TOP FEATURES TO MASTER SUMMARY
    # ------------------------------------------------------

    for _, feature_row in (
        local_df
        .head(
            TOP_FEATURES
        )
        .iterrows()
    ):

        local_summary_rows.append(
            {
                "case_type":
                    case_type,

                "serial_number":
                    case[
                        "serial_number"
                    ],

                "date":
                    case[
                        "date"
                    ],

                "actual":
                    case[
                        "actual"
                    ],

                "predicted_probability":
                    case[
                        "predicted_probability"
                    ],

                "feature":
                    feature_row[
                        "feature"
                    ],

                "feature_value":
                    feature_row[
                        "feature_value"
                    ],

                "shap_value":
                    feature_row[
                        "shap_value"
                    ],

                "abs_shap":
                    feature_row[
                        "abs_shap"
                    ],

                "direction":
                    feature_row[
                        "direction"
                    ],

                "rank":
                    feature_row[
                        "rank"
                    ],
            }
        )


    # ------------------------------------------------------
    # WATERFALL PLOT
    # ------------------------------------------------------

    plt.figure()

    shap.plots.waterfall(
        shap_values[0],
        max_display=15,
        show=False
    )

    plt.title(
        (
            f"{case_type} | "
            f"P(failure within 30d)="
            f"{case['predicted_probability']:.4f}"
        )
    )

    plt.tight_layout()

    figure_file = (
        FIGURE_DIR
        / f"{case_type}_waterfall.png"
    )

    plt.savefig(
        figure_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    print(
        f"\nTop SHAP features — {case_type}:\n"
    )

    print(
        local_df[
            [
                "feature",
                "feature_value",
                "shap_value",
                "direction",
            ]
        ]
        .head(
            TOP_FEATURES
        )
        .to_string(
            index=False
        )
    )


# ==========================================================
# SAVE MASTER LOCAL SUMMARY
# ==========================================================

local_summary_df = pd.DataFrame(
    local_summary_rows
)

local_summary_file = (
    RESULT_DIR
    / "local_shap_top_features.csv"
)

local_summary_df.to_csv(
    local_summary_file,
    index=False
)


# ==========================================================
# FINAL OUTPUT
# ==========================================================

print("\n")
print("=" * 70)

print(
    "30-DAY LOCAL SHAP ANALYSIS COMPLETE"
)

print("=" * 70)

print(
    "\nSelected cases:"
)

print(
    selected_file
)

print(
    "\nLocal SHAP summaries:"
)

print(
    local_summary_file
)

print(
    "\nFigures saved to:"
)

print(
    FIGURE_DIR
)