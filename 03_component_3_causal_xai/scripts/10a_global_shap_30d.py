from pathlib import Path

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

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "reference_model"
    / f"{HORIZON}d"
    / "xgboost_reference.joblib"
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
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "figures"
    / "shap"
    / f"{HORIZON}d"
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

RANDOM_STATE = 42

POPULATION_SAMPLE_SIZE = 10000

FAILURE_NEGATIVE_SAMPLE_SIZE = 5000


# ==========================================================
# LOAD FEATURE LIST
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
# LOAD MODEL
# ==========================================================

print(
    "Loading model..."
)

model = joblib.load(
    MODEL_FILE
)


# ==========================================================
# LOAD TEST DATA
# ==========================================================

print(
    "Loading test dataset..."
)

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

print(
    "Test rows:",
    f"{len(test):,}"
)

print(
    "Positive rows:",
    f"{(test[TARGET] == 1).sum():,}"
)


# ==========================================================
# POPULATION SAMPLE
# ==========================================================

population_n = min(
    POPULATION_SAMPLE_SIZE,
    len(test)
)

population_sample = (
    test.sample(
        n=population_n,
        random_state=RANDOM_STATE
    )
    .reset_index(
        drop=True
    )
)


# ==========================================================
# FAILURE-FOCUSED SAMPLE
# ==========================================================

positive_rows = (
    test[
        test[TARGET] == 1
    ]
    .copy()
)


negative_rows = (
    test[
        test[TARGET] == 0
    ]
)


negative_n = min(
    FAILURE_NEGATIVE_SAMPLE_SIZE,
    len(negative_rows)
)


negative_sample = (
    negative_rows.sample(
        n=negative_n,
        random_state=RANDOM_STATE
    )
)


failure_focused_sample = (
    pd.concat(
        [
            positive_rows,
            negative_sample
        ],
        ignore_index=True
    )
    .sample(
        frac=1,
        random_state=RANDOM_STATE
    )
    .reset_index(
        drop=True
    )
)


print(
    "\nPopulation SHAP sample:",
    f"{len(population_sample):,}"
)

print(
    "Failure-focused SHAP sample:",
    f"{len(failure_focused_sample):,}"
)


# ==========================================================
# SHAP EXPLAINER
# ==========================================================

print(
    "\nCreating SHAP TreeExplainer..."
)

explainer = shap.TreeExplainer(
    model
)


# ==========================================================
# HELPER FUNCTION
# ==========================================================

def run_shap_analysis(
    data,
    name
):

    print(
        f"\nCalculating SHAP: {name}"
    )

    X = (
        data[
            FEATURES
        ]
        .astype(
            "float32"
        )
    )


    shap_values = (
        explainer(
            X
        )
    )


    # ======================================================
    # GLOBAL MEAN ABSOLUTE SHAP
    # ======================================================

    mean_abs_shap = (
        np.abs(
            shap_values.values
        )
        .mean(
            axis=0
        )
    )


    importance = pd.DataFrame(
        {
            "feature":
                FEATURES,

            "mean_abs_shap":
                mean_abs_shap
        }
    )


    importance = (
        importance
        .sort_values(
            "mean_abs_shap",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


    importance[
        "rank"
    ] = (
        np.arange(
            1,
            len(
                importance
            ) + 1
        )
    )


    importance_file = (
        RESULT_DIR
        / f"{name}_global_shap_importance.csv"
    )


    importance.to_csv(
        importance_file,
        index=False
    )


    # ======================================================
    # SAVE SHAP VALUES
    # ======================================================

    shap_df = pd.DataFrame(
        shap_values.values,
        columns=[
            f"shap_{feature}"
            for feature in FEATURES
        ]
    )


    metadata = (
        data[
            [
                "date",
                "serial_number",
                TARGET
            ]
        ]
        .reset_index(
            drop=True
        )
    )


    shap_output = pd.concat(
        [
            metadata,
            shap_df
        ],
        axis=1
    )


    shap_output.to_parquet(
        RESULT_DIR
        / f"{name}_shap_values.parquet",
        index=False
    )


    # ======================================================
    # SHAP BAR PLOT
    # ======================================================

    plt.figure()

    shap.plots.bar(
        shap_values,
        max_display=20,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / f"{name}_shap_bar_top20.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    # ======================================================
    # SHAP BEESWARM
    # ======================================================

    plt.figure()

    shap.plots.beeswarm(
        shap_values,
        max_display=20,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / f"{name}_shap_beeswarm_top20.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    print(
        f"\nTop 20 SHAP features — {name}:\n"
    )

    print(
        importance
        .head(20)
        .to_string(
            index=False
        )
    )


    return importance


# ==========================================================
# RUN BOTH ANALYSES
# ==========================================================

population_importance = (
    run_shap_analysis(
        population_sample,
        "population"
    )
)


failure_importance = (
    run_shap_analysis(
        failure_focused_sample,
        "failure_focused"
    )
)


# ==========================================================
# COMPARE GLOBAL RANKINGS
# ==========================================================

comparison = (
    population_importance[
        [
            "feature",
            "mean_abs_shap",
            "rank"
        ]
    ]
    .rename(
        columns={
            "mean_abs_shap":
                "population_mean_abs_shap",

            "rank":
                "population_rank"
        }
    )
    .merge(
        failure_importance[
            [
                "feature",
                "mean_abs_shap",
                "rank"
            ]
        ].rename(
            columns={
                "mean_abs_shap":
                    "failure_mean_abs_shap",

                "rank":
                    "failure_rank"
            }
        ),

        on="feature",
        how="inner"
    )
)


comparison[
    "rank_difference"
] = (
    comparison[
        "population_rank"
    ]
    -
    comparison[
        "failure_rank"
    ]
)


comparison.to_csv(
    RESULT_DIR
    / "population_vs_failure_shap_ranking.csv",
    index=False
)


print("\n")
print("=" * 70)

print(
    "30-DAY GLOBAL SHAP ANALYSIS COMPLETE"
)

print("=" * 70)


print(
    "\nResults saved to:"
)

print(
    RESULT_DIR
)


print(
    "\nFigures saved to:"
)

print(
    FIGURE_DIR
)