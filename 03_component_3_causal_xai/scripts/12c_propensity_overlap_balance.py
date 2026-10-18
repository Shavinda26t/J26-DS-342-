from pathlib import Path

import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

COHORT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "causal_effect"
    / "pretreatment_failure_cohort.parquet"
)

MANIFEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "pretreatment_cohort"
    / "pretreatment_adjustment_manifest.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "propensity_diagnostics"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "figures"
    / "causal_effect"
    / "propensity"
)

OUTPUT_DIR.mkdir(
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

TREATMENTS = [

    "treat_reallocated",

    "treat_uncorrectable",

    "treat_pending",
]


RANDOM_STATE = 42


# Propensity clipping is only for diagnostic ATT weights.
# Raw propensity values are also retained and reported.

PROPENSITY_MIN = 1e-5
PROPENSITY_MAX = 1 - 1e-5


# Practical balance interpretation:
#
# |SMD| < 0.10    generally acceptable
# |SMD| < 0.05    strong balance
#
# We will not automatically discard a treatment here.
# We only diagnose.


# ==========================================================
# LOAD ADJUSTMENT VARIABLES
# ==========================================================

manifest = pd.read_csv(
    MANIFEST_FILE
)


ADJUSTMENT_FEATURES = (
    manifest[
        "adjustment_feature"
    ]
    .tolist()
)


print(
    "Adjustment variables:",
    len(
        ADJUSTMENT_FEATURES
    )
)


for feature in ADJUSTMENT_FEATURES:

    print(
        " -",
        feature
    )


# ==========================================================
# LOAD COHORT
# ==========================================================

columns = (

    [
        "date",
        "serial_number",
        TARGET,
    ]

    + TREATMENTS

    + ADJUSTMENT_FEATURES
)


# Remove any accidental duplicate column requests.

columns = list(
    dict.fromkeys(
        columns
    )
)


print(
    "\nLoading pre-treatment cohort..."
)


df = pd.read_parquet(
    COHORT_FILE,
    columns=columns
)


df[
    "date"
] = pd.to_datetime(
    df[
        "date"
    ],
    errors="coerce"
)


print(
    "Rows:",
    f"{len(df):,}"
)


print(
    "Unique HDDs:",
    f"{df['serial_number'].nunique():,}"
)


# ==========================================================
# NUMERIC SAFETY CHECK
# ==========================================================

for feature in ADJUSTMENT_FEATURES:

    df[
        feature
    ] = pd.to_numeric(
        df[
            feature
        ],
        errors="coerce"
    )


missing_adjustment_rows = (
    df[
        ADJUSTMENT_FEATURES
    ]
    .isna()
    .any(
        axis=1
    )
    .sum()
)


if missing_adjustment_rows > 0:

    raise RuntimeError(
        f"Found {missing_adjustment_rows} rows with "
        "missing adjustment variables."
    )


# ==========================================================
# HELPERS
# ==========================================================

def weighted_mean(
    values,
    weights
):

    values = np.asarray(
        values,
        dtype=np.float64
    )

    weights = np.asarray(
        weights,
        dtype=np.float64
    )


    valid = (
        np.isfinite(
            values
        )
        &
        np.isfinite(
            weights
        )
        &
        (
            weights >= 0
        )
    )


    values = values[
        valid
    ]

    weights = weights[
        valid
    ]


    if (
        len(
            values
        ) == 0
        or
        weights.sum() == 0
    ):

        return np.nan


    return np.average(
        values,
        weights=weights
    )


def weighted_variance(
    values,
    weights
):

    values = np.asarray(
        values,
        dtype=np.float64
    )

    weights = np.asarray(
        weights,
        dtype=np.float64
    )


    valid = (
        np.isfinite(
            values
        )
        &
        np.isfinite(
            weights
        )
        &
        (
            weights >= 0
        )
    )


    values = values[
        valid
    ]

    weights = weights[
        valid
    ]


    if (
        len(
            values
        ) == 0
        or
        weights.sum() == 0
    ):

        return np.nan


    mean = np.average(
        values,
        weights=weights
    )


    variance = np.average(
        (
            values
            - mean
        ) ** 2,
        weights=weights
    )


    return variance


def standardized_mean_difference(
    treated_values,
    control_values,
    treated_weights=None,
    control_weights=None,
):

    treated_values = np.asarray(
        treated_values,
        dtype=np.float64
    )

    control_values = np.asarray(
        control_values,
        dtype=np.float64
    )


    if treated_weights is None:

        treated_weights = np.ones(
            len(
                treated_values
            ),
            dtype=np.float64
        )


    if control_weights is None:

        control_weights = np.ones(
            len(
                control_values
            ),
            dtype=np.float64
        )


    treated_mean = weighted_mean(
        treated_values,
        treated_weights
    )


    control_mean = weighted_mean(
        control_values,
        control_weights
    )


    treated_var = weighted_variance(
        treated_values,
        treated_weights
    )


    control_var = weighted_variance(
        control_values,
        control_weights
    )


    pooled_sd = np.sqrt(
        (
            treated_var
            +
            control_var
        )
        / 2
    )


    if (
        not np.isfinite(
            pooled_sd
        )
        or
        pooled_sd == 0
    ):

        if np.isclose(
            treated_mean,
            control_mean
        ):

            return 0.0

        return np.nan


    return (
        treated_mean
        -
        control_mean
    ) / pooled_sd


def effective_sample_size(
    weights
):

    weights = np.asarray(
        weights,
        dtype=np.float64
    )


    weights = weights[
        np.isfinite(
            weights
        )
        &
        (
            weights >= 0
        )
    ]


    if (
        len(
            weights
        ) == 0
        or
        np.sum(
            weights ** 2
        ) == 0
    ):

        return np.nan


    return (
        weights.sum() ** 2
        /
        np.sum(
            weights ** 2
        )
    )


# ==========================================================
# STORAGE
# ==========================================================

treatment_summary_rows = []

overlap_summary_rows = []

balance_frames = []

propensity_quantile_frames = []


# ==========================================================
# RUN EACH TREATMENT
# ==========================================================

for treatment in TREATMENTS:

    print("\n")
    print("=" * 70)

    print(
        f"PROPENSITY DIAGNOSTICS: {treatment}"
    )

    print("=" * 70)


    analysis = df[
        [
            treatment,
            TARGET,
        ]
        +
        ADJUSTMENT_FEATURES
    ].copy()


    analysis[
        treatment
    ] = (
        analysis[
            treatment
        ]
        .astype(
            "int8"
        )
    )


    X = (
        analysis[
            ADJUSTMENT_FEATURES
        ]
        .astype(
            "float64"
        )
    )


    y = (
        analysis[
            treatment
        ]
        .astype(
            "int8"
        )
    )


    treated_mask = (
        y == 1
    )

    control_mask = (
        y == 0
    )


    treated_n = int(
        treated_mask.sum()
    )

    control_n = int(
        control_mask.sum()
    )


    print(
        "Treated rows:",
        f"{treated_n:,}"
    )


    print(
        "Control rows:",
        f"{control_n:,}"
    )


    if treated_n == 0:

        print(
            "Skipping: no treated observations."
        )

        continue


    # ======================================================
    # PROPENSITY MODEL
    #
    # IMPORTANT:
    #
    # Do NOT use class_weight='balanced'.
    #
    # Balanced class weights would alter the probability
    # scale and make the output unsuitable as an ordinary
    # propensity score.
    #
    # SGD logistic regression is used because the cohort has
    # >2 million rows.
    # ======================================================

    propensity_model = Pipeline(
        steps=[

            (
                "scaler",

                StandardScaler()
            ),

            (
                "model",

                SGDClassifier(

                    loss="log_loss",

                    penalty="l2",

                    alpha=1e-5,

                    max_iter=2000,

                    tol=1e-4,

                    random_state=RANDOM_STATE,

                    average=True,
                )
            ),
        ]
    )


    print(
        "Fitting propensity model..."
    )


    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore"
        )


        propensity_model.fit(
            X,
            y
        )


    propensity_raw = (
        propensity_model
        .predict_proba(
            X
        )[:, 1]
    )


    if not np.isfinite(
        propensity_raw
    ).all():

        raise RuntimeError(
            f"Invalid propensity values for {treatment}"
        )


    propensity_auc = roc_auc_score(
        y,
        propensity_raw
    )


    print(
        "Propensity-model ROC-AUC:",
        round(
            propensity_auc,
            6
        )
    )


    # ======================================================
    # PROPENSITY CLIPPING FOR WEIGHTS
    # ======================================================

    propensity = np.clip(

        propensity_raw,

        PROPENSITY_MIN,

        PROPENSITY_MAX,
    )


    # ======================================================
    # ATT WEIGHTS
    #
    # Treated = 1
    #
    # Controls = e(X) / [1 - e(X)]
    #
    # ATT asks:
    #
    # What would have happened to HDD-days that actually
    # experienced this degradation event if they had not
    # experienced the event?
    # ======================================================

    att_weights = np.where(

        treated_mask,

        1.0,

        propensity
        /
        (
            1.0
            -
            propensity
        )
    )


    # ======================================================
    # RAW PROPENSITY QUANTILES
    # ======================================================

    quantiles = [

        0.00,
        0.01,
        0.05,
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
        1.00,
    ]


    for group_name, mask in [

        (
            "treated",
            treated_mask
        ),

        (
            "control",
            control_mask
        ),
    ]:

        values = propensity_raw[
            mask
        ]


        q_values = np.quantile(
            values,
            quantiles
        )


        temp = pd.DataFrame(
            {
                "treatment":
                    treatment,

                "group":
                    group_name,

                "quantile":
                    quantiles,

                "propensity":
                    q_values,
            }
        )


        propensity_quantile_frames.append(
            temp
        )


    # ======================================================
    # COMMON SUPPORT
    # ======================================================

    treated_propensity = (
        propensity_raw[
            treated_mask
        ]
    )


    control_propensity = (
        propensity_raw[
            control_mask
        ]
    )


    common_support_min = max(

        treated_propensity.min(),

        control_propensity.min(),
    )


    common_support_max = min(

        treated_propensity.max(),

        control_propensity.max(),
    )


    treated_in_common_support = (

        (
            treated_propensity
            >= common_support_min
        )
        &
        (
            treated_propensity
            <= common_support_max
        )
    )


    control_in_common_support = (

        (
            control_propensity
            >= common_support_min
        )
        &
        (
            control_propensity
            <= common_support_max
        )
    )


    # ------------------------------------------------------
    # Robust 1%-99% support range
    #
    # This is often more informative than extreme min/max.
    # ------------------------------------------------------

    robust_support_min = max(

        np.quantile(
            treated_propensity,
            0.01
        ),

        np.quantile(
            control_propensity,
            0.01
        ),
    )


    robust_support_max = min(

        np.quantile(
            treated_propensity,
            0.99
        ),

        np.quantile(
            control_propensity,
            0.99
        ),
    )


    robust_support_exists = (
        robust_support_min
        <= robust_support_max
    )


    if robust_support_exists:

        treated_in_robust_support = (

            (
                treated_propensity
                >= robust_support_min
            )
            &
            (
                treated_propensity
                <= robust_support_max
            )
        )


        control_in_robust_support = (

            (
                control_propensity
                >= robust_support_min
            )
            &
            (
                control_propensity
                <= robust_support_max
            )
        )


        treated_robust_support_pct = (

            treated_in_robust_support.mean()
            * 100
        )


        control_robust_support_pct = (

            control_in_robust_support.mean()
            * 100
        )

    else:

        treated_robust_support_pct = 0.0

        control_robust_support_pct = 0.0


    # ======================================================
    # ATT EFFECTIVE SAMPLE SIZE
    # ======================================================

    treated_weights = (
        att_weights[
            treated_mask
        ]
    )


    control_weights = (
        att_weights[
            control_mask
        ]
    )


    treated_ess = effective_sample_size(
        treated_weights
    )


    control_ess = effective_sample_size(
        control_weights
    )


    # ======================================================
    # WEIGHT DIAGNOSTICS
    # ======================================================

    control_weight_quantiles = np.quantile(

        control_weights,

        [
            0.50,
            0.90,
            0.95,
            0.99,
            0.999,
            1.00,
        ]
    )


    # ======================================================
    # COVARIATE BALANCE
    # ======================================================

    balance_rows = []


    for feature in ADJUSTMENT_FEATURES:

        values = (
            analysis[
                feature
            ]
            .to_numpy(
                dtype=np.float64
            )
        )


        treated_values = values[
            treated_mask
        ]


        control_values = values[
            control_mask
        ]


        smd_before = (
            standardized_mean_difference(

                treated_values,

                control_values,
            )
        )


        smd_after = (
            standardized_mean_difference(

                treated_values,

                control_values,

                treated_weights=
                    treated_weights,

                control_weights=
                    control_weights,
            )
        )


        balance_rows.append(
            {
                "treatment":
                    treatment,

                "feature":
                    feature,

                "smd_before":
                    smd_before,

                "abs_smd_before":
                    abs(
                        smd_before
                    )
                    if np.isfinite(
                        smd_before
                    )
                    else np.nan,

                "smd_after_att":
                    smd_after,

                "abs_smd_after_att":
                    abs(
                        smd_after
                    )
                    if np.isfinite(
                        smd_after
                    )
                    else np.nan,
            }
        )


    balance_df = pd.DataFrame(
        balance_rows
    )


    balance_df = (
        balance_df
        .sort_values(
            "abs_smd_after_att",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


    balance_frames.append(
        balance_df
    )


    max_abs_smd_before = (
        balance_df[
            "abs_smd_before"
        ].max()
    )


    max_abs_smd_after = (
        balance_df[
            "abs_smd_after_att"
        ].max()
    )


    features_above_010_before = int(
        (
            balance_df[
                "abs_smd_before"
            ] >= 0.10
        ).sum()
    )


    features_above_010_after = int(
        (
            balance_df[
                "abs_smd_after_att"
            ] >= 0.10
        ).sum()
    )


    features_above_005_after = int(
        (
            balance_df[
                "abs_smd_after_att"
            ] >= 0.05
        ).sum()
    )


    # ======================================================
    # SUMMARY ROW
    # ======================================================

    treatment_summary_rows.append(
        {
            "treatment":
                treatment,

            "treated_rows":
                treated_n,

            "control_rows":
                control_n,

            "propensity_auc":
                propensity_auc,

            "treated_propensity_min":
                treated_propensity.min(),

            "treated_propensity_median":
                np.median(
                    treated_propensity
                ),

            "treated_propensity_max":
                treated_propensity.max(),

            "control_propensity_min":
                control_propensity.min(),

            "control_propensity_median":
                np.median(
                    control_propensity
                ),

            "control_propensity_max":
                control_propensity.max(),

            "common_support_min":
                common_support_min,

            "common_support_max":
                common_support_max,

            "treated_common_support_pct":
                treated_in_common_support.mean()
                * 100,

            "control_common_support_pct":
                control_in_common_support.mean()
                * 100,

            "robust_support_min":
                robust_support_min,

            "robust_support_max":
                robust_support_max,

            "treated_robust_support_pct":
                treated_robust_support_pct,

            "control_robust_support_pct":
                control_robust_support_pct,

            "att_treated_ess":
                treated_ess,

            "att_control_ess":
                control_ess,

            "att_control_weight_p50":
                control_weight_quantiles[
                    0
                ],

            "att_control_weight_p90":
                control_weight_quantiles[
                    1
                ],

            "att_control_weight_p95":
                control_weight_quantiles[
                    2
                ],

            "att_control_weight_p99":
                control_weight_quantiles[
                    3
                ],

            "att_control_weight_p999":
                control_weight_quantiles[
                    4
                ],

            "att_control_weight_max":
                control_weight_quantiles[
                    5
                ],

            "max_abs_smd_before":
                max_abs_smd_before,

            "max_abs_smd_after_att":
                max_abs_smd_after,

            "features_abs_smd_ge_010_before":
                features_above_010_before,

            "features_abs_smd_ge_010_after_att":
                features_above_010_after,

            "features_abs_smd_ge_005_after_att":
                features_above_005_after,
        }
    )


    # ======================================================
    # OVERLAP SUMMARY
    # ======================================================

    overlap_summary_rows.append(
        {
            "treatment":
                treatment,

            "treated_rows":
                treated_n,

            "control_rows":
                control_n,

            "treated_in_common_support":
                int(
                    treated_in_common_support.sum()
                ),

            "control_in_common_support":
                int(
                    control_in_common_support.sum()
                ),

            "treated_common_support_pct":
                treated_in_common_support.mean()
                * 100,

            "control_common_support_pct":
                control_in_common_support.mean()
                * 100,

            "treated_robust_support_pct":
                treated_robust_support_pct,

            "control_robust_support_pct":
                control_robust_support_pct,
        }
    )


    # ======================================================
    # PROPENSITY DISTRIBUTION FIGURE
    # ======================================================

    plt.figure(
        figsize=(
            9,
            5
        )
    )


    plt.hist(

        control_propensity,

        bins=100,

        density=True,

        alpha=0.6,

        label="Control"
    )


    plt.hist(

        treated_propensity,

        bins=100,

        density=True,

        alpha=0.6,

        label="Treated"
    )


    plt.xlabel(
        "Estimated propensity score"
    )


    plt.ylabel(
        "Density"
    )


    plt.title(
        f"Propensity-score overlap: {treatment}"
    )


    plt.legend()


    plt.tight_layout()


    figure_file = (
        FIGURE_DIR
        / f"{treatment}_propensity_overlap.png"
    )


    plt.savefig(
        figure_file,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    # ======================================================
    # SMD LOVE-PLOT STYLE FIGURE
    # ======================================================

    plot_balance = (
        balance_df
        .sort_values(
            "abs_smd_before",
            ascending=True
        )
    )


    y_positions = np.arange(
        len(
            plot_balance
        )
    )


    plt.figure(
        figsize=(
            10,
            8
        )
    )


    plt.scatter(

        plot_balance[
            "smd_before"
        ],

        y_positions,

        label="Before weighting"
    )


    plt.scatter(

        plot_balance[
            "smd_after_att"
        ],

        y_positions,

        label="After ATT weighting"
    )


    plt.axvline(
        0,
        linewidth=1
    )


    plt.axvline(
        -0.10,
        linestyle="--",
        linewidth=1
    )


    plt.axvline(
        0.10,
        linestyle="--",
        linewidth=1
    )


    plt.yticks(

        y_positions,

        plot_balance[
            "feature"
        ]
    )


    plt.xlabel(
        "Standardized mean difference"
    )


    plt.title(
        f"Covariate balance: {treatment}"
    )


    plt.legend()


    plt.tight_layout()


    balance_figure_file = (
        FIGURE_DIR
        / f"{treatment}_covariate_balance.png"
    )


    plt.savefig(
        balance_figure_file,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    # ======================================================
    # PRINT TREATMENT RESULT
    # ======================================================

    print(
        "\nATT overlap diagnostics:"
    )


    print(
        "Treated common support:",
        f"{treated_in_common_support.mean() * 100:.2f}%"
    )


    print(
        "Control common support:",
        f"{control_in_common_support.mean() * 100:.2f}%"
    )


    print(
        "Treated robust 1-99% support:",
        f"{treated_robust_support_pct:.2f}%"
    )


    print(
        "Control robust 1-99% support:",
        f"{control_robust_support_pct:.2f}%"
    )


    print(
        "ATT control ESS:",
        round(
            control_ess,
            2
        )
    )


    print(
        "Maximum |SMD| before weighting:",
        round(
            max_abs_smd_before,
            4
        )
    )


    print(
        "Maximum |SMD| after ATT weighting:",
        round(
            max_abs_smd_after,
            4
        )
    )


    print(
        "Features with |SMD| >= 0.10 after weighting:",
        features_above_010_after
    )


    print(
        "\nWorst post-weighting balance:\n"
    )


    print(
        balance_df[
            [
                "feature",
                "smd_before",
                "smd_after_att",
            ]
        ]
        .head(
            10
        )
        .to_string(
            index=False
        )
    )


# ==========================================================
# COMBINE RESULTS
# ==========================================================

summary_df = pd.DataFrame(
    treatment_summary_rows
)


overlap_df = pd.DataFrame(
    overlap_summary_rows
)


balance_all_df = pd.concat(
    balance_frames,
    ignore_index=True
)


propensity_quantiles_df = pd.concat(
    propensity_quantile_frames,
    ignore_index=True
)


# ==========================================================
# SAVE OUTPUTS
# ==========================================================

summary_file = (
    OUTPUT_DIR
    / "propensity_diagnostic_summary.csv"
)


balance_file = (
    OUTPUT_DIR
    / "covariate_balance_all_treatments.csv"
)


overlap_file = (
    OUTPUT_DIR
    / "propensity_overlap_summary.csv"
)


quantile_file = (
    OUTPUT_DIR
    / "propensity_quantiles.csv"
)


summary_df.to_csv(
    summary_file,
    index=False
)


balance_all_df.to_csv(
    balance_file,
    index=False
)


overlap_df.to_csv(
    overlap_file,
    index=False
)


propensity_quantiles_df.to_csv(
    quantile_file,
    index=False
)


# ==========================================================
# SIMPLE READINESS CLASSIFICATION
# ==========================================================

readiness_rows = []


for _, row in summary_df.iterrows():

    treatment = row[
        "treatment"
    ]


    # Conservative diagnostic classification.
    #
    # READY:
    # max post-weighting SMD < 0.10
    # and >= 80% treated observations within robust support
    #
    # REVIEW:
    # otherwise.


    balance_ok = (
        row[
            "max_abs_smd_after_att"
        ] < 0.10
    )


    support_ok = (
        row[
            "treated_robust_support_pct"
        ] >= 80.0
    )


    ess_ok = (
        row[
            "att_control_ess"
        ] >= 100
    )


    if (
        balance_ok
        and
        support_ok
        and
        ess_ok
    ):

        status = (
            "READY_FOR_ATT_ESTIMATION"
        )

    else:

        status = (
            "REVIEW_BEFORE_EFFECT_ESTIMATION"
        )


    readiness_rows.append(
        {
            "treatment":
                treatment,

            "balance_ok":
                balance_ok,

            "robust_support_ok":
                support_ok,

            "control_ess_ok":
                ess_ok,

            "status":
                status,
        }
    )


readiness_df = pd.DataFrame(
    readiness_rows
)


readiness_file = (
    OUTPUT_DIR
    / "causal_effect_readiness_decision.csv"
)


readiness_df.to_csv(
    readiness_file,
    index=False
)


# ==========================================================
# SAVE METADATA
# ==========================================================

metadata = {

    "estimand":
        "ATT",

    "propensity_model":
        "SGD logistic regression",

    "class_weight":
        None,

    "adjustment_variables":
        ADJUSTMENT_FEATURES,

    "number_of_adjustment_variables":
        len(
            ADJUSTMENT_FEATURES
        ),

    "propensity_clip_min":
        PROPENSITY_MIN,

    "propensity_clip_max":
        PROPENSITY_MAX,

    "treatments":
        TREATMENTS,

    "random_state":
        RANDOM_STATE,
}


metadata_file = (
    OUTPUT_DIR
    / "propensity_diagnostic_metadata.json"
)


with open(
    metadata_file,
    "w"
) as file:

    json.dump(
        metadata,
        file,
        indent=4
    )


# ==========================================================
# FINAL PRINT
# ==========================================================

print("\n")
print("=" * 70)

print(
    "PROPENSITY OVERLAP AND BALANCE DIAGNOSTICS COMPLETE"
)

print("=" * 70)


print(
    "\nDiagnostic summary:\n"
)


display_columns = [

    "treatment",

    "treated_rows",

    "propensity_auc",

    "treated_robust_support_pct",

    "control_robust_support_pct",

    "att_control_ess",

    "max_abs_smd_before",

    "max_abs_smd_after_att",

    "features_abs_smd_ge_010_after_att",
]


print(
    summary_df[
        display_columns
    ].to_string(
        index=False
    )
)


print(
    "\nReadiness decision:\n"
)


print(
    readiness_df.to_string(
        index=False
    )
)


print(
    "\nResults saved to:"
)


print(
    OUTPUT_DIR
)


print(
    "\nFigures saved to:"
)


print(
    FIGURE_DIR
)