from pathlib import Path

import numpy as np
import pandas as pd


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_ROOT = (
    PROJECT_ROOT
    / "results"
)

INTEGRATED_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "integrated_evidence"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "integrated_evidence"
    / "global_shap"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# INPUT FILES
# ==========================================================

POPULATION_SHAP_FILE = (
    RESULTS_ROOT
    / "shap"
    / "30d"
    / "population_global_shap_importance.csv"
)

FAILURE_SHAP_FILE = (
    RESULTS_ROOT
    / "shap"
    / "30d"
    / "failure_focused_global_shap_importance.csv"
)

MAPPING_FILE = (
    INTEGRATED_DIR
    / "shap_feature_to_factor_mapping.csv"
)

FACTOR_MANIFEST_FILE = (
    INTEGRATED_DIR
    / "canonical_factor_manifest.csv"
)


# ==========================================================
# VALIDATE INPUT FILES
# ==========================================================

required_files = [

    POPULATION_SHAP_FILE,

    FAILURE_SHAP_FILE,

    MAPPING_FILE,

    FACTOR_MANIFEST_FILE,
]


missing_files = [

    str(
        path
    )

    for path in required_files

    if not path.exists()
]


if missing_files:

    raise FileNotFoundError(
        "Missing required input files:\n"
        +
        "\n".join(
            missing_files
        )
    )


# ==========================================================
# LOAD DATA
# ==========================================================

population = pd.read_csv(
    POPULATION_SHAP_FILE
)


failure = pd.read_csv(
    FAILURE_SHAP_FILE
)


mapping = pd.read_csv(
    MAPPING_FILE
)


manifest = pd.read_csv(
    FACTOR_MANIFEST_FILE
)


# ==========================================================
# BASIC VALIDATION
# ==========================================================

required_shap_columns = {

    "feature",

    "mean_abs_shap",

    "rank",
}


for name, frame in [

    (
        "population",
        population
    ),

    (
        "failure",
        failure
    ),
]:

    missing = (

        required_shap_columns

        -

        set(
            frame.columns
        )
    )


    if missing:

        raise RuntimeError(
            f"{name} SHAP file missing columns: "
            f"{sorted(missing)}"
        )


required_mapping_columns = {

    "original_feature",

    "mapped",

    "factor_id",

    "factor_name",

    "category",

    "semantic_status",

    "representation_type",

    "aggregation",

    "window_days",

    "temporal_scope",
}


missing_mapping = (

    required_mapping_columns

    -

    set(
        mapping.columns
    )
)


if missing_mapping:

    raise RuntimeError(
        "Mapping file missing columns: "
        f"{sorted(missing_mapping)}"
    )


# ==========================================================
# ENSURE EVERY SHAP FEATURE IS MAPPED
# ==========================================================

mapping = mapping[
    mapping[
        "mapped"
    ] == True
].copy()


population_features = set(
    population[
        "feature"
    ].astype(str)
)


failure_features = set(
    failure[
        "feature"
    ].astype(str)
)


mapped_features = set(
    mapping[
        "original_feature"
    ].astype(str)
)


population_unmapped = (
    population_features
    -
    mapped_features
)


failure_unmapped = (
    failure_features
    -
    mapped_features
)


if population_unmapped:

    raise RuntimeError(
        "Population SHAP contains unmapped features:\n"
        +
        "\n".join(
            sorted(
                population_unmapped
            )
        )
    )


if failure_unmapped:

    raise RuntimeError(
        "Failure-focused SHAP contains unmapped features:\n"
        +
        "\n".join(
            sorted(
                failure_unmapped
            )
        )
    )


# ==========================================================
# ENSURE EXPECTED FEATURE COUNTS
# ==========================================================

print(
    "\nPopulation SHAP features:",
    len(
        population
    )
)


print(
    "Failure-focused SHAP features:",
    len(
        failure
    )
)


print(
    "Mapped features:",
    len(
        mapping
    )
)


# ==========================================================
# PREPARE MAPPING COLUMNS
# ==========================================================

mapping_columns = [

    "original_feature",

    "factor_id",

    "factor_name",

    "category",

    "semantic_status",

    "representation_type",

    "aggregation",

    "window_days",

    "temporal_scope",
]


mapping_small = mapping[
    mapping_columns
].copy()


# ==========================================================
# MERGE POPULATION SHAP
# ==========================================================

population_mapped = population.merge(

    mapping_small,

    left_on="feature",

    right_on="original_feature",

    how="left",

    validate="one_to_one",
)


# ==========================================================
# MERGE FAILURE-FOCUSED SHAP
# ==========================================================

failure_mapped = failure.merge(

    mapping_small,

    left_on="feature",

    right_on="original_feature",

    how="left",

    validate="one_to_one",
)


# ==========================================================
# FINAL MERGE VALIDATION
# ==========================================================

if population_mapped[
    "factor_id"
].isna().any():

    raise RuntimeError(
        "Population SHAP mapping produced missing factor IDs."
    )


if failure_mapped[
    "factor_id"
].isna().any():

    raise RuntimeError(
        "Failure-focused SHAP mapping produced missing factor IDs."
    )


# ==========================================================
# NUMERIC SAFETY
# ==========================================================

population_mapped[
    "mean_abs_shap"
] = pd.to_numeric(
    population_mapped[
        "mean_abs_shap"
    ],
    errors="raise"
)


failure_mapped[
    "mean_abs_shap"
] = pd.to_numeric(
    failure_mapped[
        "mean_abs_shap"
    ],
    errors="raise"
)


# ==========================================================
# HELPER:
# AGGREGATE ONE SHAP DATASET TO FACTOR LEVEL
# ==========================================================

def aggregate_factor_shap(
    mapped_df,
    evidence_population
):

    grouped = (

        mapped_df

        .groupby(
            [
                "factor_id",

                "factor_name",

                "category",

                "semantic_status",
            ],
            as_index=False
        )

        .agg(

            representation_count=(
                "feature",
                "count"
            ),

            factor_total_mean_abs_shap=(
                "mean_abs_shap",
                "sum"
            ),

            factor_mean_representation_shap=(
                "mean_abs_shap",
                "mean"
            ),

            factor_max_representation_shap=(
                "mean_abs_shap",
                "max"
            ),
        )
    )


    total_shap_mass = (
        grouped[
            "factor_total_mean_abs_shap"
        ].sum()
    )


    grouped[
        "factor_shap_share"
    ] = (

        grouped[
            "factor_total_mean_abs_shap"
        ]

        /

        total_shap_mass
    )


    grouped[
        "factor_shap_share_pct"
    ] = (

        grouped[
            "factor_shap_share"
        ]

        *
        100
    )


    grouped[
        "factor_rank"
    ] = (

        grouped[
            "factor_total_mean_abs_shap"
        ]

        .rank(
            method="min",
            ascending=False
        )

        .astype(
            int
        )
    )


    grouped[
        "evidence_population"
    ] = evidence_population


    grouped = (
        grouped
        .sort_values(
            "factor_rank"
        )
        .reset_index(
            drop=True
        )
    )


    return grouped


# ==========================================================
# POPULATION FACTOR IMPORTANCE
# ==========================================================

population_factor = aggregate_factor_shap(

    population_mapped,

    "population",
)


# ==========================================================
# FAILURE-FOCUSED FACTOR IMPORTANCE
# ==========================================================

failure_factor = aggregate_factor_shap(

    failure_mapped,

    "failure_focused",
)


# ==========================================================
# VALIDATE REPRESENTATION COUNT
#
# Current design should produce 11 representations
# for each of the 16 canonical factors.
# ==========================================================

representation_count_audit = (

    pd.concat(
        [

            population_factor[
                [
                    "factor_id",
                    "representation_count"
                ]
            ]
            .assign(
                evidence_population="population"
            ),

            failure_factor[
                [
                    "factor_id",
                    "representation_count"
                ]
            ]
            .assign(
                evidence_population="failure_focused"
            ),
        ],

        ignore_index=True
    )
)


unexpected_counts = (
    representation_count_audit[
        representation_count_audit[
            "representation_count"
        ] != 11
    ]
)


if len(
    unexpected_counts
) > 0:

    print(
        "\nWARNING: Some factors do not have "
        "11 SHAP representations:\n"
    )


    print(
        unexpected_counts.to_string(
            index=False
        )
    )


# ==========================================================
# DOMINANT REPRESENTATION FOR EACH FACTOR
# ==========================================================

def dominant_features(
    mapped_df,
    evidence_population
):

    ranked = (

        mapped_df

        .sort_values(
            [
                "factor_id",
                "mean_abs_shap",
            ],
            ascending=[
                True,
                False,
            ]
        )

        .copy()
    )


    dominant = (

        ranked

        .groupby(
            "factor_id",
            as_index=False
        )

        .first()
    )


    dominant = dominant[
        [
            "factor_id",

            "factor_name",

            "feature",

            "representation_type",

            "aggregation",

            "window_days",

            "temporal_scope",

            "mean_abs_shap",

            "rank",
        ]
    ].copy()


    dominant = dominant.rename(
        columns={

            "feature":
                "dominant_feature",

            "representation_type":
                "dominant_representation_type",

            "aggregation":
                "dominant_aggregation",

            "window_days":
                "dominant_window_days",

            "temporal_scope":
                "dominant_temporal_scope",

            "mean_abs_shap":
                "dominant_feature_mean_abs_shap",

            "rank":
                "dominant_feature_global_rank",
        }
    )


    dominant[
        "evidence_population"
    ] = evidence_population


    return dominant


population_dominant = dominant_features(

    population_mapped,

    "population",
)


failure_dominant = dominant_features(

    failure_mapped,

    "failure_focused",
)


# ==========================================================
# REPRESENTATION-LEVEL PROFILE
#
# This tells us whether factor importance is coming from:
#
# raw value
# daily change
# rolling mean
# rolling variability
# rolling maximum
# ==========================================================

def representation_profile(
    mapped_df,
    evidence_population
):

    profile = (

        mapped_df

        .groupby(
            [
                "factor_id",

                "factor_name",

                "representation_type",
            ],
            as_index=False
        )

        .agg(

            feature_count=(
                "feature",
                "count"
            ),

            representation_total_mean_abs_shap=(
                "mean_abs_shap",
                "sum"
            ),

            representation_mean_abs_shap=(
                "mean_abs_shap",
                "mean"
            ),

            representation_max_abs_shap=(
                "mean_abs_shap",
                "max"
            ),
        )
    )


    factor_totals = (

        profile

        .groupby(
            "factor_id"
        )[
            "representation_total_mean_abs_shap"
        ]

        .transform(
            "sum"
        )
    )


    profile[
        "within_factor_share"
    ] = (

        profile[
            "representation_total_mean_abs_shap"
        ]

        /

        factor_totals
    )


    profile[
        "within_factor_share_pct"
    ] = (

        profile[
            "within_factor_share"
        ]

        *
        100
    )


    profile[
        "evidence_population"
    ] = evidence_population


    return profile


population_representation = representation_profile(

    population_mapped,

    "population",
)


failure_representation = representation_profile(

    failure_mapped,

    "failure_focused",
)


# ==========================================================
# TEMPORAL-WINDOW PROFILE
#
# Keep raw/current, daily change and individual
# 7/14/30-day windows visible.
# ==========================================================

def temporal_window_profile(
    mapped_df,
    evidence_population
):

    temp = mapped_df.copy()


    temp[
        "window_label"
    ] = np.where(

        temp[
            "representation_type"
        ]
        ==
        "raw_state",

        "current_day",

        np.where(

            temp[
                "representation_type"
            ]
            ==
            "daily_change",

            "1_day_change",

            temp[
                "window_days"
            ]
            .fillna(
                -1
            )
            .astype(
                int
            )
            .astype(
                str
            )
            +
            "_day_window"
        )
    )


    profile = (

        temp

        .groupby(
            [
                "factor_id",

                "factor_name",

                "window_label",
            ],
            as_index=False
        )

        .agg(

            feature_count=(
                "feature",
                "count"
            ),

            window_total_mean_abs_shap=(
                "mean_abs_shap",
                "sum"
            ),

            window_mean_abs_shap=(
                "mean_abs_shap",
                "mean"
            ),

            window_max_abs_shap=(
                "mean_abs_shap",
                "max"
            ),
        )
    )


    factor_totals = (

        profile

        .groupby(
            "factor_id"
        )[
            "window_total_mean_abs_shap"
        ]

        .transform(
            "sum"
        )
    )


    profile[
        "within_factor_window_share"
    ] = (

        profile[
            "window_total_mean_abs_shap"
        ]

        /

        factor_totals
    )


    profile[
        "within_factor_window_share_pct"
    ] = (

        profile[
            "within_factor_window_share"
        ]

        *
        100
    )


    profile[
        "evidence_population"
    ] = evidence_population


    return profile


population_window = temporal_window_profile(

    population_mapped,

    "population",
)


failure_window = temporal_window_profile(

    failure_mapped,

    "failure_focused",
)


# ==========================================================
# POPULATION VS FAILURE-FOCUSED FACTOR STABILITY
# ==========================================================

population_compare = population_factor[
    [
        "factor_id",

        "factor_name",

        "factor_total_mean_abs_shap",

        "factor_shap_share_pct",

        "factor_rank",
    ]
].copy()


population_compare = population_compare.rename(
    columns={

        "factor_total_mean_abs_shap":
            "population_total_mean_abs_shap",

        "factor_shap_share_pct":
            "population_shap_share_pct",

        "factor_rank":
            "population_factor_rank",
    }
)


failure_compare = failure_factor[
    [
        "factor_id",

        "factor_total_mean_abs_shap",

        "factor_shap_share_pct",

        "factor_rank",
    ]
].copy()


failure_compare = failure_compare.rename(
    columns={

        "factor_total_mean_abs_shap":
            "failure_total_mean_abs_shap",

        "factor_shap_share_pct":
            "failure_shap_share_pct",

        "factor_rank":
            "failure_factor_rank",
    }
)


factor_stability = population_compare.merge(

    failure_compare,

    on="factor_id",

    how="inner",

    validate="one_to_one",
)


factor_stability[
    "rank_difference"
] = (

    factor_stability[
        "population_factor_rank"
    ]

    -

    factor_stability[
        "failure_factor_rank"
    ]
)


factor_stability[
    "abs_rank_difference"
] = (

    factor_stability[
        "rank_difference"
    ].abs()
)


factor_stability[
    "shap_share_difference_pct"
] = (

    factor_stability[
        "population_shap_share_pct"
    ]

    -

    factor_stability[
        "failure_shap_share_pct"
    ]
)


factor_stability = (
    factor_stability
    .sort_values(
        [
            "population_factor_rank",
            "failure_factor_rank",
        ]
    )
    .reset_index(
        drop=True
    )
)


# ==========================================================
# SPEARMAN FACTOR-RANK STABILITY
# ==========================================================

factor_rank_spearman = (

    factor_stability[
        "population_factor_rank"
    ]

    .corr(

        factor_stability[
            "failure_factor_rank"
        ],

        method="spearman"
    )
)


# ==========================================================
# TOP-K OVERLAP
# ==========================================================

def top_k_overlap(
    stability_df,
    k
):

    population_top = set(

        stability_df
        .nsmallest(
            k,
            "population_factor_rank"
        )[
            "factor_id"
        ]
    )


    failure_top = set(

        stability_df
        .nsmallest(
            k,
            "failure_factor_rank"
        )[
            "factor_id"
        ]
    )


    intersection = (
        population_top
        &
        failure_top
    )


    union = (
        population_top
        |
        failure_top
    )


    overlap_count = len(
        intersection
    )


    jaccard = (

        overlap_count

        /

        len(
            union
        )

        if len(
            union
        ) > 0

        else np.nan
    )


    return {

        "top_k":
            k,

        "overlap_count":
            overlap_count,

        "overlap_pct":
            overlap_count
            /
            k
            *
            100,

        "jaccard":
            jaccard,

        "shared_factors":
            ";".join(
                sorted(
                    intersection
                )
            ),
    }


top_overlap_df = pd.DataFrame(
    [

        top_k_overlap(
            factor_stability,
            3
        ),

        top_k_overlap(
            factor_stability,
            5
        ),

        top_k_overlap(
            factor_stability,
            10
        ),
    ]
)


# ==========================================================
# DOMINANT REPRESENTATION MERGE
# ==========================================================

population_factor_complete = population_factor.merge(

    population_dominant.drop(
        columns=[
            "factor_name",
            "evidence_population",
        ]
    ),

    on="factor_id",

    how="left",

    validate="one_to_one",
)


failure_factor_complete = failure_factor.merge(

    failure_dominant.drop(
        columns=[
            "factor_name",
            "evidence_population",
        ]
    ),

    on="factor_id",

    how="left",

    validate="one_to_one",
)


# ==========================================================
# ADD DOMINANT FEATURE SHARE WITHIN FACTOR
# ==========================================================

population_factor_complete[
    "dominant_feature_share_of_factor"
] = (

    population_factor_complete[
        "dominant_feature_mean_abs_shap"
    ]

    /

    population_factor_complete[
        "factor_total_mean_abs_shap"
    ]
)


population_factor_complete[
    "dominant_feature_share_of_factor_pct"
] = (

    population_factor_complete[
        "dominant_feature_share_of_factor"
    ]

    *
    100
)


failure_factor_complete[
    "dominant_feature_share_of_factor"
] = (

    failure_factor_complete[
        "dominant_feature_mean_abs_shap"
    ]

    /

    failure_factor_complete[
        "factor_total_mean_abs_shap"
    ]
)


failure_factor_complete[
    "dominant_feature_share_of_factor_pct"
] = (

    failure_factor_complete[
        "dominant_feature_share_of_factor"
    ]

    *
    100
)


# ==========================================================
# FACTOR-LEVEL CONSENSUS VIEW
#
# We do NOT average raw SHAP values between the two cohorts.
#
# Instead, use mean normalized share and mean rank only
# as a descriptive stability summary.
# ==========================================================

consensus = factor_stability.copy()


consensus[
    "mean_normalized_shap_share_pct"
] = (

    consensus[
        [
            "population_shap_share_pct",

            "failure_shap_share_pct",
        ]
    ]
    .mean(
        axis=1
    )
)


consensus[
    "mean_factor_rank"
] = (

    consensus[
        [
            "population_factor_rank",

            "failure_factor_rank",
        ]
    ]
    .mean(
        axis=1
    )
)


consensus[
    "rank_stability_label"
] = np.select(

    [

        consensus[
            "abs_rank_difference"
        ] <= 1,

        consensus[
            "abs_rank_difference"
        ] <= 3,
    ],

    [

        "HIGH",

        "MODERATE",
    ],

    default="LOW",
)


consensus = (

    consensus

    .sort_values(
        [
            "mean_factor_rank",

            "mean_normalized_shap_share_pct",
        ],

        ascending=[
            True,
            False,
        ]
    )

    .reset_index(
        drop=True
    )
)


consensus[
    "consensus_rank"
] = (

    np.arange(
        1,
        len(
            consensus
        )
        + 1
    )
)


# ==========================================================
# SAVE FEATURE-LEVEL MAPPED TABLES
# ==========================================================

population_feature_file = (
    OUTPUT_DIR
    / "population_shap_features_mapped.csv"
)


failure_feature_file = (
    OUTPUT_DIR
    / "failure_focused_shap_features_mapped.csv"
)


population_mapped.to_csv(
    population_feature_file,
    index=False
)


failure_mapped.to_csv(
    failure_feature_file,
    index=False
)


# ==========================================================
# SAVE FACTOR-LEVEL RESULTS
# ==========================================================

population_factor_file = (
    OUTPUT_DIR
    / "population_factor_shap_importance.csv"
)


failure_factor_file = (
    OUTPUT_DIR
    / "failure_focused_factor_shap_importance.csv"
)


population_factor_complete.to_csv(
    population_factor_file,
    index=False
)


failure_factor_complete.to_csv(
    failure_factor_file,
    index=False
)


# ==========================================================
# SAVE REPRESENTATION PROFILES
# ==========================================================

representation_profile_df = pd.concat(
    [

        population_representation,

        failure_representation,
    ],

    ignore_index=True
)


representation_file = (
    OUTPUT_DIR
    / "factor_shap_representation_profile.csv"
)


representation_profile_df.to_csv(
    representation_file,
    index=False
)


# ==========================================================
# SAVE TEMPORAL WINDOW PROFILES
# ==========================================================

window_profile_df = pd.concat(
    [

        population_window,

        failure_window,
    ],

    ignore_index=True
)


window_file = (
    OUTPUT_DIR
    / "factor_shap_temporal_window_profile.csv"
)


window_profile_df.to_csv(
    window_file,
    index=False
)


# ==========================================================
# SAVE STABILITY
# ==========================================================

stability_file = (
    OUTPUT_DIR
    / "population_vs_failure_factor_stability.csv"
)


factor_stability.to_csv(
    stability_file,
    index=False
)


overlap_file = (
    OUTPUT_DIR
    / "factor_topk_overlap.csv"
)


top_overlap_df.to_csv(
    overlap_file,
    index=False
)


consensus_file = (
    OUTPUT_DIR
    / "global_shap_factor_consensus.csv"
)


consensus.to_csv(
    consensus_file,
    index=False
)


# ==========================================================
# SAVE SUMMARY METRICS
# ==========================================================

summary_metrics_df = pd.DataFrame(
    [
        {
            "metric":
                "factor_count",

            "value":
                len(
                    factor_stability
                ),
        },

        {
            "metric":
                "population_feature_count",

            "value":
                len(
                    population_mapped
                ),
        },

        {
            "metric":
                "failure_feature_count",

            "value":
                len(
                    failure_mapped
                ),
        },

        {
            "metric":
                "factor_rank_spearman",

            "value":
                factor_rank_spearman,
        },
    ]
)


summary_metrics_file = (
    OUTPUT_DIR
    / "global_shap_factor_summary_metrics.csv"
)


summary_metrics_df.to_csv(
    summary_metrics_file,
    index=False
)


# ==========================================================
# PRINT RESULTS
# ==========================================================

print("\n")
print("=" * 80)

print(
    "STEP 13C - GLOBAL SHAP FACTOR AGGREGATION COMPLETE"
)

print("=" * 80)


print(
    "\nPopulation factor ranking:\n"
)


print(
    population_factor_complete[
        [
            "factor_rank",

            "factor_id",

            "factor_name",

            "factor_shap_share_pct",

            "dominant_feature",

            "dominant_representation_type",

            "dominant_window_days",

            "dominant_feature_global_rank",
        ]
    ]
    .head(
        16
    )
    .to_string(
        index=False
    )
)


print(
    "\nFailure-focused factor ranking:\n"
)


print(
    failure_factor_complete[
        [
            "factor_rank",

            "factor_id",

            "factor_name",

            "factor_shap_share_pct",

            "dominant_feature",

            "dominant_representation_type",

            "dominant_window_days",

            "dominant_feature_global_rank",
        ]
    ]
    .head(
        16
    )
    .to_string(
        index=False
    )
)


print(
    "\nFactor-rank stability:"
)


print(
    "Spearman correlation:",
    round(
        factor_rank_spearman,
        6
    )
)


print(
    "\nTop-k overlap:\n"
)


print(
    top_overlap_df.to_string(
        index=False
    )
)


print(
    "\nConsensus factor ranking:\n"
)


print(
    consensus[
        [
            "consensus_rank",

            "factor_id",

            "factor_name",

            "population_factor_rank",

            "failure_factor_rank",

            "abs_rank_difference",

            "mean_normalized_shap_share_pct",

            "rank_stability_label",
        ]
    ]
    .head(
        16
    )
    .to_string(
        index=False
    )
)


print(
    "\nOutputs saved to:"
)


print(
    OUTPUT_DIR
)