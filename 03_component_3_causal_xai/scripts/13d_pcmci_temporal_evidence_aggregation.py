from pathlib import Path
import json

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
    / "pcmci_temporal"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# INPUT FILES
# ==========================================================

STABLE_EXACT_FILE = (
    RESULTS_ROOT
    / "causal_discovery"
    / "pcmci_stability"
    / "stable_exact_links.csv"
)

STABLE_PAIR_FILE = (
    RESULTS_ROOT
    / "causal_discovery"
    / "pcmci_stability"
    / "stable_pair_links.csv"
)

EXACT_STABILITY_FILE = (
    RESULTS_ROOT
    / "causal_discovery"
    / "pcmci_stability"
    / "exact_link_stability.csv"
)

PAIR_STABILITY_FILE = (
    RESULTS_ROOT
    / "causal_discovery"
    / "pcmci_stability"
    / "pair_level_stability.csv"
)

PCMCI_MAPPING_FILE = (
    INTEGRATED_DIR
    / "pcmci_variable_to_factor_mapping.csv"
)

FACTOR_MANIFEST_FILE = (
    INTEGRATED_DIR
    / "canonical_factor_manifest.csv"
)


# ==========================================================
# VALIDATE INPUT FILES
# ==========================================================

required_files = [

    STABLE_EXACT_FILE,

    STABLE_PAIR_FILE,

    EXACT_STABILITY_FILE,

    PAIR_STABILITY_FILE,

    PCMCI_MAPPING_FILE,

    FACTOR_MANIFEST_FILE,
]


missing_files = [

    str(
        file
    )

    for file in required_files

    if not file.exists()
]


if missing_files:

    raise FileNotFoundError(
        "Missing required files:\n"
        +
        "\n".join(
            missing_files
        )
    )


# ==========================================================
# LOAD FILES
# ==========================================================

stable_exact = pd.read_csv(
    STABLE_EXACT_FILE
)


stable_pairs = pd.read_csv(
    STABLE_PAIR_FILE
)


exact_all = pd.read_csv(
    EXACT_STABILITY_FILE
)


pair_all = pd.read_csv(
    PAIR_STABILITY_FILE
)


pcmci_mapping = pd.read_csv(
    PCMCI_MAPPING_FILE
)


factor_manifest = pd.read_csv(
    FACTOR_MANIFEST_FILE
)


# ==========================================================
# EXPECTED COLUMNS
# ==========================================================

EXPECTED_EXACT_COLUMNS = {

    "source",

    "target",

    "lag_days",

    "runs_detected",

    "detection_rate",

    "positive_runs",

    "negative_runs",

    "sign_consistency",

    "mean_mci",

    "median_mci",

    "median_abs_mci",

    "max_abs_mci",
}


EXPECTED_PAIR_COLUMNS = {

    "source",

    "target",

    "runs_detected",

    "detection_rate",

    "modal_lag_days",

    "min_lag_days",

    "max_lag_days",

    "positive_runs",

    "negative_runs",

    "sign_consistency",

    "median_mci",

    "median_abs_mci",
}


missing_exact_columns = (

    EXPECTED_EXACT_COLUMNS

    -

    set(
        stable_exact.columns
    )
)


missing_pair_columns = (

    EXPECTED_PAIR_COLUMNS

    -

    set(
        stable_pairs.columns
    )
)


if missing_exact_columns:

    raise RuntimeError(
        "Stable exact-link file missing columns: "
        f"{sorted(missing_exact_columns)}"
    )


if missing_pair_columns:

    raise RuntimeError(
        "Stable pair-link file missing columns: "
        f"{sorted(missing_pair_columns)}"
    )


# ==========================================================
# MAPPING VALIDATION
# ==========================================================

required_mapping_columns = {

    "pcmci_variable",

    "mapped",

    "factor_id",

    "factor_name",

    "category",
}


missing_mapping_columns = (

    required_mapping_columns

    -

    set(
        pcmci_mapping.columns
    )
)


if missing_mapping_columns:

    raise RuntimeError(
        "PCMCI mapping file missing columns: "
        f"{sorted(missing_mapping_columns)}"
    )


pcmci_mapping = pcmci_mapping[
    pcmci_mapping[
        "mapped"
    ] == True
].copy()


# ==========================================================
# CREATE PCMCI LOOKUP
# ==========================================================

variable_lookup = (

    pcmci_mapping

    .set_index(
        "pcmci_variable"
    )[
        [
            "factor_id",

            "factor_name",

            "category",
        ]
    ]

    .to_dict(
        orient="index"
    )
)


# ==========================================================
# FIND ALL VARIABLES USED BY PCMCI
# ==========================================================

all_pcmci_variables = sorted(

    set(
        stable_exact[
            "source"
        ].astype(str)
    )

    |

    set(
        stable_exact[
            "target"
        ].astype(str)
    )

    |

    set(
        stable_pairs[
            "source"
        ].astype(str)
    )

    |

    set(
        stable_pairs[
            "target"
        ].astype(str)
    )
)


unmapped_variables = [

    variable

    for variable in all_pcmci_variables

    if variable
    not in variable_lookup
]


if unmapped_variables:

    raise RuntimeError(
        "Unmapped PCMCI variables detected:\n"
        +
        "\n".join(
            unmapped_variables
        )
    )


# ==========================================================
# HELPER:
# MAP SOURCE/TARGET TO CANONICAL FACTORS
# ==========================================================

def map_pcmci_links(
    frame
):

    mapped = frame.copy()


    mapped[
        "source_factor_id"
    ] = mapped[
        "source"
    ].map(
        lambda variable:
            variable_lookup[
                variable
            ][
                "factor_id"
            ]
    )


    mapped[
        "source_factor_name"
    ] = mapped[
        "source"
    ].map(
        lambda variable:
            variable_lookup[
                variable
            ][
                "factor_name"
            ]
    )


    mapped[
        "source_category"
    ] = mapped[
        "source"
    ].map(
        lambda variable:
            variable_lookup[
                variable
            ][
                "category"
            ]
    )


    mapped[
        "target_factor_id"
    ] = mapped[
        "target"
    ].map(
        lambda variable:
            variable_lookup[
                variable
            ][
                "factor_id"
            ]
    )


    mapped[
        "target_factor_name"
    ] = mapped[
        "target"
    ].map(
        lambda variable:
            variable_lookup[
                variable
            ][
                "factor_name"
            ]
    )


    mapped[
        "target_category"
    ] = mapped[
        "target"
    ].map(
        lambda variable:
            variable_lookup[
                variable
            ][
                "category"
            ]
    )


    mapped[
        "relation_type"
    ] = np.where(

        mapped[
            "source_factor_id"
        ]
        ==
        mapped[
            "target_factor_id"
        ],

        "SELF_TEMPORAL",

        "CROSS_FACTOR",
    )


    return mapped


# ==========================================================
# MAP STABLE LINKS
# ==========================================================

stable_exact_mapped = map_pcmci_links(
    stable_exact
)


stable_pairs_mapped = map_pcmci_links(
    stable_pairs
)


# ==========================================================
# NUMERIC SAFETY
# ==========================================================

exact_numeric_columns = [

    "lag_days",

    "runs_detected",

    "detection_rate",

    "positive_runs",

    "negative_runs",

    "sign_consistency",

    "mean_mci",

    "median_mci",

    "median_abs_mci",

    "max_abs_mci",
]


pair_numeric_columns = [

    "runs_detected",

    "detection_rate",

    "modal_lag_days",

    "min_lag_days",

    "max_lag_days",

    "positive_runs",

    "negative_runs",

    "sign_consistency",

    "median_mci",

    "median_abs_mci",
]


for column in exact_numeric_columns:

    stable_exact_mapped[
        column
    ] = pd.to_numeric(
        stable_exact_mapped[
            column
        ],
        errors="raise"
    )


for column in pair_numeric_columns:

    stable_pairs_mapped[
        column
    ] = pd.to_numeric(
        stable_pairs_mapped[
            column
        ],
        errors="raise"
    )


# ==========================================================
# CONDITIONAL-DEPENDENCE SIGN
#
# IMPORTANT:
# Positive/negative refers to PCMCI MCI coefficient sign.
# It is NOT a positive/negative causal effect.
# ==========================================================

def dependency_sign(
    value
):

    if value > 0:

        return "POSITIVE_CONDITIONAL_DEPENDENCE"

    if value < 0:

        return "NEGATIVE_CONDITIONAL_DEPENDENCE"

    return "ZERO_OR_NEUTRAL"


stable_exact_mapped[
    "dependency_sign"
] = stable_exact_mapped[
    "median_mci"
].apply(
    dependency_sign
)


stable_pairs_mapped[
    "dependency_sign"
] = stable_pairs_mapped[
    "median_mci"
].apply(
    dependency_sign
)


# ==========================================================
# EXACT-LINK STRENGTH RANK
#
# This ranks MCI magnitude ONLY within the stable PCMCI
# results. It is not a causal-effect score.
# ==========================================================

stable_exact_mapped[
    "temporal_strength_rank"
] = (

    stable_exact_mapped[
        "median_abs_mci"
    ]

    .rank(
        method="min",
        ascending=False
    )

    .astype(
        int
    )
)


stable_pairs_mapped[
    "temporal_strength_rank"
] = (

    stable_pairs_mapped[
        "median_abs_mci"
    ]

    .rank(
        method="min",
        ascending=False
    )

    .astype(
        int
    )
)


# ==========================================================
# SPLIT SELF AND CROSS-FACTOR LINKS
# ==========================================================

exact_self = stable_exact_mapped[
    stable_exact_mapped[
        "relation_type"
    ]
    ==
    "SELF_TEMPORAL"
].copy()


exact_cross = stable_exact_mapped[
    stable_exact_mapped[
        "relation_type"
    ]
    ==
    "CROSS_FACTOR"
].copy()


pair_self = stable_pairs_mapped[
    stable_pairs_mapped[
        "relation_type"
    ]
    ==
    "SELF_TEMPORAL"
].copy()


pair_cross = stable_pairs_mapped[
    stable_pairs_mapped[
        "relation_type"
    ]
    ==
    "CROSS_FACTOR"
].copy()


# ==========================================================
# AGGREGATE STABLE EXACT LINKS BY CANONICAL FACTOR PAIR
#
# Multiple lags may exist between the same source/target.
# ==========================================================

factor_pair_rows = []


for (
    source_factor_id,
    target_factor_id
), group in exact_cross.groupby(
    [
        "source_factor_id",
        "target_factor_id",
    ]
):

    strongest_index = (
        group[
            "median_abs_mci"
        ]
        .idxmax()
    )


    strongest = group.loc[
        strongest_index
    ]


    lag_values = sorted(
        group[
            "lag_days"
        ]
        .astype(
            int
        )
        .unique()
        .tolist()
    )


    signs = set(
        group[
            "dependency_sign"
        ]
    )


    if len(
        signs
    ) == 1:

        lag_sign_pattern = list(
            signs
        )[0]

    else:

        lag_sign_pattern = (
            "MIXED_ACROSS_LAGS"
        )


    factor_pair_rows.append(
        {
            "source_factor_id":
                source_factor_id,

            "source_factor_name":
                strongest[
                    "source_factor_name"
                ],

            "source_category":
                strongest[
                    "source_category"
                ],

            "target_factor_id":
                target_factor_id,

            "target_factor_name":
                strongest[
                    "target_factor_name"
                ],

            "target_category":
                strongest[
                    "target_category"
                ],

            "stable_exact_lag_count":
                len(
                    group
                ),

            "stable_lags_days":
                ";".join(
                    str(
                        value
                    )

                    for value in lag_values
                ),

            "minimum_stable_lag_days":
                min(
                    lag_values
                ),

            "maximum_stable_lag_days":
                max(
                    lag_values
                ),

            "strongest_lag_days":
                int(
                    strongest[
                        "lag_days"
                    ]
                ),

            "strongest_median_mci":
                float(
                    strongest[
                        "median_mci"
                    ]
                ),

            "strongest_median_abs_mci":
                float(
                    strongest[
                        "median_abs_mci"
                    ]
                ),

            "strongest_max_abs_mci":
                float(
                    strongest[
                        "max_abs_mci"
                    ]
                ),

            "strongest_detection_rate":
                float(
                    strongest[
                        "detection_rate"
                    ]
                ),

            "strongest_sign_consistency":
                float(
                    strongest[
                        "sign_consistency"
                    ]
                ),

            "minimum_detection_rate_across_stable_lags":
                float(
                    group[
                        "detection_rate"
                    ].min()
                ),

            "mean_detection_rate_across_stable_lags":
                float(
                    group[
                        "detection_rate"
                    ].mean()
                ),

            "minimum_sign_consistency_across_stable_lags":
                float(
                    group[
                        "sign_consistency"
                    ].min()
                ),

            "mean_sign_consistency_across_stable_lags":
                float(
                    group[
                        "sign_consistency"
                    ].mean()
                ),

            "lag_sign_pattern":
                lag_sign_pattern,
        }
    )


factor_pair_exact_df = pd.DataFrame(
    factor_pair_rows
)


# ==========================================================
# AGGREGATE STABLE PAIR-LEVEL EVIDENCE
#
# Normally one PCMCI variable maps to one canonical factor.
# Aggregation also makes the script robust if multiple
# variables later map to one factor.
# ==========================================================

pair_factor_rows = []


for (
    source_factor_id,
    target_factor_id
), group in pair_cross.groupby(
    [
        "source_factor_id",
        "target_factor_id",
    ]
):

    strongest_index = (
        group[
            "median_abs_mci"
        ]
        .idxmax()
    )


    strongest = group.loc[
        strongest_index
    ]


    pair_factor_rows.append(
        {
            "source_factor_id":
                source_factor_id,

            "source_factor_name":
                strongest[
                    "source_factor_name"
                ],

            "target_factor_id":
                target_factor_id,

            "target_factor_name":
                strongest[
                    "target_factor_name"
                ],

            "variable_pair_count":
                len(
                    group
                ),

            "pair_detection_rate":
                float(
                    group[
                        "detection_rate"
                    ].max()
                ),

            "pair_sign_consistency":
                float(
                    group[
                        "sign_consistency"
                    ].max()
                ),

            "pair_median_mci":
                float(
                    strongest[
                        "median_mci"
                    ]
                ),

            "pair_median_abs_mci":
                float(
                    strongest[
                        "median_abs_mci"
                    ]
                ),

            "pair_dependency_sign":
                strongest[
                    "dependency_sign"
                ],

            "modal_lag_days":
                int(
                    strongest[
                        "modal_lag_days"
                    ]
                ),

            "minimum_lag_days":
                int(
                    group[
                        "min_lag_days"
                    ].min()
                ),

            "maximum_lag_days":
                int(
                    group[
                        "max_lag_days"
                    ].max()
                ),
        }
    )


factor_pair_stable_df = pd.DataFrame(
    pair_factor_rows
)


# ==========================================================
# MERGE EXACT-LAG AND PAIR-LEVEL TEMPORAL EVIDENCE
# ==========================================================

if len(
    factor_pair_exact_df
) > 0:

    factor_pair_temporal = (
        factor_pair_exact_df.merge(

            factor_pair_stable_df,

            on=[
                "source_factor_id",

                "target_factor_id",
            ],

            how="outer",

            suffixes=(
                "_exact",
                "_pair"
            ),
        )
    )

else:

    factor_pair_temporal = (
        factor_pair_stable_df.copy()
    )


# ==========================================================
# CLEAN DUPLICATE FACTOR-NAME COLUMNS AFTER MERGE
# ==========================================================

if (
    "source_factor_name_exact"
    in factor_pair_temporal.columns
):

    factor_pair_temporal[
        "source_factor_name"
    ] = (

        factor_pair_temporal[
            "source_factor_name_exact"
        ]
        .combine_first(
            factor_pair_temporal[
                "source_factor_name_pair"
            ]
        )
    )


    factor_pair_temporal[
        "target_factor_name"
    ] = (

        factor_pair_temporal[
            "target_factor_name_exact"
        ]
        .combine_first(
            factor_pair_temporal[
                "target_factor_name_pair"
            ]
        )
    )


    factor_pair_temporal = (
        factor_pair_temporal.drop(
            columns=[
                "source_factor_name_exact",

                "source_factor_name_pair",

                "target_factor_name_exact",

                "target_factor_name_pair",
            ]
        )
    )


# ==========================================================
# TEMPORAL EVIDENCE AVAILABILITY FLAGS
# ==========================================================

factor_pair_temporal[
    "stable_exact_evidence"
] = (

    factor_pair_temporal[
        "stable_exact_lag_count"
    ]
    .notna()
)


factor_pair_temporal[
    "stable_pair_evidence"
] = (

    factor_pair_temporal[
        "pair_detection_rate"
    ]
    .notna()
)


# ==========================================================
# PRIMARY TEMPORAL MAGNITUDE
#
# Prefer pair-level median |MCI| when available.
# Otherwise use strongest exact-lag median |MCI|.
#
# This is for ordering/visualization only.
# ==========================================================

factor_pair_temporal[
    "temporal_magnitude"
] = (

    factor_pair_temporal[
        "pair_median_abs_mci"
    ]
    .combine_first(
        factor_pair_temporal[
            "strongest_median_abs_mci"
        ]
    )
)


factor_pair_temporal[
    "temporal_direction"
] = (

    factor_pair_temporal[
        "pair_dependency_sign"
    ]
    .combine_first(
        factor_pair_temporal[
            "lag_sign_pattern"
        ]
    )
)


# ==========================================================
# TEMPORAL PAIR RANK
# ==========================================================

factor_pair_temporal[
    "temporal_pair_rank"
] = (

    factor_pair_temporal[
        "temporal_magnitude"
    ]

    .rank(
        method="min",
        ascending=False
    )

    .astype(
        int
    )
)


factor_pair_temporal = (
    factor_pair_temporal
    .sort_values(
        [
            "temporal_pair_rank",

            "source_factor_id",

            "target_factor_id",
        ]
    )
    .reset_index(
        drop=True
    )
)


# ==========================================================
# FACTOR ROLE SUMMARY
#
# Incoming:
# how many stable factors temporally precede this factor.
#
# Outgoing:
# how many stable factors this factor temporally precedes.
#
# This is graph structure only, not causal-effect strength.
# ==========================================================

all_factor_ids = set(
    factor_manifest[
        "factor_id"
    ].astype(str)
)


role_rows = []


for factor_id in sorted(
    all_factor_ids
):

    factor_info = factor_manifest[
        factor_manifest[
            "factor_id"
        ]
        ==
        factor_id
    ]


    if factor_info.empty:

        continue


    factor_info = factor_info.iloc[
        0
    ]


    outgoing = factor_pair_temporal[
        factor_pair_temporal[
            "source_factor_id"
        ]
        ==
        factor_id
    ]


    incoming = factor_pair_temporal[
        factor_pair_temporal[
            "target_factor_id"
        ]
        ==
        factor_id
    ]


    exact_outgoing = exact_cross[
        exact_cross[
            "source_factor_id"
        ]
        ==
        factor_id
    ]


    exact_incoming = exact_cross[
        exact_cross[
            "target_factor_id"
        ]
        ==
        factor_id
    ]


    role_rows.append(
        {
            "factor_id":
                factor_id,

            "factor_name":
                factor_info[
                    "factor_name"
                ],

            "category":
                factor_info[
                    "category"
                ],

            "semantic_status":
                factor_info[
                    "semantic_status"
                ],

            "stable_outgoing_factor_pairs":
                outgoing[
                    "target_factor_id"
                ].nunique(),

            "stable_incoming_factor_pairs":
                incoming[
                    "source_factor_id"
                ].nunique(),

            "stable_cross_factor_degree":
                (
                    outgoing[
                        "target_factor_id"
                    ].nunique()

                    +

                    incoming[
                        "source_factor_id"
                    ].nunique()
                ),

            "stable_outgoing_exact_links":
                len(
                    exact_outgoing
                ),

            "stable_incoming_exact_links":
                len(
                    exact_incoming
                ),

            "maximum_outgoing_temporal_magnitude":
                (
                    outgoing[
                        "temporal_magnitude"
                    ].max()

                    if len(
                        outgoing
                    ) > 0

                    else np.nan
                ),

            "maximum_incoming_temporal_magnitude":
                (
                    incoming[
                        "temporal_magnitude"
                    ].max()

                    if len(
                        incoming
                    ) > 0

                    else np.nan
                ),
        }
    )


factor_role_df = pd.DataFrame(
    role_rows
)


factor_role_df = (
    factor_role_df
    .sort_values(
        [
            "stable_cross_factor_degree",

            "stable_outgoing_factor_pairs",
        ],
        ascending=[
            False,
            False,
        ]
    )
    .reset_index(
        drop=True
    )
)


# ==========================================================
# DEGRADATION-FOCUSED SUBGRAPH
#
# These are the three factors later used in matched
# failure-effect analysis.
#
# This does NOT merge ATT values yet.
# ==========================================================

DEGRADATION_FACTORS = {

    "REALLOCATED_SECTORS",

    "REPORTED_UNCORRECTABLE",

    "PENDING_SECTORS",
}


degradation_temporal = (
    factor_pair_temporal[
        (
            factor_pair_temporal[
                "source_factor_id"
            ].isin(
                DEGRADATION_FACTORS
            )
        )
        |
        (
            factor_pair_temporal[
                "target_factor_id"
            ].isin(
                DEGRADATION_FACTORS
            )
        )
    ]
    .copy()
)


# ==========================================================
# STRONGEST TEMPORAL RELATIONSHIP PER SOURCE FACTOR
# ==========================================================

if len(
    factor_pair_temporal
) > 0:

    strongest_per_source = (

        factor_pair_temporal

        .sort_values(
            [
                "source_factor_id",

                "temporal_magnitude",
            ],
            ascending=[
                True,
                False,
            ]
        )

        .groupby(
            "source_factor_id",
            as_index=False
        )

        .first()
    )

else:

    strongest_per_source = (
        pd.DataFrame()
    )


# ==========================================================
# STRONGEST TEMPORAL RELATIONSHIP INTO EACH TARGET FACTOR
# ==========================================================

if len(
    factor_pair_temporal
) > 0:

    strongest_per_target = (

        factor_pair_temporal

        .sort_values(
            [
                "target_factor_id",

                "temporal_magnitude",
            ],
            ascending=[
                True,
                False,
            ]
        )

        .groupby(
            "target_factor_id",
            as_index=False
        )

        .first()
    )

else:

    strongest_per_target = (
        pd.DataFrame()
    )


# ==========================================================
# EXACT-LINK SUMMARY
# ==========================================================

summary_rows = [

    {
        "metric":
            "stable_exact_links",

        "value":
            len(
                stable_exact_mapped
            ),
    },

    {
        "metric":
            "stable_exact_self_links",

        "value":
            len(
                exact_self
            ),
    },

    {
        "metric":
            "stable_exact_cross_factor_links",

        "value":
            len(
                exact_cross
            ),
    },

    {
        "metric":
            "stable_pair_links",

        "value":
            len(
                stable_pairs_mapped
            ),
    },

    {
        "metric":
            "stable_pair_self_links",

        "value":
            len(
                pair_self
            ),
    },

    {
        "metric":
            "stable_pair_cross_factor_links",

        "value":
            len(
                pair_cross
            ),
    },

    {
        "metric":
            "canonical_cross_factor_pairs",

        "value":
            len(
                factor_pair_temporal
            ),
    },

    {
        "metric":
            "pcmci_variables_in_stable_results",

        "value":
            len(
                all_pcmci_variables
            ),
    },
]


summary_df = pd.DataFrame(
    summary_rows
)


# ==========================================================
# SAVE OUTPUTS
# ==========================================================

stable_exact_output = (
    OUTPUT_DIR
    / "stable_exact_links_factor_mapped.csv"
)


stable_pairs_output = (
    OUTPUT_DIR
    / "stable_pair_links_factor_mapped.csv"
)


exact_self_output = (
    OUTPUT_DIR
    / "stable_self_temporal_exact_links.csv"
)


pair_self_output = (
    OUTPUT_DIR
    / "stable_self_temporal_pairs.csv"
)


factor_pair_output = (
    OUTPUT_DIR
    / "factor_pair_temporal_evidence.csv"
)


factor_role_output = (
    OUTPUT_DIR
    / "factor_temporal_role_summary.csv"
)


degradation_output = (
    OUTPUT_DIR
    / "degradation_factor_temporal_links.csv"
)


strongest_source_output = (
    OUTPUT_DIR
    / "strongest_temporal_link_per_source.csv"
)


strongest_target_output = (
    OUTPUT_DIR
    / "strongest_temporal_link_per_target.csv"
)


summary_output = (
    OUTPUT_DIR
    / "pcmci_temporal_summary_metrics.csv"
)


stable_exact_mapped.to_csv(
    stable_exact_output,
    index=False
)


stable_pairs_mapped.to_csv(
    stable_pairs_output,
    index=False
)


exact_self.to_csv(
    exact_self_output,
    index=False
)


pair_self.to_csv(
    pair_self_output,
    index=False
)


factor_pair_temporal.to_csv(
    factor_pair_output,
    index=False
)


factor_role_df.to_csv(
    factor_role_output,
    index=False
)


degradation_temporal.to_csv(
    degradation_output,
    index=False
)


strongest_per_source.to_csv(
    strongest_source_output,
    index=False
)


strongest_per_target.to_csv(
    strongest_target_output,
    index=False
)


summary_df.to_csv(
    summary_output,
    index=False
)


# ==========================================================
# METADATA
# ==========================================================

metadata = {

    "method":
        "PCMCI stability aggregation",

    "input_exact_links":
        str(
            STABLE_EXACT_FILE
        ),

    "input_pair_links":
        str(
            STABLE_PAIR_FILE
        ),

    "evidence_meaning":
        (
            "Stable lagged conditional-dependency evidence "
            "observed across repeated PCMCI subsamples."
        ),

    "causal_interpretation_warning":
        (
            "PCMCI temporal links are treated as candidate "
            "causal/dependency relationships under method "
            "assumptions. They are not interpreted as proof "
            "of physical causation."
        ),

    "strength_measure":
        "median absolute MCI",

    "direction_measure":
        "sign of median MCI",

    "degradation_factors":
        sorted(
            DEGRADATION_FACTORS
        ),
}


metadata_file = (
    OUTPUT_DIR
    / "pcmci_temporal_evidence_metadata.json"
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
# PRINT RESULTS
# ==========================================================

print("\n")
print("=" * 80)

print(
    "STEP 13D - PCMCI TEMPORAL EVIDENCE AGGREGATION COMPLETE"
)

print("=" * 80)


print(
    "\nPCMCI summary:\n"
)


print(
    summary_df.to_string(
        index=False
    )
)


# ==========================================================
# TOP CROSS-FACTOR TEMPORAL LINKS
# ==========================================================

print(
    "\nTop stable cross-factor temporal relationships:\n"
)


display_columns = [

    "temporal_pair_rank",

    "source_factor_name",

    "target_factor_name",

    "modal_lag_days",

    "temporal_direction",

    "pair_detection_rate",

    "pair_sign_consistency",

    "temporal_magnitude",
]


existing_display_columns = [

    column

    for column in display_columns

    if column
    in factor_pair_temporal.columns
]


print(
    factor_pair_temporal[
        existing_display_columns
    ]
    .head(
        20
    )
    .to_string(
        index=False
    )
)


# ==========================================================
# DEGRADATION-FOCUSED TEMPORAL LINKS
# ==========================================================

print(
    "\nDegradation-focused temporal relationships:\n"
)


if len(
    degradation_temporal
) > 0:

    degradation_columns = [

        "temporal_pair_rank",

        "source_factor_name",

        "target_factor_name",

        "modal_lag_days",

        "strongest_lag_days",

        "temporal_direction",

        "temporal_magnitude",

        "pair_detection_rate",

        "pair_sign_consistency",
    ]


    degradation_columns = [

        column

        for column in degradation_columns

        if column
        in degradation_temporal.columns
    ]


    print(
        degradation_temporal[
            degradation_columns
        ]
        .to_string(
            index=False
        )
    )

else:

    print(
        "No stable cross-factor temporal links involving "
        "the selected degradation factors."
    )


# ==========================================================
# FACTOR GRAPH ROLE SUMMARY
# ==========================================================

print(
    "\nFactor temporal-role summary:\n"
)


print(
    factor_role_df[
        [
            "factor_id",

            "factor_name",

            "stable_outgoing_factor_pairs",

            "stable_incoming_factor_pairs",

            "stable_cross_factor_degree",
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