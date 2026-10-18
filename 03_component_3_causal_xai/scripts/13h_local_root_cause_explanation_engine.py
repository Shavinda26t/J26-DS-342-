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

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
)

SHAP_LOCAL_DIR = (
    RESULTS_ROOT
    / "shap"
    / "30d"
    / "local"
)

INTEGRATED_INTERIM_DIR = (
    DATA_ROOT
    / "interim"
    / "integrated_evidence"
)

ROOT_CAUSE_DIR = (
    RESULTS_ROOT
    / "integrated_evidence"
    / "root_cause_matrix"
)

OUTPUT_DIR = (
    RESULTS_ROOT
    / "integrated_evidence"
    / "local_root_cause"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# INPUT FILES
# ==========================================================

SHAP_MAPPING_FILE = (
    INTEGRATED_INTERIM_DIR
    / "shap_feature_to_factor_mapping.csv"
)

ROOT_CAUSE_MATRIX_FILE = (
    ROOT_CAUSE_DIR
    / "integrated_root_cause_evidence_matrix.csv"
)


LOCAL_CASE_FILES = {

    "TP":
        SHAP_LOCAL_DIR
        / "TP_local_shap.csv",

    "FP":
        SHAP_LOCAL_DIR
        / "FP_local_shap.csv",

    "FN":
        SHAP_LOCAL_DIR
        / "FN_local_shap.csv",

    "TN":
        SHAP_LOCAL_DIR
        / "TN_local_shap.csv",
}


# ==========================================================
# VALIDATE REQUIRED FILES
# ==========================================================

required_files = [

    SHAP_MAPPING_FILE,

    ROOT_CAUSE_MATRIX_FILE,

    *LOCAL_CASE_FILES.values(),
]


missing_files = [

    str(path)

    for path in required_files

    if not path.exists()
]


if missing_files:

    raise FileNotFoundError(
        "Missing required Step 13H inputs:\n"
        +
        "\n".join(
            missing_files
        )
    )


# ==========================================================
# LOAD GLOBAL SUPPORT TABLES
# ==========================================================

mapping = pd.read_csv(
    SHAP_MAPPING_FILE
)

root_cause_matrix = pd.read_csv(
    ROOT_CAUSE_MATRIX_FILE
)


# ==========================================================
# REQUIRED MAPPING COLUMNS
# ==========================================================

required_mapping_columns = {

    "original_feature",

    "factor_id",

    "factor_name",

    "category",

    "semantic_status",

    "representation_type",

    "aggregation",

    "window_days",
}


missing_mapping_columns = (

    required_mapping_columns

    -

    set(
        mapping.columns
    )
)


if missing_mapping_columns:

    raise RuntimeError(
        "SHAP mapping missing columns: "
        f"{sorted(missing_mapping_columns)}"
    )


# ==========================================================
# REQUIRED ROOT-CAUSE MATRIX COLUMNS
# ==========================================================

required_root_columns = {

    "factor_id",

    "factor_name",

    "consensus_rank",

    "mean_normalized_shap_share_pct",

    "pcmci_available",

    "matched_att_available",

    "root_cause_candidate",

    "integrated_evidence_class",

    "stable_cross_factor_degree",

    "degradation_temporal_pathway_available",

    "degradation_temporal_pathways",

    "matched_effect_evidence_class",

    "failure_risk_difference_pp",

    "failure_risk_ci_lower_pp",

    "failure_risk_ci_upper_pp",
}


missing_root_columns = (

    required_root_columns

    -

    set(
        root_cause_matrix.columns
    )
)


if missing_root_columns:

    raise RuntimeError(
        "Root-cause matrix missing columns: "
        f"{sorted(missing_root_columns)}"
    )


# ==========================================================
# BOOLEAN HELPER
# ==========================================================

def as_bool(
    value
):

    if isinstance(
        value,
        bool
    ):

        return value


    if pd.isna(
        value
    ):

        return False


    return (
        str(value)
        .strip()
        .lower()
        in {
            "true",
            "1",
            "yes",
            "y",
        }
    )


# ==========================================================
# COLUMN DETECTION HELPER
#
# Makes Step 13H robust to slightly different column names
# in the local SHAP output files.
# ==========================================================

def detect_column(
    frame,
    candidates,
    required=True
):

    lower_lookup = {

        str(column).lower():
            column

        for column in frame.columns
    }


    for candidate in candidates:

        if candidate.lower() in lower_lookup:

            return lower_lookup[
                candidate.lower()
            ]


    if required:

        raise RuntimeError(
            "Could not detect any of these columns: "
            f"{candidates}\n"
            f"Available columns: "
            f"{list(frame.columns)}"
        )


    return None


# ==========================================================
# POSSIBLE LOCAL SHAP COLUMN NAMES
# ==========================================================

FEATURE_COLUMN_CANDIDATES = [

    "feature",

    "feature_name",

    "variable",

    "input_feature",
]


SHAP_COLUMN_CANDIDATES = [

    "shap_value",

    "shap",

    "local_shap_value",

    "shap_contribution",

    "contribution",
]


FEATURE_VALUE_COLUMN_CANDIDATES = [

    "feature_value",

    "value",

    "observed_value",

    "input_value",
]


RANK_COLUMN_CANDIDATES = [

    "rank",

    "shap_rank",

    "importance_rank",
]


SERIAL_COLUMN_CANDIDATES = [

    "serial_number",

    "serial",

    "hdd_serial",

    "drive_serial",
]


DATE_COLUMN_CANDIDATES = [

    "date",

    "observation_date",

    "prediction_date",
]


PROBABILITY_COLUMN_CANDIDATES = [

    "predicted_probability",

    "prediction_probability",

    "failure_probability",

    "probability",

    "pred_proba",
]


ACTUAL_LABEL_COLUMN_CANDIDATES = [

    "actual_label",

    "label",

    "y_true",

    "actual",
]


PREDICTED_LABEL_COLUMN_CANDIDATES = [

    "predicted_label",

    "prediction",

    "y_pred",

    "predicted_class",
]


THRESHOLD_COLUMN_CANDIDATES = [

    "threshold",

    "decision_threshold",

    "classification_threshold",
]


# ==========================================================
# LOAD MAPPING LOOKUP
# ==========================================================

mapping_lookup = (

    mapping

    .set_index(
        "original_feature"
    )

    .to_dict(
        orient="index"
    )
)


# ==========================================================
# LOAD ROOT-CAUSE LOOKUP
# ==========================================================

root_lookup = (

    root_cause_matrix

    .set_index(
        "factor_id"
    )

    .to_dict(
        orient="index"
    )
)


# ==========================================================
# HELPER:
# Extract one repeated metadata value from a SHAP table.
# ==========================================================

def extract_metadata_value(
    frame,
    column
):

    if column is None:

        return None


    values = (

        frame[
            column
        ]

        .dropna()

        .unique()
    )


    if len(
        values
    ) == 0:

        return None


    return values[
        0
    ]


# ==========================================================
# LOCAL SHAP DIRECTION
#
# Importance:
#   sum absolute SHAP over all temporal representations.
#
# Direction:
#   based on the relative absolute mass of positive and
#   negative SHAP values.
#
# This prevents simple signed cancellation from giving a
# misleading factor interpretation.
# ==========================================================

def determine_local_direction(
    group
):

    positive_mass = (

        group.loc[
            group[
                "shap_value"
            ] > 0,
            "abs_shap"
        ].sum()
    )


    negative_mass = (

        group.loc[
            group[
                "shap_value"
            ] < 0,
            "abs_shap"
        ].sum()
    )


    total_mass = (

        positive_mass

        +

        negative_mass
    )


    if total_mass <= 0:

        return (
            "NEUTRAL",
            0.0,
            0.0
        )


    positive_share = (

        positive_mass

        /

        total_mass
    )


    negative_share = (

        negative_mass

        /

        total_mass
    )


    if positive_share >= 0.65:

        direction = (
            "TOWARD_FAILURE_OUTPUT"
        )


    elif negative_share >= 0.65:

        direction = (
            "AWAY_FROM_FAILURE_OUTPUT"
        )


    else:

        direction = (
            "MIXED_LOCAL_EFFECT"
        )


    return (

        direction,

        positive_share,

        negative_share
    )


# ==========================================================
# LOCAL / FLEET EVIDENCE ALIGNMENT
# ==========================================================

def classify_alignment(
    row
):

    direction = row[
        "local_direction"
    ]


    integrated_class = row[
        "integrated_evidence_class"
    ]


    root_candidate = as_bool(
        row[
            "root_cause_candidate"
        ]
    )


    if root_candidate:

        if direction == (
            "TOWARD_FAILURE_OUTPUT"
        ):

            return (
                "LOCAL_PREDICTIVE_SUPPORT_ALIGNS_"
                "WITH_FLEET_ROOT_CAUSE_EVIDENCE"
            )


        if direction == (
            "AWAY_FROM_FAILURE_OUTPUT"
        ):

            return (
                "FLEET_ROOT_CAUSE_CANDIDATE_"
                "NOT_LOCALLY_RISK_INCREASING"
            )


        return (
            "FLEET_ROOT_CAUSE_CANDIDATE_"
            "WITH_MIXED_LOCAL_ATTRIBUTION"
        )


    if integrated_class == (
        "PREDICTIVE_TEMPORAL_CONTEXT"
    ):

        if direction == (
            "TOWARD_FAILURE_OUTPUT"
        ):

            return (
                "LOCAL_PREDICTIVE_SUPPORT_"
                "WITH_TEMPORAL_CONTEXT"
            )


        return (
            "TEMPORAL_CONTEXT_"
            "WITHOUT_LOCAL_RISK_INCREASE"
        )


    if (
        "SEMANTIC_REVIEW"
        in str(
            integrated_class
        )
    ):

        return (
            "LOCAL_SIGNAL_REQUIRES_"
            "SEMANTIC_REVIEW"
        )


    return (
        "LOCAL_PREDICTIVE_SIGNAL_ONLY"
    )


# ==========================================================
# CASE TYPE DESCRIPTION
# ==========================================================

CASE_DESCRIPTIONS = {

    "TP":
        (
            "True Positive: the HDD was predicted as "
            "failure-risk and the evaluation label was positive."
        ),

    "FP":
        (
            "False Positive: the HDD was predicted as "
            "failure-risk but the evaluation label was negative."
        ),

    "FN":
        (
            "False Negative: the HDD had a positive evaluation "
            "label but the model did not classify it as failure-risk."
        ),

    "TN":
        (
            "True Negative: the HDD was classified as lower-risk "
            "and the evaluation label was negative."
        ),
}


# ==========================================================
# PROCESS ONE LOCAL CASE
# ==========================================================

def process_local_case(
    case_type,
    file_path
):

    local = pd.read_csv(
        file_path
    )


    # ------------------------------------------------------
    # Detect columns
    # ------------------------------------------------------

    feature_column = detect_column(
        local,
        FEATURE_COLUMN_CANDIDATES,
        required=True
    )


    shap_column = detect_column(
        local,
        SHAP_COLUMN_CANDIDATES,
        required=True
    )


    feature_value_column = detect_column(
        local,
        FEATURE_VALUE_COLUMN_CANDIDATES,
        required=False
    )


    rank_column = detect_column(
        local,
        RANK_COLUMN_CANDIDATES,
        required=False
    )


    serial_column = detect_column(
        local,
        SERIAL_COLUMN_CANDIDATES,
        required=False
    )


    date_column = detect_column(
        local,
        DATE_COLUMN_CANDIDATES,
        required=False
    )


    probability_column = detect_column(
        local,
        PROBABILITY_COLUMN_CANDIDATES,
        required=False
    )


    actual_label_column = detect_column(
        local,
        ACTUAL_LABEL_COLUMN_CANDIDATES,
        required=False
    )


    predicted_label_column = detect_column(
        local,
        PREDICTED_LABEL_COLUMN_CANDIDATES,
        required=False
    )


    threshold_column = detect_column(
        local,
        THRESHOLD_COLUMN_CANDIDATES,
        required=False
    )


    # ------------------------------------------------------
    # Standardize core columns
    # ------------------------------------------------------

    standardized = pd.DataFrame(
        {
            "feature":
                local[
                    feature_column
                ].astype(str),

            "shap_value":
                pd.to_numeric(
                    local[
                        shap_column
                    ],
                    errors="raise"
                ),
        }
    )


    standardized[
        "abs_shap"
    ] = standardized[
        "shap_value"
    ].abs()


    if feature_value_column is not None:

        standardized[
            "feature_value"
        ] = local[
            feature_value_column
        ]


    else:

        standardized[
            "feature_value"
        ] = np.nan


    if rank_column is not None:

        standardized[
            "original_local_rank"
        ] = local[
            rank_column
        ]


    else:

        standardized[
            "original_local_rank"
        ] = np.nan


    # ------------------------------------------------------
    # Ensure all features can be mapped
    # ------------------------------------------------------

    unmapped_features = sorted(

        set(
            standardized[
                "feature"
            ]
        )

        -

        set(
            mapping_lookup.keys()
        )
    )


    if unmapped_features:

        raise RuntimeError(
            f"{case_type}: local SHAP contains unmapped "
            f"features:\n"
            +
            "\n".join(
                unmapped_features
            )
        )


    # ------------------------------------------------------
    # Add canonical factor information
    # ------------------------------------------------------

    standardized[
        "factor_id"
    ] = standardized[
        "feature"
    ].map(
        lambda feature:
            mapping_lookup[
                feature
            ][
                "factor_id"
            ]
    )


    standardized[
        "factor_name"
    ] = standardized[
        "feature"
    ].map(
        lambda feature:
            mapping_lookup[
                feature
            ][
                "factor_name"
            ]
    )


    standardized[
        "representation_type"
    ] = standardized[
        "feature"
    ].map(
        lambda feature:
            mapping_lookup[
                feature
            ][
                "representation_type"
            ]
    )


    standardized[
        "aggregation"
    ] = standardized[
        "feature"
    ].map(
        lambda feature:
            mapping_lookup[
                feature
            ][
                "aggregation"
            ]
    )


    standardized[
        "window_days"
    ] = standardized[
        "feature"
    ].map(
        lambda feature:
            mapping_lookup[
                feature
            ][
                "window_days"
            ]
    )


    standardized[
        "semantic_status"
    ] = standardized[
        "feature"
    ].map(
        lambda feature:
            mapping_lookup[
                feature
            ][
                "semantic_status"
            ]
    )


    standardized[
        "case_type"
    ] = case_type


    # ------------------------------------------------------
    # Feature-level rank by local absolute SHAP
    # ------------------------------------------------------

    standardized[
        "local_feature_rank"
    ] = (

        standardized[
            "abs_shap"
        ]

        .rank(
            method="min",
            ascending=False
        )

        .astype(int)
    )


    standardized = (

        standardized

        .sort_values(
            [
                "local_feature_rank",
                "feature",
            ]
        )

        .reset_index(
            drop=True
        )
    )


    # ------------------------------------------------------
    # Factor-level aggregation
    # ------------------------------------------------------

    factor_rows = []


    for factor_id, group in standardized.groupby(
        "factor_id"
    ):

        factor_name = group[
            "factor_name"
        ].iloc[
            0
        ]


        total_abs = float(
            group[
                "abs_shap"
            ].sum()
        )


        signed_sum = float(
            group[
                "shap_value"
            ].sum()
        )


        strongest_index = (

            group[
                "abs_shap"
            ]
            .idxmax()
        )


        strongest = standardized.loc[
            strongest_index
        ]


        (
            local_direction,
            positive_share,
            negative_share
        ) = determine_local_direction(
            group
        )


        global_info = root_lookup.get(
            factor_id
        )


        if global_info is None:

            raise RuntimeError(
                f"{case_type}: factor {factor_id} "
                "missing from root-cause matrix."
            )


        factor_rows.append(
            {
                "case_type":
                    case_type,

                "factor_id":
                    factor_id,

                "factor_name":
                    factor_name,

                "local_abs_shap_total":
                    total_abs,

                "local_signed_shap_sum":
                    signed_sum,

                "local_direction":
                    local_direction,

                "positive_abs_shap_share":
                    positive_share,

                "negative_abs_shap_share":
                    negative_share,

                "representation_count":
                    len(
                        group
                    ),

                "dominant_local_feature":
                    strongest[
                        "feature"
                    ],

                "dominant_local_feature_value":
                    strongest[
                        "feature_value"
                    ],

                "dominant_local_feature_shap":
                    strongest[
                        "shap_value"
                    ],

                "dominant_representation_type":
                    strongest[
                        "representation_type"
                    ],

                "dominant_window_days":
                    strongest[
                        "window_days"
                    ],

                "global_shap_consensus_rank":
                    global_info[
                        "consensus_rank"
                    ],

                "global_shap_share_pct":
                    global_info[
                        "mean_normalized_shap_share_pct"
                    ],

                "pcmci_available":
                    global_info[
                        "pcmci_available"
                    ],

                "stable_cross_factor_degree":
                    global_info[
                        "stable_cross_factor_degree"
                    ],

                "matched_att_available":
                    global_info[
                        "matched_att_available"
                    ],

                "root_cause_candidate":
                    global_info[
                        "root_cause_candidate"
                    ],

                "integrated_evidence_class":
                    global_info[
                        "integrated_evidence_class"
                    ],

                "matched_effect_evidence_class":
                    global_info[
                        "matched_effect_evidence_class"
                    ],

                "failure_risk_difference_pp":
                    global_info[
                        "failure_risk_difference_pp"
                    ],

                "failure_risk_ci_lower_pp":
                    global_info[
                        "failure_risk_ci_lower_pp"
                    ],

                "failure_risk_ci_upper_pp":
                    global_info[
                        "failure_risk_ci_upper_pp"
                    ],

                "degradation_temporal_pathway_available":
                    global_info[
                        "degradation_temporal_pathway_available"
                    ],

                "degradation_temporal_pathways":
                    global_info[
                        "degradation_temporal_pathways"
                    ],
            }
        )


    factor_df = pd.DataFrame(
        factor_rows
    )


    # ------------------------------------------------------
    # Normalize within-instance attribution mass
    # ------------------------------------------------------

    total_case_abs_shap = float(
        factor_df[
            "local_abs_shap_total"
        ].sum()
    )


    if total_case_abs_shap > 0:

        factor_df[
            "local_importance_share"
        ] = (

            factor_df[
                "local_abs_shap_total"
            ]

            /

            total_case_abs_shap
        )


    else:

        factor_df[
            "local_importance_share"
        ] = 0.0


    factor_df[
        "local_importance_share_pct"
    ] = (

        factor_df[
            "local_importance_share"
        ]

        *
        100
    )


    factor_df[
        "local_factor_rank"
    ] = (

        factor_df[
            "local_abs_shap_total"
        ]

        .rank(
            method="min",
            ascending=False
        )

        .astype(int)
    )


    factor_df[
        "local_fleet_evidence_alignment"
    ] = factor_df.apply(

        classify_alignment,

        axis=1
    )


    factor_df = (

        factor_df

        .sort_values(
            [
                "local_factor_rank",
                "factor_name",
            ]
        )

        .reset_index(
            drop=True
        )
    )


    # ------------------------------------------------------
    # Extract case metadata if available
    # ------------------------------------------------------

    metadata = {

        "case_type":
            case_type,

        "case_description":
            CASE_DESCRIPTIONS[
                case_type
            ],

        "serial_number":
            extract_metadata_value(
                local,
                serial_column
            ),

        "date":
            extract_metadata_value(
                local,
                date_column
            ),

        "predicted_probability":
            extract_metadata_value(
                local,
                probability_column
            ),

        "actual_label":
            extract_metadata_value(
                local,
                actual_label_column
            ),

        "predicted_label":
            extract_metadata_value(
                local,
                predicted_label_column
            ),

        "threshold":
            extract_metadata_value(
                local,
                threshold_column
            ),

        "feature_count":
            len(
                standardized
            ),

        "factor_count":
            len(
                factor_df
            ),
    }


    # ------------------------------------------------------
    # Build human-readable local explanation
    # ------------------------------------------------------

    top_factors = factor_df.head(
        5
    )


    explanation_parts = []


    if metadata[
        "serial_number"
    ] is not None:

        explanation_parts.append(

            f"HDD "
            f"{metadata['serial_number']}"
        )


    else:

        explanation_parts.append(
            f"{case_type} representative HDD"
        )


    if metadata[
        "date"
    ] is not None:

        explanation_parts[
            0
        ] += (
            f" on {metadata['date']}"
        )


    if metadata[
        "predicted_probability"
    ] is not None:

        try:

            probability_value = float(
                metadata[
                    "predicted_probability"
                ]
            )


            explanation_parts.append(

                f"The model's recorded failure-risk "
                f"prediction was "
                f"{probability_value:.4f}."
            )


        except (
            TypeError,
            ValueError
        ):

            pass


    # ------------------------------------------------------
    # Local attribution explanation
    # ------------------------------------------------------

    risk_increasing = top_factors[
        top_factors[
            "local_direction"
        ]
        ==
        "TOWARD_FAILURE_OUTPUT"
    ]


    if len(
        risk_increasing
    ) > 0:

        factor_text = ", ".join(

            risk_increasing[
                "factor_name"
            ]
            .head(
                3
            )
            .tolist()
        )


        explanation_parts.append(

            "The strongest factor-level local SHAP "
            "contributions pushing the model toward "
            "the failure output included "
            f"{factor_text}."
        )


    risk_reducing = top_factors[
        top_factors[
            "local_direction"
        ]
        ==
        "AWAY_FROM_FAILURE_OUTPUT"
    ]


    if len(
        risk_reducing
    ) > 0:

        factor_text = ", ".join(

            risk_reducing[
                "factor_name"
            ]
            .head(
                3
            )
            .tolist()
        )


        explanation_parts.append(

            "Important local contributions pushing "
            "the model away from the failure output "
            f"included {factor_text}."
        )


    # ------------------------------------------------------
    # Fleet root-cause evidence alignment
    # ------------------------------------------------------

    aligned_root = factor_df[
        (
            factor_df[
                "root_cause_candidate"
            ].apply(
                as_bool
            )
        )
        &
        (
            factor_df[
                "local_direction"
            ]
            ==
            "TOWARD_FAILURE_OUTPUT"
        )
    ]


    if len(
        aligned_root
    ) > 0:

        aligned_names = ", ".join(

            aligned_root[
                "factor_name"
            ]
            .head(
                3
            )
            .tolist()
        )


        explanation_parts.append(

            "The local prediction therefore aligns "
            "with fleet-level root-cause-oriented "
            f"evidence for {aligned_names}."
        )


    else:

        explanation_parts.append(

            "The strongest local prediction drivers "
            "do not provide direct instance-level "
            "confirmation of the fleet-level "
            "root-cause candidates."
        )


    # ------------------------------------------------------
    # High-priority root-cause candidate specifically
    # ------------------------------------------------------

    high_priority_local = factor_df[
        (
            factor_df[
                "integrated_evidence_class"
            ]
            ==
            "HIGH_PRIORITY_ROOT_CAUSE_CANDIDATE"
        )
    ]


    if len(
        high_priority_local
    ) == 1:

        row = high_priority_local.iloc[
            0
        ]


        if row[
            "local_direction"
        ] == (
            "TOWARD_FAILURE_OUTPUT"
        ):

            explanation_parts.append(

                f"{row['factor_name']} contributed "
                "toward the model's failure output "
                "for this HDD and is also the "
                "high-priority fleet-level "
                "root-cause candidate."
            )


        elif row[
            "local_direction"
        ] == (
            "AWAY_FROM_FAILURE_OUTPUT"
        ):

            explanation_parts.append(

                f"{row['factor_name']} is the "
                "high-priority fleet-level root-cause "
                "candidate, but for this particular "
                "HDD its local SHAP contribution did "
                "not push the model toward failure."
            )


        else:

            explanation_parts.append(

                f"{row['factor_name']} is the "
                "high-priority fleet-level root-cause "
                "candidate, while its local SHAP "
                "representations showed mixed "
                "direction for this HDD."
            )


    # ------------------------------------------------------
    # Safety statement
    # ------------------------------------------------------

    explanation_parts.append(

        "This explanation separates instance-level "
        "predictive attribution from fleet-level "
        "temporal and matched observational evidence. "
        "It should be interpreted as root-cause-oriented "
        "support rather than proof of an individual "
        "physical causal mechanism."
    )


    explanation_text = (
        " ".join(
            explanation_parts
        )
    )


    # ------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------

    feature_output_file = (
        OUTPUT_DIR
        /
        f"{case_type}_local_feature_evidence.csv"
    )


    factor_output_file = (
        OUTPUT_DIR
        /
        f"{case_type}_local_factor_evidence.csv"
    )


    json_output_file = (
        OUTPUT_DIR
        /
        f"{case_type}_local_root_cause_explanation.json"
    )


    standardized.to_csv(
        feature_output_file,
        index=False
    )


    factor_df.to_csv(
        factor_output_file,
        index=False
    )


    json_payload = {

        "case_metadata":
            metadata,

        "local_explanation":
            explanation_text,

        "top_local_factors":
            factor_df.head(
                8
            ).replace(
                {
                    np.nan:
                        None
                }
            ).to_dict(
                orient="records"
            ),

        "interpretation_scope":
            (
                "Local SHAP is predictive attribution. "
                "Fleet-level PCMCI and matched ATT evidence "
                "provide supporting causal context under "
                "observational assumptions. No individual "
                "physical causal claim is made."
            ),
    }


    with open(
        json_output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            json_payload,
            file,
            indent=4,
            default=str
        )


    return {

        "metadata":
            metadata,

        "feature_df":
            standardized,

        "factor_df":
            factor_df,

        "explanation":
            explanation_text,

        "feature_output_file":
            feature_output_file,

        "factor_output_file":
            factor_output_file,

        "json_output_file":
            json_output_file,
    }


# ==========================================================
# PROCESS ALL FOUR REPRESENTATIVE CASES
# ==========================================================

case_results = {}


for case_type, file_path in LOCAL_CASE_FILES.items():

    print(
        "\n"
        + "=" * 80
    )


    print(
        f"Processing local case: {case_type}"
    )


    print(
        "=" * 80
    )


    result = process_local_case(
        case_type,
        file_path
    )


    case_results[
        case_type
    ] = result


    print(
        "\nTop local factors:\n"
    )


    print(
        result[
            "factor_df"
        ][
            [
                "local_factor_rank",

                "factor_name",

                "local_importance_share_pct",

                "local_direction",

                "dominant_local_feature",

                "integrated_evidence_class",

                "local_fleet_evidence_alignment",
            ]
        ]
        .head(
            8
        )
        .to_string(
            index=False
        )
    )


    print(
        "\nExplanation:\n"
    )


    print(
        result[
            "explanation"
        ]
    )


# ==========================================================
# CROSS-CASE SUMMARY
# ==========================================================

summary_rows = []


for case_type, result in case_results.items():

    factor_df = result[
        "factor_df"
    ]


    metadata = result[
        "metadata"
    ]


    top_factor = factor_df.iloc[
        0
    ]


    high_priority = factor_df[
        factor_df[
            "integrated_evidence_class"
        ]
        ==
        "HIGH_PRIORITY_ROOT_CAUSE_CANDIDATE"
    ]


    if len(
        high_priority
    ) == 1:

        high_priority = (
            high_priority.iloc[
                0
            ]
        )


        high_priority_direction = (
            high_priority[
                "local_direction"
            ]
        )


        high_priority_rank = (
            high_priority[
                "local_factor_rank"
            ]
        )


        high_priority_share = (
            high_priority[
                "local_importance_share_pct"
            ]
        )


    else:

        high_priority_direction = (
            None
        )


        high_priority_rank = (
            np.nan
        )


        high_priority_share = (
            np.nan
        )


    root_candidates_local_positive = factor_df[
        (
            factor_df[
                "root_cause_candidate"
            ].apply(
                as_bool
            )
        )
        &
        (
            factor_df[
                "local_direction"
            ]
            ==
            "TOWARD_FAILURE_OUTPUT"
        )
    ]


    summary_rows.append(
        {
            "case_type":
                case_type,

            "serial_number":
                metadata[
                    "serial_number"
                ],

            "date":
                metadata[
                    "date"
                ],

            "predicted_probability":
                metadata[
                    "predicted_probability"
                ],

            "top_local_factor":
                top_factor[
                    "factor_name"
                ],

            "top_local_factor_direction":
                top_factor[
                    "local_direction"
                ],

            "top_local_factor_share_pct":
                top_factor[
                    "local_importance_share_pct"
                ],

            "high_priority_root_candidate_local_rank":
                high_priority_rank,

            "high_priority_root_candidate_local_share_pct":
                high_priority_share,

            "high_priority_root_candidate_direction":
                high_priority_direction,

            "local_positive_root_candidate_count":
                len(
                    root_candidates_local_positive
                ),

            "local_positive_root_candidates":
                "; ".join(

                    root_candidates_local_positive[
                        "factor_name"
                    ].tolist()
                ),
        }
    )


cross_case_summary = pd.DataFrame(
    summary_rows
)


cross_case_summary_file = (
    OUTPUT_DIR
    / "local_case_explanation_summary.csv"
)


cross_case_summary.to_csv(
    cross_case_summary_file,
    index=False
)


# ==========================================================
# TOP-5 FACTOR OCCURRENCE ACROSS CASE TYPES
# ==========================================================

top5_rows = []


for case_type, result in case_results.items():

    top5 = (
        result[
            "factor_df"
        ]
        .head(
            5
        )
    )


    for _, row in top5.iterrows():

        top5_rows.append(
            {
                "case_type":
                    case_type,

                "factor_id":
                    row[
                        "factor_id"
                    ],

                "factor_name":
                    row[
                        "factor_name"
                    ],

                "local_factor_rank":
                    row[
                        "local_factor_rank"
                    ],

                "local_direction":
                    row[
                        "local_direction"
                    ],

                "local_importance_share_pct":
                    row[
                        "local_importance_share_pct"
                    ],

                "integrated_evidence_class":
                    row[
                        "integrated_evidence_class"
                    ],
            }
        )


top5_df = pd.DataFrame(
    top5_rows
)


top5_file = (
    OUTPUT_DIR
    / "local_top5_factor_comparison.csv"
)


top5_df.to_csv(
    top5_file,
    index=False
)


# ==========================================================
# METADATA
# ==========================================================

metadata = {

    "method":
        "Local SHAP plus fleet-level causal evidence enrichment",

    "local_predictive_evidence":
        (
            "Local SHAP values are aggregated to canonical "
            "HDD factors using sum of absolute SHAP for "
            "importance. Direction is determined from "
            "positive/negative absolute SHAP mass."
        ),

    "fleet_temporal_evidence":
        (
            "PCMCI stability evidence is inherited from "
            "the integrated fleet-level evidence matrix."
        ),

    "fleet_failure_risk_evidence":
        (
            "Matched incident ATT evidence is inherited "
            "from the integrated fleet-level evidence matrix."
        ),

    "important_limitation":
        (
            "Fleet-level causal evidence is not converted "
            "into an individual counterfactual causal claim. "
            "Local SHAP remains predictive attribution."
        ),

    "representative_cases":
        [
            "TP",
            "FP",
            "FN",
            "TN",
        ],
}


metadata_file = (
    OUTPUT_DIR
    / "local_root_cause_explanation_metadata.json"
)


with open(
    metadata_file,
    "w",
    encoding="utf-8"
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
print("=" * 84)

print(
    "STEP 13H - LOCAL ROOT-CAUSE EXPLANATION ENGINE COMPLETE"
)

print("=" * 84)


print(
    "\nCross-case summary:\n"
)


print(
    cross_case_summary.to_string(
        index=False
    )
)


print(
    "\nOutputs saved to:"
)


print(
    OUTPUT_DIR
)
