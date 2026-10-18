from pathlib import Path
from datetime import datetime, timezone
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

ROOT_CAUSE_DIR = (
    RESULTS_ROOT
    / "integrated_evidence"
    / "root_cause_matrix"
)

PCMCI_DIR = (
    RESULTS_ROOT
    / "integrated_evidence"
    / "pcmci_temporal"
)

ATT_DIR = (
    RESULTS_ROOT
    / "integrated_evidence"
    / "matched_att"
)

LOCAL_DIR = (
    RESULTS_ROOT
    / "integrated_evidence"
    / "local_root_cause"
)

OUTPUT_DIR = (
    RESULTS_ROOT
    / "api_exports"
)

CASE_OUTPUT_DIR = (
    OUTPUT_DIR
    / "representative_cases"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CASE_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# INPUT FILES
# ==========================================================

ROOT_CAUSE_MATRIX_FILE = (
    ROOT_CAUSE_DIR
    / "integrated_root_cause_evidence_matrix.csv"
)

ROOT_CAUSE_CANDIDATES_FILE = (
    ROOT_CAUSE_DIR
    / "root_cause_candidate_summary.csv"
)

TEMPORAL_PATHWAYS_FILE = (
    ROOT_CAUSE_DIR
    / "degradation_temporal_pathways.csv"
)

EVIDENCE_STREAM_FILE = (
    ROOT_CAUSE_DIR
    / "integrated_evidence_stream_summary.csv"
)

ATT_FILE = (
    ATT_DIR
    / "factor_matched_att_evidence.csv"
)

LOCAL_SUMMARY_FILE = (
    LOCAL_DIR
    / "local_case_explanation_summary.csv"
)

LOCAL_TOP5_FILE = (
    LOCAL_DIR
    / "local_top5_factor_comparison.csv"
)


LOCAL_JSON_FILES = {

    "TP":
        LOCAL_DIR
        / "TP_local_root_cause_explanation.json",

    "FP":
        LOCAL_DIR
        / "FP_local_root_cause_explanation.json",

    "FN":
        LOCAL_DIR
        / "FN_local_root_cause_explanation.json",

    "TN":
        LOCAL_DIR
        / "TN_local_root_cause_explanation.json",
}


# ==========================================================
# VALIDATE INPUT FILES
# ==========================================================

required_files = [

    ROOT_CAUSE_MATRIX_FILE,

    ROOT_CAUSE_CANDIDATES_FILE,

    TEMPORAL_PATHWAYS_FILE,

    EVIDENCE_STREAM_FILE,

    ATT_FILE,

    LOCAL_SUMMARY_FILE,

    LOCAL_TOP5_FILE,

    *LOCAL_JSON_FILES.values(),
]


missing_files = [

    str(path)

    for path in required_files

    if not path.exists()
]


if missing_files:

    raise FileNotFoundError(
        "Missing required Step 13I inputs:\n"
        +
        "\n".join(
            missing_files
        )
    )


# ==========================================================
# LOAD DATA
# ==========================================================

root_matrix = pd.read_csv(
    ROOT_CAUSE_MATRIX_FILE
)

root_candidates = pd.read_csv(
    ROOT_CAUSE_CANDIDATES_FILE
)

temporal_pathways = pd.read_csv(
    TEMPORAL_PATHWAYS_FILE
)

evidence_streams = pd.read_csv(
    EVIDENCE_STREAM_FILE
)

att = pd.read_csv(
    ATT_FILE
)

local_summary = pd.read_csv(
    LOCAL_SUMMARY_FILE
)

local_top5 = pd.read_csv(
    LOCAL_TOP5_FILE
)


# ==========================================================
# GENERAL HELPERS
# ==========================================================

def safe_value(
    value
):

    if value is None:

        return None


    if isinstance(
        value,
        (
            np.integer,
        )
    ):

        return int(
            value
        )


    if isinstance(
        value,
        (
            np.floating,
        )
    ):

        if np.isnan(
            value
        ):

            return None

        return float(
            value
        )


    if isinstance(
        value,
        float
    ):

        if np.isnan(
            value
        ):

            return None

        return value


    if pd.isna(
        value
    ):

        return None


    if isinstance(
        value,
        (
            np.bool_,
        )
    ):

        return bool(
            value
        )


    return value


def clean_record(
    record
):

    cleaned = {}


    for key, value in record.items():

        cleaned[
            key
        ] = safe_value(
            value
        )


    return cleaned


def dataframe_to_records(
    dataframe
):

    records = dataframe.to_dict(
        orient="records"
    )


    return [

        clean_record(
            record
        )

        for record in records
    ]


def as_bool(
    value
):

    if isinstance(
        value,
        bool
    ):

        return value


    if value is None:

        return False


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


def save_json(
    payload,
    path
):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            payload,
            file,
            indent=4,
            ensure_ascii=False,
            allow_nan=False,
            default=str
        )


    print(
        "Saved:",
        path
    )


# ==========================================================
# EXPORT METADATA
# ==========================================================

generated_at = datetime.now(
    timezone.utc
).isoformat()


component_metadata = {

    "component_id":
        "C3",

    "component_name":
        "Causal Explainable AI for HDD Failure Analysis",

    "prediction_horizon_days":
        30,

    "model_context":
        "XGBoost reference failure-risk model",

    "api_schema_version":
        "1.0",

    "generated_at_utc":
        generated_at,

    "evidence_framework":
        {
            "predictive":
                "Global and local SHAP",

            "temporal":
                "PCMCI repeated stability analysis",

            "failure_risk":
                (
                    "Incident matched observational "
                    "ATT risk-difference analysis"
                ),
        },

    "interpretation_scope":
        (
            "The exported results distinguish predictive "
            "attribution, stable temporal relationships, "
            "and matched observational failure-risk evidence. "
            "They support root-cause-oriented reasoning but "
            "do not prove physical causation."
        ),
}


# ==========================================================
# ROOT-CAUSE CATALOG
# ==========================================================

root_catalog = []


for _, row in root_matrix.iterrows():

    factor_payload = {

        "factor_id":
            safe_value(
                row[
                    "factor_id"
                ]
            ),

        "factor_name":
            safe_value(
                row[
                    "factor_name"
                ]
            ),

        "category":
            safe_value(
                row[
                    "category"
                ]
            ),

        "semantic_status":
            safe_value(
                row[
                    "semantic_status"
                ]
            ),

        "predictive_evidence":
            {
                "consensus_rank":
                    safe_value(
                        row[
                            "consensus_rank"
                        ]
                    ),

                "normalized_shap_share_pct":
                    safe_value(
                        row[
                            "mean_normalized_shap_share_pct"
                        ]
                    ),

                "band":
                    safe_value(
                        row[
                            "predictive_evidence_band"
                        ]
                    ),

                "rank_stability":
                    safe_value(
                        row[
                            "rank_stability_label"
                        ]
                    ),
            },

        "temporal_evidence":
            {
                "available":
                    as_bool(
                        row[
                            "pcmci_available"
                        ]
                    ),

                "stable_cross_factor_degree":
                    safe_value(
                        row[
                            "stable_cross_factor_degree"
                        ]
                    ),

                "connectivity_class":
                    safe_value(
                        row[
                            "temporal_connectivity_class"
                        ]
                    ),

                "strongest_relation":
                    safe_value(
                        row[
                            "strongest_temporal_relation"
                        ]
                    ),

                "strongest_magnitude":
                    safe_value(
                        row[
                            "strongest_temporal_magnitude"
                        ]
                    ),

                "degradation_pathway_available":
                    as_bool(
                        row[
                            "degradation_temporal_pathway_available"
                        ]
                    ),

                "degradation_pathways":
                    safe_value(
                        row[
                            "degradation_temporal_pathways"
                        ]
                    ),
            },

        "matched_failure_risk_evidence":
            {
                "available":
                    as_bool(
                        row[
                            "matched_att_available"
                        ]
                    ),

                "analysis_role":
                    safe_value(
                        row[
                            "analysis_role"
                        ]
                    ),

                "risk_difference_pp":
                    safe_value(
                        row[
                            "failure_risk_difference_pp"
                        ]
                    ),

                "ci_lower_pp":
                    safe_value(
                        row[
                            "failure_risk_ci_lower_pp"
                        ]
                    ),

                "ci_upper_pp":
                    safe_value(
                        row[
                            "failure_risk_ci_upper_pp"
                        ]
                    ),

                "evidence_class":
                    safe_value(
                        row[
                            "matched_effect_evidence_class"
                        ]
                    ),
            },

        "integrated_evidence":
            {
                "evidence_stream_count":
                    safe_value(
                        row[
                            "evidence_stream_count"
                        ]
                    ),

                "classification":
                    safe_value(
                        row[
                            "integrated_evidence_class"
                        ]
                    ),

                "root_cause_candidate":
                    as_bool(
                        row[
                            "root_cause_candidate"
                        ]
                    ),

                "explanation":
                    safe_value(
                        row[
                            "integrated_explanation"
                        ]
                    ),
            },
    }


    root_catalog.append(
        factor_payload
    )


# ==========================================================
# ROOT-CAUSE CANDIDATE PAYLOAD
# ==========================================================

candidate_payload = []


for _, row in root_candidates.iterrows():

    candidate_payload.append(
        {
            "factor_id":
                safe_value(
                    row[
                        "factor_id"
                    ]
                ),

            "factor_name":
                safe_value(
                    row[
                        "factor_name"
                    ]
                ),

            "global_shap_rank":
                safe_value(
                    row[
                        "consensus_rank"
                    ]
                ),

            "stable_temporal_degree":
                safe_value(
                    row[
                        "stable_cross_factor_degree"
                    ]
                ),

            "degradation_temporal_pathway":
                as_bool(
                    row[
                        "degradation_temporal_pathway_available"
                    ]
                ),

            "matched_failure_risk_difference_pp":
                safe_value(
                    row[
                        "failure_risk_difference_pp"
                    ]
                ),

            "matched_failure_risk_ci":
                [
                    safe_value(
                        row[
                            "failure_risk_ci_lower_pp"
                        ]
                    ),

                    safe_value(
                        row[
                            "failure_risk_ci_upper_pp"
                        ]
                    ),
                ],

            "classification":
                safe_value(
                    row[
                        "integrated_evidence_class"
                    ]
                ),
        }
    )


# ==========================================================
# TEMPORAL PATHWAY PAYLOAD
# ==========================================================

temporal_payload = []


for _, row in temporal_pathways.iterrows():

    pathway = {

        "source_factor_id":
            safe_value(
                row[
                    "source_factor_id"
                ]
            ),

        "source_factor_name":
            safe_value(
                row[
                    "source_factor_name"
                ]
            ),

        "target_factor_id":
            safe_value(
                row[
                    "target_factor_id"
                ]
            ),

        "target_factor_name":
            safe_value(
                row[
                    "target_factor_name"
                ]
            ),
    }


    optional_fields = [

        "temporal_pair_rank",

        "temporal_magnitude",

        "modal_lag_days",

        "strongest_lag_days",

        "temporal_direction",

        "pair_detection_rate",

        "pair_sign_consistency",

        "lag_sign_pattern",
    ]


    for field in optional_fields:

        if field in temporal_pathways.columns:

            pathway[
                field
            ] = safe_value(
                row[
                    field
                ]
            )


    pathway[
        "interpretation"
    ] = (

        "Stable fleet-level temporal dependency. "
        "This relationship supports temporal ordering "
        "but does not independently prove a physical "
        "causal mechanism."
    )


    temporal_payload.append(
        pathway
    )


# ==========================================================
# MATCHED FAILURE-RISK PAYLOAD
# ==========================================================

att_payload = []


for _, row in att.iterrows():

    att_payload.append(
        {
            "factor_id":
                safe_value(
                    row[
                        "factor_id"
                    ]
                ),

            "factor_name":
                safe_value(
                    row[
                        "factor_name"
                    ]
                ),

            "analysis_role":
                safe_value(
                    row[
                        "analysis_role"
                    ]
                ),

            "matched_pairs":
                safe_value(
                    row[
                        "matched_pairs"
                    ]
                ),

            "treated_failures":
                safe_value(
                    row[
                        "treated_events"
                    ]
                ),

            "control_failures":
                safe_value(
                    row[
                        "control_events"
                    ]
                ),

            "risk_difference_pp":
                safe_value(
                    row[
                        "failure_risk_difference_pp"
                    ]
                ),

            "cluster_bootstrap_95ci_pp":
                [
                    safe_value(
                        row[
                            "failure_risk_ci_lower_pp"
                        ]
                    ),

                    safe_value(
                        row[
                            "failure_risk_ci_upper_pp"
                        ]
                    ),
                ],

            "mcnemar_exact_p_value":
                safe_value(
                    row[
                        "mcnemar_exact_p_value"
                    ]
                ),

            "max_abs_smd_after_matching":
                safe_value(
                    row[
                        "max_abs_smd_after"
                    ]
                ),

            "evidence_class":
                safe_value(
                    row[
                        "matched_effect_evidence_class"
                    ]
                ),

            "interpretation":
                safe_value(
                    row[
                        "interpretation"
                    ]
                ),

            "interpretation_scope":
                (
                    "Matched observational failure-risk "
                    "contrast under measured-confounding "
                    "and temporal-ordering assumptions."
                ),
        }
    )


# ==========================================================
# REPRESENTATIVE LOCAL CASE PAYLOADS
# ==========================================================

representative_cases = {}


for case_type, json_file in LOCAL_JSON_FILES.items():

    with open(
        json_file,
        "r",
        encoding="utf-8"
    ) as file:

        local_payload = json.load(
            file
        )


    case_metadata = local_payload.get(
        "case_metadata",
        {}
    )


    api_case = {

        "case_type":
            case_type,

        "hdd":
            {
                "serial_number":
                    case_metadata.get(
                        "serial_number"
                    ),

                "observation_date":
                    case_metadata.get(
                        "date"
                    ),
            },

        "prediction":
            {
                "horizon_days":
                    30,

                "failure_probability":
                    case_metadata.get(
                        "predicted_probability"
                    ),

                "predicted_label":
                    case_metadata.get(
                        "predicted_label"
                    ),

                "actual_label":
                    case_metadata.get(
                        "actual_label"
                    ),

                "threshold":
                    case_metadata.get(
                        "threshold"
                    ),
            },

        "local_explanation":
            {
                "text":
                    local_payload.get(
                        "local_explanation"
                    ),

                "top_factors":
                    local_payload.get(
                        "top_local_factors",
                        []
                    ),
            },

        "interpretation_scope":
            local_payload.get(
                "interpretation_scope"
            ),
    }


    representative_cases[
        case_type
    ] = api_case


    case_output_file = (
        CASE_OUTPUT_DIR
        /
        f"{case_type.lower()}_dashboard_payload.json"
    )


    save_json(
        api_case,
        case_output_file
    )


# ==========================================================
# EVIDENCE STREAM SUMMARY
# ==========================================================

stream_payload = dataframe_to_records(
    evidence_streams
)


# ==========================================================
# DASHBOARD SUMMARY CARDS
# ==========================================================

high_priority = root_matrix[
    root_matrix[
        "integrated_evidence_class"
    ]
    ==
    "HIGH_PRIORITY_ROOT_CAUSE_CANDIDATE"
]


supporting = root_matrix[
    root_matrix[
        "integrated_evidence_class"
    ]
    ==
    "SUPPORTING_DEGRADATION_FACTOR"
]


exploratory = root_matrix[
    root_matrix[
        "integrated_evidence_class"
    ]
    ==
    "EXPLORATORY_ROOT_CAUSE_CANDIDATE"
]


dashboard_cards = {

    "canonical_factor_count":
        int(
            len(
                root_matrix
            )
        ),

    "high_priority_root_cause_candidates":
        int(
            len(
                high_priority
            )
        ),

    "supporting_degradation_factors":
        int(
            len(
                supporting
            )
        ),

    "exploratory_root_cause_candidates":
        int(
            len(
                exploratory
            )
        ),

    "factors_with_stable_pcmci_evidence":
        int(
            root_matrix[
                "pcmci_available"
            ]
            .apply(
                as_bool
            )
            .sum()
        ),

    "factors_with_matched_failure_risk_evidence":
        int(
            root_matrix[
                "matched_att_available"
            ]
            .apply(
                as_bool
            )
            .sum()
        ),

    "primary_root_cause_candidate":
        (
            safe_value(
                high_priority.iloc[
                    0
                ][
                    "factor_name"
                ]
            )

            if len(
                high_priority
            ) > 0

            else None
        ),
}


# ==========================================================
# COMPLETE COMPONENT 3 API EXPORT
# ==========================================================

component3_payload = {

    "metadata":
        component_metadata,

    "dashboard_summary":
        dashboard_cards,

    "root_cause_candidates":
        candidate_payload,

    "factor_evidence_catalog":
        root_catalog,

    "degradation_temporal_pathways":
        temporal_payload,

    "matched_failure_risk_evidence":
        att_payload,

    "evidence_stream_coverage":
        stream_payload,

    "representative_local_cases":
        representative_cases,

    "warnings":
        [
            (
                "SHAP values describe predictive attribution "
                "and are not causal effects."
            ),

            (
                "PCMCI links describe stable temporal "
                "conditional dependencies under method assumptions."
            ),

            (
                "Matched ATT values are observational "
                "failure-risk contrasts and remain subject "
                "to unmeasured confounding."
            ),

            (
                "Fleet-level root-cause evidence must not "
                "be interpreted as proof that a factor physically "
                "caused failure in an individual HDD."
            ),
        ],
}


# ==========================================================
# SAVE MASTER PAYLOAD
# ==========================================================

master_file = (
    OUTPUT_DIR
    / "component3_api_payload.json"
)


save_json(
    component3_payload,
    master_file
)


# ==========================================================
# SAVE DASHBOARD-ONLY PAYLOAD
# ==========================================================

dashboard_payload = {

    "component":
        {
            "id":
                "C3",

            "name":
                "Causal Explainable AI for HDD Failure Analysis",
        },

    "summary":
        dashboard_cards,

    "root_cause_candidates":
        candidate_payload,

    "temporal_pathways":
        temporal_payload,

    "matched_failure_risk_evidence":
        att_payload,

    "representative_cases":
        {
            case:
                {
                    "case_type":
                        payload[
                            "case_type"
                        ],

                    "hdd":
                        payload[
                            "hdd"
                        ],

                    "prediction":
                        payload[
                            "prediction"
                        ],

                    "top_factors":
                        payload[
                            "local_explanation"
                        ][
                            "top_factors"
                        ][:5],
                }

            for case, payload
            in representative_cases.items()
        },

    "interpretation_warning":
        (
            "Root-cause classifications integrate "
            "predictive, temporal, and matched "
            "observational evidence. They do not "
            "constitute proof of physical causation."
        ),
}


dashboard_file = (
    OUTPUT_DIR
    / "component3_dashboard_payload.json"
)


save_json(
    dashboard_payload,
    dashboard_file
)


# ==========================================================
# API CONTRACT EXAMPLE
#
# This demonstrates what a future live HDD endpoint can
# return after the system receives serial/date/prediction.
# ==========================================================

api_contract_example = {

    "request_example":
        {
            "serial_number":
                "HDD_SERIAL",

            "observation_date":
                "YYYY-MM-DD",

            "prediction_horizon_days":
                30,
        },

    "response_example":
        {
            "serial_number":
                "HDD_SERIAL",

            "observation_date":
                "YYYY-MM-DD",

            "prediction":
                {
                    "failure_probability":
                        0.0,

                    "threshold":
                        0.0,

                    "predicted_class":
                        "LOW_RISK_OR_HIGH_RISK",
                },

            "local_predictive_explanation":
                {
                    "top_factors":
                        [
                            {
                                "factor_id":
                                    "FACTOR_ID",

                                "factor_name":
                                    "Factor Name",

                                "local_importance_share_pct":
                                    0.0,

                                "local_direction":
                                    "TOWARD_FAILURE_OUTPUT",

                                "dominant_feature":
                                    "smart_x_raw_std_30d",
                            }
                        ],
                },

            "root_cause_context":
                {
                    "candidate_factors":
                        [
                            {
                                "factor_id":
                                    "REPORTED_UNCORRECTABLE",

                                "classification":
                                    (
                                        "HIGH_PRIORITY_"
                                        "ROOT_CAUSE_CANDIDATE"
                                    ),

                                "local_alignment":
                                    (
                                        "LOCAL_PREDICTIVE_SUPPORT_"
                                        "ALIGNS_WITH_FLEET_"
                                        "ROOT_CAUSE_EVIDENCE"
                                    ),
                            }
                        ],

                    "temporal_pathways":
                        [],

                    "matched_failure_risk_evidence":
                        [],
                },

            "interpretation_warning":
                (
                    "Local SHAP explains the prediction. "
                    "Fleet-level causal evidence provides "
                    "context but does not establish an "
                    "individual counterfactual causal effect."
                ),
        },
}


contract_file = (
    OUTPUT_DIR
    / "component3_api_contract_example.json"
)


save_json(
    api_contract_example,
    contract_file
)


# ==========================================================
# API FIELD MANIFEST
# ==========================================================

field_manifest = pd.DataFrame(
    [
        {
            "section":
                "prediction",

            "field":
                "failure_probability",

            "type":
                "float/null",

            "meaning":
                "Reference model 30-day failure probability",
        },

        {
            "section":
                "local_predictive_explanation",

            "field":
                "top_factors",

            "type":
                "array",

            "meaning":
                (
                    "Canonical factor-level local "
                    "SHAP explanation"
                ),
        },

        {
            "section":
                "root_cause_context",

            "field":
                "candidate_factors",

            "type":
                "array",

            "meaning":
                (
                    "Fleet-level integrated root-cause "
                    "candidate evidence"
                ),
        },

        {
            "section":
                "root_cause_context",

            "field":
                "temporal_pathways",

            "type":
                "array",

            "meaning":
                (
                    "Stable PCMCI temporal relationships"
                ),
        },

        {
            "section":
                "root_cause_context",

            "field":
                "matched_failure_risk_evidence",

            "type":
                "array",

            "meaning":
                (
                    "Matched observational 30-day "
                    "failure-risk contrasts"
                ),
        },

        {
            "section":
                "response",

            "field":
                "interpretation_warning",

            "type":
                "string",

            "meaning":
                (
                    "Required limitation / safe "
                    "causal interpretation text"
                ),
        },
    ]
)


field_manifest_file = (
    OUTPUT_DIR
    / "component3_api_field_manifest.csv"
)


field_manifest.to_csv(
    field_manifest_file,
    index=False
)


# ==========================================================
# FINAL VALIDATION
# ==========================================================

json_files = [

    master_file,

    dashboard_file,

    contract_file,

    *[
        CASE_OUTPUT_DIR
        / f"{case.lower()}_dashboard_payload.json"

        for case in LOCAL_JSON_FILES
    ],
]


validation_rows = []


for path in json_files:

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        loaded = json.load(
            file
        )


    validation_rows.append(
        {
            "file":
                path.name,

            "valid_json":
                True,

            "size_bytes":
                path.stat().st_size,

            "top_level_keys":
                ";".join(
                    loaded.keys()
                ),
        }
    )


validation_df = pd.DataFrame(
    validation_rows
)


validation_file = (
    OUTPUT_DIR
    / "api_export_validation.csv"
)


validation_df.to_csv(
    validation_file,
    index=False
)


# ==========================================================
# PRINT SUMMARY
# ==========================================================

print("\n")
print("=" * 84)

print(
    "STEP 13I - API / DASHBOARD EXPORT COMPLETE"
)

print("=" * 84)


print(
    "\nDashboard summary:\n"
)


for key, value in dashboard_cards.items():

    print(
        f"{key}: {value}"
    )


print(
    "\nRoot-cause candidates:\n"
)


for candidate in candidate_payload:

    print(
        "-",
        candidate[
            "factor_name"
        ],
        "->",
        candidate[
            "classification"
        ]
    )


print(
    "\nJSON validation:\n"
)


print(
    validation_df[
        [
            "file",

            "valid_json",

            "size_bytes",
        ]
    ].to_string(
        index=False
    )
)


print(
    "\nOutputs saved to:"
)


print(
    OUTPUT_DIR
)