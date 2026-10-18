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

INTEGRATED_INTERIM_DIR = (
    DATA_ROOT
    / "interim"
    / "integrated_evidence"
)

GLOBAL_SHAP_DIR = (
    RESULTS_ROOT
    / "integrated_evidence"
    / "global_shap"
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

OUTPUT_DIR = (
    RESULTS_ROOT
    / "integrated_evidence"
    / "root_cause_matrix"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# INPUT FILES
# ==========================================================

MANIFEST_FILE = (
    INTEGRATED_INTERIM_DIR
    / "canonical_factor_manifest.csv"
)

SHAP_CONSENSUS_FILE = (
    GLOBAL_SHAP_DIR
    / "global_shap_factor_consensus.csv"
)

PCMCI_ROLE_FILE = (
    PCMCI_DIR
    / "factor_temporal_role_summary.csv"
)

PCMCI_PAIR_FILE = (
    PCMCI_DIR
    / "factor_pair_temporal_evidence.csv"
)

ATT_FILE = (
    ATT_DIR
    / "factor_matched_att_evidence.csv"
)


# ==========================================================
# VALIDATE INPUT FILES
# ==========================================================

required_files = [

    MANIFEST_FILE,

    SHAP_CONSENSUS_FILE,

    PCMCI_ROLE_FILE,

    PCMCI_PAIR_FILE,

    ATT_FILE,
]


missing_files = [

    str(path)

    for path in required_files

    if not path.exists()
]


if missing_files:

    raise FileNotFoundError(
        "Missing Step 13 evidence files:\n"
        +
        "\n".join(
            missing_files
        )
    )


# ==========================================================
# LOAD INPUTS
# ==========================================================

manifest = pd.read_csv(
    MANIFEST_FILE
)

shap = pd.read_csv(
    SHAP_CONSENSUS_FILE
)

pcmci_role = pd.read_csv(
    PCMCI_ROLE_FILE
)

pcmci_pairs = pd.read_csv(
    PCMCI_PAIR_FILE
)

att = pd.read_csv(
    ATT_FILE
)


print(
    "\nCanonical factors:",
    len(manifest)
)

print(
    "SHAP factors:",
    len(shap)
)

print(
    "PCMCI factor-role rows:",
    len(pcmci_role)
)

print(
    "PCMCI cross-factor pairs:",
    len(pcmci_pairs)
)

print(
    "Matched ATT factors:",
    len(att)
)


# ==========================================================
# REQUIRED COLUMN VALIDATION
# ==========================================================

required_manifest_columns = {

    "factor_id",

    "factor_name",

    "category",

    "semantic_status",
}


required_shap_columns = {

    "factor_id",

    "consensus_rank",

    "population_factor_rank",

    "failure_factor_rank",

    "mean_normalized_shap_share_pct",

    "rank_stability_label",
}


required_pcmci_role_columns = {

    "factor_id",

    "stable_outgoing_factor_pairs",

    "stable_incoming_factor_pairs",

    "stable_cross_factor_degree",

    "maximum_outgoing_temporal_magnitude",

    "maximum_incoming_temporal_magnitude",
}


required_pcmci_pair_columns = {

    "source_factor_id",

    "target_factor_id",

    "source_factor_name",

    "target_factor_name",

    "temporal_pair_rank",

    "temporal_magnitude",
}


required_att_columns = {

    "factor_id",

    "analysis_role",

    "matched_pairs",

    "treated_events",

    "control_events",

    "failure_risk_difference_pp",

    "failure_risk_ci_lower_pp",

    "failure_risk_ci_upper_pp",

    "mcnemar_exact_p_value",

    "max_abs_smd_after",

    "matched_effect_evidence_class",

    "interpretation",
}


for label, frame, required in [

    (
        "manifest",
        manifest,
        required_manifest_columns
    ),

    (
        "SHAP",
        shap,
        required_shap_columns
    ),

    (
        "PCMCI role",
        pcmci_role,
        required_pcmci_role_columns
    ),

    (
        "PCMCI pairs",
        pcmci_pairs,
        required_pcmci_pair_columns
    ),

    (
        "matched ATT",
        att,
        required_att_columns
    ),
]:

    missing = (
        required
        -
        set(
            frame.columns
        )
    )


    if missing:

        raise RuntimeError(
            f"{label} missing columns: "
            f"{sorted(missing)}"
        )


# ==========================================================
# START WITH ALL 16 CANONICAL FACTORS
# ==========================================================

matrix = manifest[
    [
        "factor_id",

        "factor_name",

        "category",

        "semantic_status",
    ]
].copy()


# ==========================================================
# MERGE SHAP EVIDENCE
# ==========================================================

shap_columns = [

    "factor_id",

    "consensus_rank",

    "population_factor_rank",

    "failure_factor_rank",

    "mean_normalized_shap_share_pct",

    "rank_stability_label",
]


matrix = matrix.merge(

    shap[
        shap_columns
    ],

    on="factor_id",

    how="left",

    validate="one_to_one",
)


if matrix[
    "consensus_rank"
].isna().any():

    missing_factors = matrix.loc[
        matrix[
            "consensus_rank"
        ].isna(),
        "factor_id"
    ].tolist()


    raise RuntimeError(
        "Missing SHAP evidence for canonical factors: "
        f"{missing_factors}"
    )


# ==========================================================
# PREDICTIVE EVIDENCE BAND
#
# Rank-based descriptive category only.
# NOT a causal category.
# ==========================================================

def predictive_band(
    rank
):

    rank = int(
        rank
    )


    if rank <= 4:

        return (
            "TOP_QUARTILE_PREDICTIVE"
        )


    if rank <= 8:

        return (
            "HIGH_PREDICTIVE"
        )


    if rank <= 12:

        return (
            "MODERATE_PREDICTIVE"
        )


    return (
        "LOWER_PREDICTIVE"
    )


matrix[
    "predictive_evidence_band"
] = matrix[
    "consensus_rank"
].apply(
    predictive_band
)


matrix[
    "shap_available"
] = True


# ==========================================================
# MERGE PCMCI FACTOR ROLE EVIDENCE
# ==========================================================

pcmci_role_columns = [

    "factor_id",

    "stable_outgoing_factor_pairs",

    "stable_incoming_factor_pairs",

    "stable_cross_factor_degree",

    "stable_outgoing_exact_links",

    "stable_incoming_exact_links",

    "maximum_outgoing_temporal_magnitude",

    "maximum_incoming_temporal_magnitude",
]


available_pcmci_role_columns = [

    column

    for column in pcmci_role_columns

    if column in pcmci_role.columns
]


matrix = matrix.merge(

    pcmci_role[
        available_pcmci_role_columns
    ],

    on="factor_id",

    how="left",

    validate="one_to_one",
)


# ==========================================================
# FILL PCMCI COUNTS
# ==========================================================

count_columns = [

    "stable_outgoing_factor_pairs",

    "stable_incoming_factor_pairs",

    "stable_cross_factor_degree",

    "stable_outgoing_exact_links",

    "stable_incoming_exact_links",
]


for column in count_columns:

    if column not in matrix.columns:

        matrix[
            column
        ] = 0


    matrix[
        column
    ] = (
        matrix[
            column
        ]
        .fillna(
            0
        )
        .astype(
            int
        )
    )


matrix[
    "pcmci_available"
] = (

    matrix[
        "stable_cross_factor_degree"
    ]
    > 0
)


# ==========================================================
# TEMPORAL CONNECTIVITY CLASS
#
# This describes graph connectivity only.
# It does NOT mean causal strength.
# ==========================================================

def temporal_connectivity_class(
    degree
):

    degree = int(
        degree
    )


    if degree >= 8:

        return (
            "DENSE_STABLE_TEMPORAL_CONTEXT"
        )


    if degree >= 3:

        return (
            "MULTIPLE_STABLE_TEMPORAL_LINKS"
        )


    if degree >= 1:

        return (
            "LIMITED_STABLE_TEMPORAL_LINKS"
        )


    return (
        "NO_STABLE_PCMCI_LINK"
    )


matrix[
    "temporal_connectivity_class"
] = matrix[
    "stable_cross_factor_degree"
].apply(
    temporal_connectivity_class
)


# ==========================================================
# STRONGEST TEMPORAL RELATIONSHIP PER FACTOR
# ==========================================================

strongest_relation_rows = []


for factor_id in matrix[
    "factor_id"
]:

    relevant = pcmci_pairs[
        (
            pcmci_pairs[
                "source_factor_id"
            ]
            ==
            factor_id
        )
        |
        (
            pcmci_pairs[
                "target_factor_id"
            ]
            ==
            factor_id
        )
    ].copy()


    if len(
        relevant
    ) == 0:

        strongest_relation_rows.append(
            {
                "factor_id":
                    factor_id,

                "strongest_temporal_relation":
                    None,

                "strongest_temporal_relation_rank":
                    np.nan,

                "strongest_temporal_magnitude":
                    np.nan,
            }
        )

        continue


    strongest = (
        relevant

        .sort_values(
            [
                "temporal_magnitude",

                "temporal_pair_rank",
            ],
            ascending=[
                False,
                True,
            ]
        )

        .iloc[
            0
        ]
    )


    relation_text = (

        f"{strongest['source_factor_name']} "
        f"-> "
        f"{strongest['target_factor_name']}"
    )


    strongest_relation_rows.append(
        {
            "factor_id":
                factor_id,

            "strongest_temporal_relation":
                relation_text,

            "strongest_temporal_relation_rank":
                strongest[
                    "temporal_pair_rank"
                ],

            "strongest_temporal_magnitude":
                strongest[
                    "temporal_magnitude"
                ],
        }
    )


strongest_relation_df = pd.DataFrame(
    strongest_relation_rows
)


matrix = matrix.merge(

    strongest_relation_df,

    on="factor_id",

    how="left",

    validate="one_to_one",
)


# ==========================================================
# DEGRADATION PATHWAY EVIDENCE
# ==========================================================

DEGRADATION_FACTORS = {

    "REALLOCATED_SECTORS",

    "REPORTED_UNCORRECTABLE",

    "PENDING_SECTORS",
}


degradation_pairs = pcmci_pairs[
    (
        pcmci_pairs[
            "source_factor_id"
        ].isin(
            DEGRADATION_FACTORS
        )
    )
    &
    (
        pcmci_pairs[
            "target_factor_id"
        ].isin(
            DEGRADATION_FACTORS
        )
    )
].copy()


# ==========================================================
# HELPER FOR OPTIONAL LAG DISPLAY
# ==========================================================

def format_lag(
    row
):

    lag = np.nan


    if (
        "modal_lag_days"
        in row.index
        and
        pd.notna(
            row[
                "modal_lag_days"
            ]
        )
    ):

        lag = row[
            "modal_lag_days"
        ]


    elif (
        "strongest_lag_days"
        in row.index
        and
        pd.notna(
            row[
                "strongest_lag_days"
            ]
        )
    ):

        lag = row[
            "strongest_lag_days"
        ]


    if pd.isna(
        lag
    ):

        return (
            "lag varies"
        )


    return (
        f"lag {int(lag)} day(s)"
    )


# ==========================================================
# BUILD DEGRADATION PATHWAY DESCRIPTIONS
# ==========================================================

degradation_path_rows = []


for factor_id in matrix[
    "factor_id"
]:

    relevant = degradation_pairs[
        (
            degradation_pairs[
                "source_factor_id"
            ]
            ==
            factor_id
        )
        |
        (
            degradation_pairs[
                "target_factor_id"
            ]
            ==
            factor_id
        )
    ]


    descriptions = []


    for _, row in relevant.iterrows():

        descriptions.append(

            f"{row['source_factor_name']} -> "
            f"{row['target_factor_name']} "
            f"({format_lag(row)})"
        )


    degradation_path_rows.append(
        {
            "factor_id":
                factor_id,

            "degradation_temporal_pathway_available":
                len(
                    descriptions
                ) > 0,

            "degradation_temporal_pathway_count":
                len(
                    descriptions
                ),

            "degradation_temporal_pathways":
                (
                    "; ".join(
                        descriptions
                    )
                    if descriptions
                    else None
                ),
        }
    )


degradation_path_df = pd.DataFrame(
    degradation_path_rows
)


matrix = matrix.merge(

    degradation_path_df,

    on="factor_id",

    how="left",

    validate="one_to_one",
)


# ==========================================================
# MERGE MATCHED ATT EVIDENCE
# ==========================================================

att_columns = [

    "factor_id",

    "analysis_role",

    "matched_pairs",

    "treated_events",

    "control_events",

    "failure_risk_difference_pp",

    "failure_risk_ci_lower_pp",

    "failure_risk_ci_upper_pp",

    "mcnemar_exact_p_value",

    "max_abs_smd_after",

    "matched_effect_evidence_class",

    "interpretation",
]


att_small = att[
    att_columns
].copy()


att_small[
    "matched_att_available"
] = True


matrix = matrix.merge(

    att_small,

    on="factor_id",

    how="left",

    validate="one_to_one",
)


matrix[
    "matched_att_available"
] = (

    matrix[
        "matched_att_available"
    ]
    .fillna(
        False
    )
    .astype(
        bool
    )
)


matrix[
    "analysis_role"
] = matrix[
    "analysis_role"
].fillna(
    "NOT_EVALUATED"
)


matrix[
    "matched_effect_evidence_class"
] = matrix[
    "matched_effect_evidence_class"
].fillna(
    "NO_MATCHED_EFFECT_EVIDENCE"
)


# ==========================================================
# EVIDENCE STREAM COUNT
#
# SHAP exists for all factors.
#
# This count is descriptive only.
# It is NOT an evidence score.
# ==========================================================

matrix[
    "evidence_stream_count"
] = (

    matrix[
        "shap_available"
    ].astype(int)

    +

    matrix[
        "pcmci_available"
    ].astype(int)

    +

    matrix[
        "matched_att_available"
    ].astype(int)
)


# ==========================================================
# INTEGRATED ROOT-CAUSE CLASSIFICATION
#
# IMPORTANT:
#
# This is a transparent rule-based qualitative
# interpretation.
#
# No SHAP, PCMCI or ATT values are numerically added.
# ==========================================================

def integrated_classification(
    row
):

    semantic_status = str(
        row[
            "semantic_status"
        ]
    )


    att_class = str(
        row[
            "matched_effect_evidence_class"
        ]
    )


    temporal_available = bool(
        row[
            "pcmci_available"
        ]
    )


    # ------------------------------------------------------
    # Strong matched failure-risk evidence
    # ------------------------------------------------------

    if att_class == (
        "STRONG_MATCHED_FAILURE_RISK_EVIDENCE"
    ):

        return (
            "HIGH_PRIORITY_ROOT_CAUSE_CANDIDATE"
        )


    # ------------------------------------------------------
    # Supportive but statistically uncertain matched
    # failure-risk evidence
    # ------------------------------------------------------

    if att_class == (
        "SUPPORTIVE_BUT_STATISTICALLY_UNCERTAIN"
    ):

        return (
            "SUPPORTING_DEGRADATION_FACTOR"
        )


    # ------------------------------------------------------
    # Sparse exploratory matched evidence
    # ------------------------------------------------------

    if att_class == (
        "EXPLORATORY_SPARSE_EVIDENCE"
    ):

        return (
            "EXPLORATORY_ROOT_CAUSE_CANDIDATE"
        )


    # ------------------------------------------------------
    # Other matched evidence
    # ------------------------------------------------------

    if att_class in {

        "SUPPORTED_MATCHED_FAILURE_RISK_EVIDENCE",

        "INCONCLUSIVE_MATCHED_EVIDENCE",
    }:

        return (
            "MATCHED_FAILURE_RISK_CANDIDATE"
        )


    # ------------------------------------------------------
    # SMART semantics still require review
    # ------------------------------------------------------

    if semantic_status == "review":

        if temporal_available:

            return (
                "TEMPORAL_SIGNAL_REQUIRES_SEMANTIC_REVIEW"
            )


        return (
            "PREDICTIVE_SIGNAL_REQUIRES_SEMANTIC_REVIEW"
        )


    # ------------------------------------------------------
    # Predictive + PCMCI temporal evidence
    # but no matched failure-effect estimate
    # ------------------------------------------------------

    if temporal_available:

        return (
            "PREDICTIVE_TEMPORAL_CONTEXT"
        )


    # ------------------------------------------------------
    # SHAP predictive evidence only
    # ------------------------------------------------------

    return (
        "PREDICTIVE_ONLY"
    )


matrix[
    "integrated_evidence_class"
] = matrix.apply(

    integrated_classification,

    axis=1
)


# ==========================================================
# ROOT-CAUSE-CANDIDATE FLAG
# ==========================================================

ROOT_CAUSE_CLASSES = {

    "HIGH_PRIORITY_ROOT_CAUSE_CANDIDATE",

    "SUPPORTING_DEGRADATION_FACTOR",

    "EXPLORATORY_ROOT_CAUSE_CANDIDATE",

    "MATCHED_FAILURE_RISK_CANDIDATE",
}


matrix[
    "root_cause_candidate"
] = matrix[
    "integrated_evidence_class"
].isin(
    ROOT_CAUSE_CLASSES
)


# ==========================================================
# DISPLAY ORDER
#
# Display order only.
# NOT a numerical causal score.
# ==========================================================

CLASS_DISPLAY_ORDER = {

    "HIGH_PRIORITY_ROOT_CAUSE_CANDIDATE":
        1,

    "SUPPORTING_DEGRADATION_FACTOR":
        2,

    "MATCHED_FAILURE_RISK_CANDIDATE":
        3,

    "EXPLORATORY_ROOT_CAUSE_CANDIDATE":
        4,

    "PREDICTIVE_TEMPORAL_CONTEXT":
        5,

    "TEMPORAL_SIGNAL_REQUIRES_SEMANTIC_REVIEW":
        6,

    "PREDICTIVE_SIGNAL_REQUIRES_SEMANTIC_REVIEW":
        7,

    "PREDICTIVE_ONLY":
        8,
}


matrix[
    "integrated_display_order"
] = matrix[
    "integrated_evidence_class"
].map(
    CLASS_DISPLAY_ORDER
)


# ==========================================================
# SAFE HUMAN-READABLE EXPLANATION
# ==========================================================

def build_integrated_explanation(
    row
):

    factor_name = row[
        "factor_name"
    ]


    rank = int(
        row[
            "consensus_rank"
        ]
    )


    share = float(
        row[
            "mean_normalized_shap_share_pct"
        ]
    )


    stability = row[
        "rank_stability_label"
    ]


    evidence_class = row[
        "integrated_evidence_class"
    ]


    sections = []


    # ------------------------------------------------------
    # SHAP
    # ------------------------------------------------------

    sections.append(

        f"{factor_name} ranked #{rank} in the "
        f"global SHAP factor consensus, representing "
        f"{share:.2f}% of normalized predictive "
        f"importance across the population and "
        f"failure-focused analyses; rank stability "
        f"was {stability}."
    )


    # ------------------------------------------------------
    # PCMCI
    # ------------------------------------------------------

    if row[
        "pcmci_available"
    ]:

        degree = int(
            row[
                "stable_cross_factor_degree"
            ]
        )


        sections.append(

            f"PCMCI stability analysis identified "
            f"{degree} stable cross-factor temporal "
            f"connection(s) involving this factor."
        )


        if pd.notna(
            row[
                "strongest_temporal_relation"
            ]
        ):

            sections.append(

                f"Its strongest stable temporal "
                f"relationship was "
                f"{row['strongest_temporal_relation']}."
            )


    else:

        sections.append(

            "No stable cross-factor PCMCI relationship "
            "was retained for this factor under the "
            "selected stability criterion."
        )


    # ------------------------------------------------------
    # Degradation pathway
    # ------------------------------------------------------

    if row[
        "degradation_temporal_pathway_available"
    ]:

        sections.append(

            "A degradation-focused temporal pathway "
            f"was observed: "
            f"{row['degradation_temporal_pathways']}."
        )


    # ------------------------------------------------------
    # Matched ATT
    # ------------------------------------------------------

    if row[
        "matched_att_available"
    ]:

        effect = float(
            row[
                "failure_risk_difference_pp"
            ]
        )


        lower = float(
            row[
                "failure_risk_ci_lower_pp"
            ]
        )


        upper = float(
            row[
                "failure_risk_ci_upper_pp"
            ]
        )


        sections.append(

            f"The matched incident analysis estimated "
            f"a {effect:.2f} percentage-point "
            f"30-day failure-risk difference "
            f"(clustered-bootstrap 95% CI "
            f"{lower:.2f} to {upper:.2f})."
        )


    else:

        sections.append(

            "No matched incident failure-effect estimate "
            "was performed for this factor."
        )


    # ------------------------------------------------------
    # Semantic caution
    # ------------------------------------------------------

    if row[
        "semantic_status"
    ] == "review":

        sections.append(

            "The SMART semantic interpretation for this "
            "factor remains marked for review, so it "
            "should not be promoted to a physical "
            "root-cause claim without vendor-specific "
            "validation."
        )


    # ------------------------------------------------------
    # Integrated conclusion
    # ------------------------------------------------------

    if evidence_class == (
        "HIGH_PRIORITY_ROOT_CAUSE_CANDIDATE"
    ):

        sections.append(

            "Overall, this is a high-priority "
            "root-cause candidate because predictive "
            "evidence is complemented by strong matched "
            "failure-risk evidence. The conclusion "
            "remains observational rather than proof "
            "of physical causation."
        )


    elif evidence_class == (
        "SUPPORTING_DEGRADATION_FACTOR"
    ):

        sections.append(

            "Overall, this is a supporting degradation "
            "factor. The direction of matched "
            "failure-risk evidence is supportive, but "
            "statistical uncertainty prevents a strong "
            "direct-cause conclusion."
        )


    elif evidence_class == (
        "EXPLORATORY_ROOT_CAUSE_CANDIDATE"
    ):

        sections.append(

            "Overall, this remains an exploratory "
            "root-cause candidate because the matched "
            "effect analysis contained very few "
            "failure outcomes."
        )


    elif evidence_class == (
        "PREDICTIVE_TEMPORAL_CONTEXT"
    ):

        sections.append(

            "Overall, this factor should be interpreted "
            "as predictive and temporal operational "
            "context rather than as an established "
            "failure root cause."
        )


    elif evidence_class in {

        "PREDICTIVE_SIGNAL_REQUIRES_SEMANTIC_REVIEW",

        "TEMPORAL_SIGNAL_REQUIRES_SEMANTIC_REVIEW",
    }:

        sections.append(

            "Overall, this factor is retained as an "
            "explanatory signal, but root-cause "
            "interpretation should wait for semantic "
            "validation."
        )


    else:

        sections.append(

            "Overall, current evidence supports use of "
            "this factor as a predictive signal, not "
            "as an established root cause."
        )


    return (
        " ".join(
            sections
        )
    )


matrix[
    "integrated_explanation"
] = matrix.apply(

    build_integrated_explanation,

    axis=1
)


# ==========================================================
# SORT FINAL MATRIX
# ==========================================================

matrix = (

    matrix

    .sort_values(
        [
            "integrated_display_order",

            "consensus_rank",
        ],
        ascending=[
            True,
            True,
        ]
    )

    .reset_index(
        drop=True
    )
)


# ==========================================================
# ROOT-CAUSE CANDIDATE SUMMARY
# ==========================================================

root_cause_candidates = matrix[
    matrix[
        "root_cause_candidate"
    ]
].copy()


# ==========================================================
# CONTEXT FACTOR SUMMARY
# ==========================================================

context_factors = matrix[
    ~matrix[
        "root_cause_candidate"
    ]
].copy()


# ==========================================================
# DEGRADATION PATHWAY TABLE
# ==========================================================

degradation_pathway_output = (
    degradation_pairs.copy()
)


if len(
    degradation_pathway_output
) > 0:

    degradation_pathway_output[
        "pathway_interpretation"
    ] = (

        "Stable PCMCI temporal-dependency pathway "
        "between degradation-related HDD factors. "
        "This supports temporal ordering evidence "
        "but does not by itself establish a physical "
        "causal mechanism."
    )


# ==========================================================
# INTEGRATED CLASS SUMMARY
# ==========================================================

class_summary = (

    matrix[
        "integrated_evidence_class"
    ]

    .value_counts()

    .rename_axis(
        "integrated_evidence_class"
    )

    .reset_index(
        name="factor_count"
    )
)


# ==========================================================
# EVIDENCE STREAM SUMMARY
# ==========================================================

stream_summary = pd.DataFrame(
    [
        {
            "evidence_stream":
                "GLOBAL_SHAP",

            "factor_count":
                int(
                    matrix[
                        "shap_available"
                    ].sum()
                ),

            "meaning":
                (
                    "Predictive contribution to the "
                    "30-day failure model."
                ),
        },

        {
            "evidence_stream":
                "PCMCI",

            "factor_count":
                int(
                    matrix[
                        "pcmci_available"
                    ].sum()
                ),

            "meaning":
                (
                    "Stable lagged conditional-dependency "
                    "relationships across PCMCI subsamples."
                ),
        },

        {
            "evidence_stream":
                "MATCHED_ATT",

            "factor_count":
                int(
                    matrix[
                        "matched_att_available"
                    ].sum()
                ),

            "meaning":
                (
                    "Matched observational 30-day "
                    "failure-risk contrast."
                ),
        },
    ]
)


# ==========================================================
# SAVE OUTPUTS
# ==========================================================

matrix_file = (
    OUTPUT_DIR
    / "integrated_root_cause_evidence_matrix.csv"
)


candidate_file = (
    OUTPUT_DIR
    / "root_cause_candidate_summary.csv"
)


context_file = (
    OUTPUT_DIR
    / "predictive_temporal_context_summary.csv"
)


pathway_file = (
    OUTPUT_DIR
    / "degradation_temporal_pathways.csv"
)


class_summary_file = (
    OUTPUT_DIR
    / "integrated_evidence_class_summary.csv"
)


stream_summary_file = (
    OUTPUT_DIR
    / "integrated_evidence_stream_summary.csv"
)


matrix.to_csv(
    matrix_file,
    index=False
)


root_cause_candidates.to_csv(
    candidate_file,
    index=False
)


context_factors.to_csv(
    context_file,
    index=False
)


degradation_pathway_output.to_csv(
    pathway_file,
    index=False
)


class_summary.to_csv(
    class_summary_file,
    index=False
)


stream_summary.to_csv(
    stream_summary_file,
    index=False
)


# ==========================================================
# METADATA
# ==========================================================

metadata = {

    "framework_name":
        "Integrated Causal-XAI Root-Cause Evidence Framework",

    "evidence_streams":
        [
            "Global SHAP predictive evidence",

            "PCMCI temporal-dependency evidence",

            "Matched incident ATT failure-risk evidence",
        ],

    "integration_method":
        (
            "Rule-based qualitative integration. "
            "Evidence-stream values are preserved "
            "separately and are not added into a "
            "numerical causal score."
        ),

    "root_cause_language":
        (
            "Root-cause candidate / supporting factor "
            "language is used because observational data "
            "cannot prove physical causation."
        ),

    "shap_interpretation":
        (
            "Predictive attribution; not causal evidence."
        ),

    "pcmci_interpretation":
        (
            "Stable lagged conditional dependencies "
            "under PCMCI assumptions; not direct "
            "physical causal proof."
        ),

    "matched_att_interpretation":
        (
            "Observational matched 30-day failure-risk "
            "contrast under measured-confounding and "
            "matching assumptions."
        ),
}


metadata_file = (
    OUTPUT_DIR
    / "integrated_root_cause_metadata.json"
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
# PRINT FINAL RESULTS
# ==========================================================

print("\n")
print("=" * 84)

print(
    "STEP 13F - INTEGRATED ROOT-CAUSE EVIDENCE MATRIX COMPLETE"
)

print("=" * 84)


print(
    "\nIntegrated evidence matrix:\n"
)


display_columns = [

    "factor_name",

    "consensus_rank",

    "predictive_evidence_band",

    "stable_cross_factor_degree",

    "matched_att_available",

    "matched_effect_evidence_class",

    "evidence_stream_count",

    "integrated_evidence_class",
]


print(
    matrix[
        display_columns
    ].to_string(
        index=False
    )
)


print(
    "\nRoot-cause candidates:\n"
)


if len(
    root_cause_candidates
) > 0:

    candidate_columns = [

        "factor_name",

        "consensus_rank",

        "stable_cross_factor_degree",

        "degradation_temporal_pathway_available",

        "failure_risk_difference_pp",

        "failure_risk_ci_lower_pp",

        "failure_risk_ci_upper_pp",

        "integrated_evidence_class",
    ]


    print(
        root_cause_candidates[
            candidate_columns
        ].to_string(
            index=False
        )
    )

else:

    print(
        "No factors currently satisfy "
        "root-cause-candidate rules."
    )


print(
    "\nDegradation temporal pathways:\n"
)


if len(
    degradation_pathway_output
) > 0:

    pathway_display_columns = [

        "source_factor_name",

        "target_factor_name",

        "temporal_pair_rank",

        "temporal_magnitude",
    ]


    for optional_column in [

        "modal_lag_days",

        "strongest_lag_days",

        "temporal_direction",
    ]:

        if (
            optional_column
            in degradation_pathway_output.columns
        ):

            pathway_display_columns.append(
                optional_column
            )


    print(
        degradation_pathway_output[
            pathway_display_columns
        ].to_string(
            index=False
        )
    )

else:

    print(
        "No degradation-specific PCMCI "
        "pathways were retained."
    )


print(
    "\nIntegrated evidence classes:\n"
)


print(
    class_summary.to_string(
        index=False
    )
)


print(
    "\nEvidence-stream coverage:\n"
)


print(
    stream_summary[
        [
            "evidence_stream",

            "factor_count",
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