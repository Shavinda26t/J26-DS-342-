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

INTEGRATED_DIR = (
    DATA_ROOT
    / "interim"
    / "integrated_evidence"
)

OUTPUT_DIR = (
    RESULTS_ROOT
    / "integrated_evidence"
    / "matched_att"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# INPUT FILES
# ==========================================================

ATT_SUMMARY_FILE = (
    RESULTS_ROOT
    / "causal_effect"
    / "matched_att"
    / "matched_att_effect_summary.csv"
)

ATT_ROLES_FILE = (
    RESULTS_ROOT
    / "causal_effect"
    / "matched_att"
    / "causal_effect_analysis_roles.csv"
)

ATT_ROBUSTNESS_FILE = (
    RESULTS_ROOT
    / "causal_effect"
    / "matched_att_robustness"
    / "clustered_att_robustness_summary.csv"
)

MATCHING_SUMMARY_FILE = (
    DATA_ROOT
    / "interim"
    / "riskset_matching_refined"
    / "history_corrected_matching_summary.csv"
)

ATT_MAPPING_FILE = (
    INTEGRATED_DIR
    / "att_treatment_to_factor_mapping.csv"
)

FACTOR_MANIFEST_FILE = (
    INTEGRATED_DIR
    / "canonical_factor_manifest.csv"
)


# ==========================================================
# VALIDATE INPUT FILES
# ==========================================================

required_files = [

    ATT_SUMMARY_FILE,

    ATT_ROLES_FILE,

    ATT_ROBUSTNESS_FILE,

    MATCHING_SUMMARY_FILE,

    ATT_MAPPING_FILE,

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
        "Missing required files:\n"
        +
        "\n".join(
            missing_files
        )
    )


# ==========================================================
# LOAD DATA
# ==========================================================

att = pd.read_csv(
    ATT_SUMMARY_FILE
)


roles = pd.read_csv(
    ATT_ROLES_FILE
)


robustness = pd.read_csv(
    ATT_ROBUSTNESS_FILE
)


matching = pd.read_csv(
    MATCHING_SUMMARY_FILE
)


mapping = pd.read_csv(
    ATT_MAPPING_FILE
)


manifest = pd.read_csv(
    FACTOR_MANIFEST_FILE
)


# ==========================================================
# REQUIRED COLUMN VALIDATION
# ==========================================================

required_att_columns = {

    "treatment",

    "matched_pairs",

    "treated_events",

    "control_events",

    "treated_risk",

    "control_risk",

    "att_risk_difference_percentage_points",

    "bootstrap_95ci_lower_percentage_points",

    "bootstrap_95ci_upper_percentage_points",

    "mcnemar_exact_p_value",
}


required_robustness_columns = {

    "treatment",

    "unique_control_clusters",

    "reused_control_hdds",

    "maximum_control_reuse",

    "att_percentage_points",

    "cluster_bootstrap_ci_lower_pp",

    "cluster_bootstrap_ci_upper_pp",

    "bootstrap_positive_fraction",
}


required_matching_columns = {

    "treatment",

    "clean_incident_treated",

    "matched_treated",

    "unmatched_treated",

    "retention_pct",

    "treated_positive_rows",

    "control_positive_rows",

    "max_abs_smd_before",

    "max_abs_smd_after",

    "features_abs_smd_ge_010_after",

    "status",
}


required_mapping_columns = {

    "treatment",

    "mapped",

    "factor_id",

    "factor_name",

    "category",
}


for label, frame, required_columns in [

    (
        "ATT summary",
        att,
        required_att_columns
    ),

    (
        "ATT robustness",
        robustness,
        required_robustness_columns
    ),

    (
        "matching summary",
        matching,
        required_matching_columns
    ),

    (
        "ATT mapping",
        mapping,
        required_mapping_columns
    ),
]:

    missing = (
        required_columns
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
# KEEP ONLY VALID MAPPED TREATMENTS
# ==========================================================

mapping = mapping[
    mapping[
        "mapped"
    ] == True
].copy()

# ==========================================================
# RENAME OVERLAPPING ROBUSTNESS COLUMNS
#
# Step 12G and Step 12H contain several columns with the
# same names. Rename the Step 12H copies explicitly before
# merging so pandas does not create _x / _y columns.
# ==========================================================

robustness_for_merge = robustness.rename(
    columns={

        "matched_pairs":
            "robustness_matched_pairs",

        "treated_risk":
            "robustness_treated_risk",

        "control_risk":
            "robustness_control_risk",

        "bootstrap_positive_fraction":
            "robustness_bootstrap_positive_fraction",

        "maximum_control_reuse":
            "robustness_maximum_control_reuse",

        "att_risk_difference":
            "robustness_att_risk_difference",
    }
)


evidence = att.merge(

    robustness_for_merge,

    on="treatment",

    how="inner",

    validate="one_to_one",
)


# ==========================================================
# CROSS-CHECK DUPLICATED VALUES
#
# These values should agree between Step 12G and Step 12H.
# ==========================================================

matched_pair_mismatch = (

    evidence[
        "matched_pairs"
    ].astype(int)

    !=

    evidence[
        "robustness_matched_pairs"
    ].astype(int)
)


if matched_pair_mismatch.any():

    raise RuntimeError(
        "Matched-pair count differs between "
        "ATT and robustness results."
    )


treated_risk_mismatch = ~np.isclose(

    evidence[
        "treated_risk"
    ].astype(float),

    evidence[
        "robustness_treated_risk"
    ].astype(float),

    rtol=1e-10,

    atol=1e-12,
)


if treated_risk_mismatch.any():

    raise RuntimeError(
        "Treated risk differs between "
        "ATT and robustness results."
    )


control_risk_mismatch = ~np.isclose(

    evidence[
        "control_risk"
    ].astype(float),

    evidence[
        "robustness_control_risk"
    ].astype(float),

    rtol=1e-10,

    atol=1e-12,
)


if control_risk_mismatch.any():

    raise RuntimeError(
        "Control risk differs between "
        "ATT and robustness results."
    )


# ==========================================================
# MERGE MATCHING QUALITY
# ==========================================================

evidence = evidence.merge(

    matching,

    on="treatment",

    how="inner",

    validate="one_to_one",
)


# ==========================================================
# MERGE ANALYSIS ROLES
# ==========================================================

evidence = evidence.merge(

    roles[
        [
            "treatment",

            "analysis_role",

            "reason",
        ]
    ],

    on="treatment",

    how="left",

    validate="one_to_one",
)


# ==========================================================
# MERGE CANONICAL FACTOR MAPPING
# ==========================================================

evidence = evidence.merge(

    mapping[
        [
            "treatment",

            "factor_id",

            "factor_name",

            "category",
        ]
    ],

    on="treatment",

    how="left",

    validate="one_to_one",
)


# ==========================================================
# FINAL MERGE VALIDATION
# ==========================================================

expected_treatments = set(
    mapping[
        "treatment"
    ].astype(str)
)


merged_treatments = set(
    evidence[
        "treatment"
    ].astype(str)
)


missing_after_merge = (

    expected_treatments

    -

    merged_treatments
)


if missing_after_merge:

    raise RuntimeError(
        "Treatments disappeared during evidence merge:\n"
        +
        "\n".join(
            sorted(
                missing_after_merge
            )
        )
    )


print(
    "\nMerged ATT evidence rows:",
    len(
        evidence
    )
)


print(
    "Merged treatments:",
    ", ".join(
        evidence[
            "treatment"
        ].astype(str)
    )
)


print(
    "ATT/robustness consistency checks: PASS"
)


# ==========================================================
# VALIDATE CANONICAL MAPPING
# ==========================================================

if evidence[
    "factor_id"
].isna().any():

    raise RuntimeError(
        "Some ATT treatments were not mapped "
        "to canonical factors."
    )


# ==========================================================
# ADD SEMANTIC STATUS
# ==========================================================

semantic_lookup = (

    manifest

    .set_index(
        "factor_id"
    )[
        "semantic_status"
    ]

    .to_dict()
)


evidence[
    "semantic_status"
] = evidence[
    "factor_id"
].map(
    semantic_lookup
)


# ==========================================================
# STANDARDIZED PRIMARY EFFECT VALUES
#
# Clustered bootstrap interval is preferred because
# some control HDDs were reused in matching.
# ==========================================================

evidence[
    "failure_risk_difference_pp"
] = pd.to_numeric(

    evidence[
        "att_percentage_points"
    ],

    errors="raise"
)


evidence[
    "failure_risk_ci_lower_pp"
] = pd.to_numeric(

    evidence[
        "cluster_bootstrap_ci_lower_pp"
    ],

    errors="raise"
)


evidence[
    "failure_risk_ci_upper_pp"
] = pd.to_numeric(

    evidence[
        "cluster_bootstrap_ci_upper_pp"
    ],

    errors="raise"
)


# ==========================================================
# EFFECT DIRECTION
# ==========================================================

evidence[
    "effect_direction"
] = np.select(

    [

        evidence[
            "failure_risk_difference_pp"
        ] > 0,

        evidence[
            "failure_risk_difference_pp"
        ] < 0,
    ],

    [

        "HIGHER_30D_FAILURE_RISK",

        "LOWER_30D_FAILURE_RISK",
    ],

    default="NO_OBSERVED_DIFFERENCE",
)


# ==========================================================
# CI ZERO CHECK
# ==========================================================

evidence[
    "cluster_ci_excludes_zero"
] = (

    (
        evidence[
            "failure_risk_ci_lower_pp"
        ] > 0
    )

    |

    (
        evidence[
            "failure_risk_ci_upper_pp"
        ] < 0
    )
)


# ==========================================================
# PAIRED TEST SUPPORT
# ==========================================================

evidence[
    "mcnemar_support_005"
] = (

    evidence[
        "mcnemar_exact_p_value"
    ]
    < 0.05
)


# ==========================================================
# MATCHING QUALITY
# ==========================================================

evidence[
    "balance_pass"
] = (

    evidence[
        "max_abs_smd_after"
    ]
    < 0.10
)


evidence[
    "retention_pass"
] = (

    evidence[
        "retention_pct"
    ]
    >= 80.0
)


evidence[
    "matching_quality_pass"
] = (

    evidence[
        "balance_pass"
    ]

    &

    evidence[
        "retention_pass"
    ]
)


# ==========================================================
# SPARSE-OUTCOME FLAGS
#
# These are not universal statistical rules.
# They are transparent study-specific evidence labels.
# ==========================================================

evidence[
    "treated_event_count"
] = evidence[
    "treated_events"
].astype(
    int
)


evidence[
    "very_sparse_treated_outcomes"
] = (

    evidence[
        "treated_event_count"
    ]
    < 5
)


evidence[
    "sparse_treated_outcomes"
] = (

    evidence[
        "treated_event_count"
    ]
    < 10
)


# ==========================================================
# EVIDENCE CLASSIFICATION
#
# IMPORTANT:
# This is a qualitative reporting category.
#
# It is NOT a numerical causal score.
# ==========================================================

def classify_evidence(
    row
):

    role = str(
        row[
            "analysis_role"
        ]
    )


    events = int(
        row[
            "treated_event_count"
        ]
    )


    matching_ok = bool(
        row[
            "matching_quality_pass"
        ]
    )


    ci_support = bool(
        row[
            "cluster_ci_excludes_zero"
        ]
    )


    paired_support = bool(
        row[
            "mcnemar_support_005"
        ]
    )


    # ------------------------------------------------------
    # Primary result
    # ------------------------------------------------------

    if (
        role == "PRIMARY"
        and
        matching_ok
        and
        events >= 10
        and
        ci_support
        and
        paired_support
    ):

        return (
            "STRONG_MATCHED_FAILURE_RISK_EVIDENCE"
        )


    # ------------------------------------------------------
    # Positive bootstrap evidence but exact paired test
    # remains uncertain / sparse.
    # ------------------------------------------------------

    if (
        matching_ok
        and
        ci_support
        and
        not paired_support
        and
        events >= 5
    ):

        return (
            "SUPPORTIVE_BUT_STATISTICALLY_UNCERTAIN"
        )


    # ------------------------------------------------------
    # Very small event sample
    # ------------------------------------------------------

    if (
        events < 5
    ):

        return (
            "EXPLORATORY_SPARSE_EVIDENCE"
        )


    # ------------------------------------------------------
    # General balanced result
    # ------------------------------------------------------

    if (
        matching_ok
        and
        ci_support
        and
        paired_support
    ):

        return (
            "SUPPORTED_MATCHED_FAILURE_RISK_EVIDENCE"
        )


    return (
        "INCONCLUSIVE_MATCHED_EVIDENCE"
    )


evidence[
    "matched_effect_evidence_class"
] = evidence.apply(

    classify_evidence,

    axis=1
)


# ==========================================================
# EVIDENCE CONFIDENCE ORDER
#
# Used only for display/ranking categories.
# NOT an effect-size score.
# ==========================================================

EVIDENCE_ORDER = {

    "STRONG_MATCHED_FAILURE_RISK_EVIDENCE":
        1,

    "SUPPORTED_MATCHED_FAILURE_RISK_EVIDENCE":
        2,

    "SUPPORTIVE_BUT_STATISTICALLY_UNCERTAIN":
        3,

    "EXPLORATORY_SPARSE_EVIDENCE":
        4,

    "INCONCLUSIVE_MATCHED_EVIDENCE":
        5,
}


evidence[
    "evidence_display_order"
] = evidence[
    "matched_effect_evidence_class"
].map(
    EVIDENCE_ORDER
)


# ==========================================================
# HUMAN-READABLE INTERPRETATION
# ==========================================================

def build_interpretation(
    row
):

    factor = row[
        "factor_name"
    ]


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


    pairs = int(
        row[
            "matched_pairs"
        ]
    )


    treated_events = int(
        row[
            "treated_events"
        ]
    )


    controls = int(
        row[
            "control_events"
        ]
    )


    evidence_class = row[
        "matched_effect_evidence_class"
    ]


    if evidence_class == (
        "STRONG_MATCHED_FAILURE_RISK_EVIDENCE"
    ):

        conclusion = (
            "This provides the strongest matched "
            "failure-risk evidence among the evaluated "
            "degradation factors."
        )


    elif evidence_class == (
        "SUPPORTIVE_BUT_STATISTICALLY_UNCERTAIN"
    ):

        conclusion = (
            "The direction is supportive, but sparse "
            "failures and the exact paired test mean "
            "the result should be treated as uncertain."
        )


    elif evidence_class == (
        "EXPLORATORY_SPARSE_EVIDENCE"
    ):

        conclusion = (
            "The estimate is exploratory because very "
            "few treated failure outcomes were observed."
        )


    else:

        conclusion = (
            "The matched evidence is not sufficient "
            "for a strong failure-risk conclusion."
        )


    return (

        f"{factor}: among {pairs} matched incident pairs, "
        f"{treated_events} treated HDDs and {controls} "
        f"matched controls failed within 30 days. "
        f"The estimated matched risk difference was "
        f"{effect:.2f} percentage points "
        f"(clustered-bootstrap 95% CI "
        f"{lower:.2f} to {upper:.2f}). "
        f"{conclusion}"
    )


evidence[
    "interpretation"
] = evidence.apply(

    build_interpretation,

    axis=1
)


# ==========================================================
# SAFE CAUSAL LANGUAGE
# ==========================================================

evidence[
    "interpretation_scope"
] = (

    "Observational matched failure-risk contrast under "
    "measured-confounding, incident-risk-set, matching, "
    "and temporal-ordering assumptions; not proof of "
    "physical causation."
)


# ==========================================================
# SELECT STANDARDIZED OUTPUT
# ==========================================================

standardized_columns = [

    "factor_id",

    "factor_name",

    "category",

    "semantic_status",

    "treatment",

    "analysis_role",

    "matched_pairs",

    "treated_events",

    "control_events",

    "treated_risk",

    "control_risk",

    "failure_risk_difference_pp",

    "failure_risk_ci_lower_pp",

    "failure_risk_ci_upper_pp",

    "bootstrap_positive_fraction",

    "mcnemar_exact_p_value",

    "cluster_ci_excludes_zero",

    "mcnemar_support_005",

    "max_abs_smd_after",

    "features_abs_smd_ge_010_after",

    "retention_pct",

    "matching_quality_pass",

    "unique_control_clusters",

    "reused_control_hdds",

    "maximum_control_reuse",

    "effect_direction",

    "treated_event_count",

    "sparse_treated_outcomes",

    "very_sparse_treated_outcomes",

    "matched_effect_evidence_class",

    "evidence_display_order",

    "interpretation",

    "interpretation_scope",
]


standardized = evidence[
    standardized_columns
].copy()


standardized = (

    standardized

    .sort_values(
        [
            "evidence_display_order",

            "failure_risk_difference_pp",
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


# ==========================================================
# CANONICAL FACTOR AVAILABILITY
#
# This produces one row for all 16 factors and marks which
# factors currently have matched failure-effect evidence.
# ==========================================================

availability_rows = []


for _, factor in manifest.iterrows():

    factor_id = factor[
        "factor_id"
    ]


    matched_row = standardized[
        standardized[
            "factor_id"
        ]
        ==
        factor_id
    ]


    if len(
        matched_row
    ) == 1:

        row = matched_row.iloc[
            0
        ]


        availability_rows.append(
            {
                "factor_id":
                    factor_id,

                "factor_name":
                    factor[
                        "factor_name"
                    ],

                "matched_att_available":
                    True,

                "analysis_role":
                    row[
                        "analysis_role"
                    ],

                "failure_risk_difference_pp":
                    row[
                        "failure_risk_difference_pp"
                    ],

                "failure_risk_ci_lower_pp":
                    row[
                        "failure_risk_ci_lower_pp"
                    ],

                "failure_risk_ci_upper_pp":
                    row[
                        "failure_risk_ci_upper_pp"
                    ],

                "matched_effect_evidence_class":
                    row[
                        "matched_effect_evidence_class"
                    ],
            }
        )


    else:

        availability_rows.append(
            {
                "factor_id":
                    factor_id,

                "factor_name":
                    factor[
                        "factor_name"
                    ],

                "matched_att_available":
                    False,

                "analysis_role":
                    "NOT_EVALUATED",

                "failure_risk_difference_pp":
                    np.nan,

                "failure_risk_ci_lower_pp":
                    np.nan,

                "failure_risk_ci_upper_pp":
                    np.nan,

                "matched_effect_evidence_class":
                    "NO_MATCHED_EFFECT_EVIDENCE",
            }
        )


availability_df = pd.DataFrame(
    availability_rows
)


# ==========================================================
# SUMMARY METRICS
# ==========================================================

summary_rows = [

    {
        "metric":
            "matched_effect_factors",

        "value":
            len(
                standardized
            ),
    },

    {
        "metric":
            "strong_evidence_factors",

        "value":
            int(
                (
                    standardized[
                        "matched_effect_evidence_class"
                    ]
                    ==
                    "STRONG_MATCHED_FAILURE_RISK_EVIDENCE"
                ).sum()
            ),
    },

    {
        "metric":
            "supportive_uncertain_factors",

        "value":
            int(
                (
                    standardized[
                        "matched_effect_evidence_class"
                    ]
                    ==
                    "SUPPORTIVE_BUT_STATISTICALLY_UNCERTAIN"
                ).sum()
            ),
    },

    {
        "metric":
            "exploratory_sparse_factors",

        "value":
            int(
                (
                    standardized[
                        "matched_effect_evidence_class"
                    ]
                    ==
                    "EXPLORATORY_SPARSE_EVIDENCE"
                ).sum()
            ),
    },

    {
        "metric":
            "all_matching_balance_pass",

        "value":
            bool(
                standardized[
                    "matching_quality_pass"
                ].all()
            ),
    },
]


summary_df = pd.DataFrame(
    summary_rows
)


# ==========================================================
# SAVE OUTPUTS
# ==========================================================

standardized_file = (
    OUTPUT_DIR
    / "factor_matched_att_evidence.csv"
)


availability_file = (
    OUTPUT_DIR
    / "factor_matched_att_availability.csv"
)


summary_file = (
    OUTPUT_DIR
    / "matched_att_evidence_summary.csv"
)


standardized.to_csv(
    standardized_file,
    index=False
)


availability_df.to_csv(
    availability_file,
    index=False
)


summary_df.to_csv(
    summary_file,
    index=False
)


# ==========================================================
# METADATA
# ==========================================================

metadata = {

    "primary_estimand":
        "matched ATT risk difference",

    "unit":
        "percentage points of 30-day failure risk",

    "preferred_interval":
        "control-HDD clustered bootstrap 95% CI",

    "paired_test":
        "exact McNemar test",

    "matching_balance_threshold":
        "maximum absolute SMD < 0.10",

    "retention_threshold":
        ">= 80% treated incident HDDs retained",

    "evidence_classification":
        (
            "Qualitative reporting categories based on "
            "analysis role, event count, matched balance, "
            "clustered CI and exact paired-test evidence. "
            "Not a numerical causal score."
        ),

    "causal_interpretation":
        (
            "Observational matched failure-risk evidence "
            "under measured-confounding and matching "
            "assumptions; not proof of physical causation."
        ),
}


metadata_file = (
    OUTPUT_DIR
    / "matched_att_evidence_metadata.json"
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
    "STEP 13E - MATCHED ATT EVIDENCE STANDARDIZATION COMPLETE"
)

print("=" * 80)


print(
    "\nStandardized matched failure-risk evidence:\n"
)


print(
    standardized[
        [
            "factor_name",

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
        ]
    ].to_string(
        index=False
    )
)


print(
    "\nEvidence summary:\n"
)


print(
    summary_df.to_string(
        index=False
    )
)


print(
    "\nInterpretations:\n"
)


for _, row in standardized.iterrows():

    print(
        "-",
        row[
            "interpretation"
        ]
    )


print(
    "\nOutputs saved to:"
)


print(
    OUTPUT_DIR
)