from pathlib import Path
import re

import pandas as pd


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_ROOT = (
    PROJECT_ROOT
    / "results"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "integrated_evidence"
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

LOCAL_SHAP_DIR = (
    RESULTS_ROOT
    / "shap"
    / "30d"
    / "local"
)

PCMCI_STABLE_EXACT_FILE = (
    RESULTS_ROOT
    / "causal_discovery"
    / "pcmci_stability"
    / "stable_exact_links.csv"
)

PCMCI_STABLE_PAIR_FILE = (
    RESULTS_ROOT
    / "causal_discovery"
    / "pcmci_stability"
    / "stable_pair_links.csv"
)

ATT_SUMMARY_FILE = (
    RESULTS_ROOT
    / "causal_effect"
    / "matched_att"
    / "matched_att_effect_summary.csv"
)

ATT_ROBUSTNESS_FILE = (
    RESULTS_ROOT
    / "causal_effect"
    / "matched_att_robustness"
    / "clustered_att_robustness_summary.csv"
)


# ==========================================================
# CANONICAL FACTOR MANIFEST
#
# Important:
# semantic_status='review' means:
#
# Keep the factor for predictive explanation,
# but do not give it strong causal/physical interpretation
# until its vendor-specific SMART semantics are validated.
# ==========================================================

CANONICAL_FACTORS = [

    {
        "factor_id":
            "READ_ERROR_RATE",

        "factor_name":
            "Read Error Rate",

        "smart_id":
            1,

        "smart_raw_feature":
            "smart_1_raw",

        "category":
            "error_indicator",

        "semantic_status":
            "review",

        "causal_discovery_initially_eligible":
            False,
    },

    {
        "factor_id":
            "START_STOP_ACTIVITY",

        "factor_name":
            "Start/Stop Activity",

        "smart_id":
            4,

        "smart_raw_feature":
            "smart_4_raw",

        "category":
            "usage_context",

        "semantic_status":
            "validated_for_analysis",

        "causal_discovery_initially_eligible":
            True,
    },

    {
        "factor_id":
            "REALLOCATED_SECTORS",

        "factor_name":
            "Reallocated Sectors",

        "smart_id":
            5,

        "smart_raw_feature":
            "smart_5_raw",

        "category":
            "degradation_indicator",

        "semantic_status":
            "validated_for_analysis",

        "causal_discovery_initially_eligible":
            True,
    },

    {
        "factor_id":
            "SEEK_ERROR_RATE",

        "factor_name":
            "Seek Error Rate",

        "smart_id":
            7,

        "smart_raw_feature":
            "smart_7_raw",

        "category":
            "error_indicator",

        "semantic_status":
            "review",

        "causal_discovery_initially_eligible":
            False,
    },

    {
        "factor_id":
            "DRIVE_AGE",

        "factor_name":
            "Drive Age / Power-On Hours",

        "smart_id":
            9,

        "smart_raw_feature":
            "smart_9_raw",

        "category":
            "age_context",

        "semantic_status":
            "validated_for_analysis",

        "causal_discovery_initially_eligible":
            True,
    },

    {
        "factor_id":
            "POWER_CYCLE_ACTIVITY",

        "factor_name":
            "Power Cycle Activity",

        "smart_id":
            12,

        "smart_raw_feature":
            "smart_12_raw",

        "category":
            "usage_context",

        "semantic_status":
            "validated_for_analysis",

        "causal_discovery_initially_eligible":
            True,
    },

    {
        "factor_id":
            "REPORTED_UNCORRECTABLE",

        "factor_name":
            "Reported Uncorrectable Errors",

        "smart_id":
            187,

        "smart_raw_feature":
            "smart_187_raw",

        "category":
            "error_indicator",

        "semantic_status":
            "validated_for_analysis",

        "causal_discovery_initially_eligible":
            True,
    },

    {
        "factor_id":
            "COMMAND_TIMEOUT",

        "factor_name":
            "Command Timeout / SMART 188",

        "smart_id":
            188,

        "smart_raw_feature":
            "smart_188_raw",

        "category":
            "error_indicator",

        "semantic_status":
            "review",

        "causal_discovery_initially_eligible":
            False,
    },

    {
        "factor_id":
            "POWEROFF_RETRACT_ACTIVITY",

        "factor_name":
            "Power-Off Retract Activity",

        "smart_id":
            192,

        "smart_raw_feature":
            "smart_192_raw",

        "category":
            "usage_context",

        "semantic_status":
            "validated_for_analysis",

        "causal_discovery_initially_eligible":
            True,
    },

    {
        "factor_id":
            "LOAD_CYCLE_ACTIVITY",

        "factor_name":
            "Load Cycle Activity",

        "smart_id":
            193,

        "smart_raw_feature":
            "smart_193_raw",

        "category":
            "mechanical_usage",

        "semantic_status":
            "validated_for_analysis",

        "causal_discovery_initially_eligible":
            True,
    },

    {
        "factor_id":
            "TEMPERATURE",

        "factor_name":
            "Drive Temperature",

        "smart_id":
            194,

        "smart_raw_feature":
            "smart_194_raw",

        "category":
            "thermal_context",

        "semantic_status":
            "validated_for_analysis",

        "causal_discovery_initially_eligible":
            True,
    },

    {
        "factor_id":
            "PENDING_SECTORS",

        "factor_name":
            "Current Pending Sectors",

        "smart_id":
            197,

        "smart_raw_feature":
            "smart_197_raw",

        "category":
            "degradation_indicator",

        "semantic_status":
            "validated_for_analysis",

        "causal_discovery_initially_eligible":
            True,
    },

    {
        "factor_id":
            "CRC_ERRORS",

        "factor_name":
            "CRC / Interface Errors",

        "smart_id":
            199,

        "smart_raw_feature":
            "smart_199_raw",

        "category":
            "interface_error",

        "semantic_status":
            "validated_for_analysis",

        "causal_discovery_initially_eligible":
            True,
    },

    {
        "factor_id":
            "HEAD_FLYING_HOURS",

        "factor_name":
            "Head Flying Hours / SMART 240",

        "smart_id":
            240,

        "smart_raw_feature":
            "smart_240_raw",

        "category":
            "usage_context",

        "semantic_status":
            "review",

        "causal_discovery_initially_eligible":
            False,
    },

    {
        "factor_id":
            "WRITE_WORKLOAD",

        "factor_name":
            "Write Workload",

        "smart_id":
            241,

        "smart_raw_feature":
            "smart_241_raw",

        "category":
            "workload_context",

        "semantic_status":
            "validated_for_analysis",

        "causal_discovery_initially_eligible":
            True,
    },

    {
        "factor_id":
            "READ_WORKLOAD",

        "factor_name":
            "Read Workload",

        "smart_id":
            242,

        "smart_raw_feature":
            "smart_242_raw",

        "category":
            "workload_context",

        "semantic_status":
            "validated_for_analysis",

        "causal_discovery_initially_eligible":
            True,
    },
]


manifest_df = pd.DataFrame(
    CANONICAL_FACTORS
)


# ==========================================================
# LOOKUP TABLES
# ==========================================================

RAW_TO_FACTOR = {

    row[
        "smart_raw_feature"
    ]:
        row[
            "factor_id"
        ]

    for row in CANONICAL_FACTORS
}


FACTOR_NAME_LOOKUP = {

    row[
        "factor_id"
    ]:
        row[
            "factor_name"
        ]

    for row in CANONICAL_FACTORS
}


CATEGORY_LOOKUP = {

    row[
        "factor_id"
    ]:
        row[
            "category"
        ]

    for row in CANONICAL_FACTORS
}


SEMANTIC_STATUS_LOOKUP = {

    row[
        "factor_id"
    ]:
        row[
            "semantic_status"
        ]

    for row in CANONICAL_FACTORS
}


# ==========================================================
# PCMCI VARIABLE MAPPING
# ==========================================================

PCMCI_TO_FACTOR = {

    "reallocated_increase_1d":
        "REALLOCATED_SECTORS",

    "reported_uncorrectable_increase_1d":
        "REPORTED_UNCORRECTABLE",

    "pending_sector_change_1d":
        "PENDING_SECTORS",

    "crc_error_increase_1d":
        "CRC_ERRORS",

    "temperature":
        "TEMPERATURE",

    "temperature_change_1d":
        "TEMPERATURE",

    "start_stop_activity_1d":
        "START_STOP_ACTIVITY",

    "power_cycle_activity_1d":
        "POWER_CYCLE_ACTIVITY",

    "poweroff_retract_activity_1d":
        "POWEROFF_RETRACT_ACTIVITY",

    "load_cycle_activity_1d":
        "LOAD_CYCLE_ACTIVITY",

    "write_workload_1d":
        "WRITE_WORKLOAD",

    "read_workload_1d":
        "READ_WORKLOAD",
}


# ==========================================================
# ATT TREATMENT MAPPING
# ==========================================================

ATT_TO_FACTOR = {

    "treat_reallocated":
        "REALLOCATED_SECTORS",

    "treat_uncorrectable":
        "REPORTED_UNCORRECTABLE",

    "treat_pending":
        "PENDING_SECTORS",
}


# ==========================================================
# SHAP FEATURE PARSER
#
# Examples:
#
# smart_187_raw
#
# smart_187_raw_diff_1d
#
# smart_187_raw_mean_7d
#
# smart_187_raw_std_30d
#
# smart_187_raw_max_14d
# ==========================================================

def parse_shap_feature(
    feature
):

    feature = str(
        feature
    )


    # ------------------------------------------------------
    # Find base SMART variable
    # ------------------------------------------------------

    base_feature = None


    for raw_feature in RAW_TO_FACTOR:

        if (
            feature
            ==
            raw_feature
        ):

            base_feature = (
                raw_feature
            )

            break


        if feature.startswith(
            raw_feature
            + "_"
        ):

            base_feature = (
                raw_feature
            )

            break


    if base_feature is None:

        return {
            "mapped":
                False,

            "original_feature":
                feature,

            "base_feature":
                None,

            "factor_id":
                None,

            "factor_name":
                None,

            "category":
                None,

            "semantic_status":
                None,

            "representation_type":
                None,

            "aggregation":
                None,

            "window_days":
                None,

            "temporal_scope":
                None,
        }


    factor_id = (
        RAW_TO_FACTOR[
            base_feature
        ]
    )


    suffix = feature[
        len(
            base_feature
        ):
    ]


    # Remove leading underscore.

    if suffix.startswith(
        "_"
    ):

        suffix = suffix[
            1:
        ]


    # ------------------------------------------------------
    # Raw state
    # ------------------------------------------------------

    if suffix == "":

        representation_type = (
            "raw_state"
        )

        aggregation = (
            "raw"
        )

        window_days = 0

        temporal_scope = (
            "current_day"
        )


    # ------------------------------------------------------
    # True one-day change
    # ------------------------------------------------------

    elif suffix == "diff_1d":

        representation_type = (
            "daily_change"
        )

        aggregation = (
            "difference"
        )

        window_days = 1

        temporal_scope = (
            "one_day_change"
        )


    # ------------------------------------------------------
    # Rolling temporal representation
    # ------------------------------------------------------

    else:

        match = re.fullmatch(

            r"(mean|std|max)_(7|14|30)d",

            suffix
        )


        if match:

            aggregation = (
                match.group(
                    1
                )
            )


            window_days = int(
                match.group(
                    2
                )
            )


            if aggregation == "mean":

                representation_type = (
                    "rolling_mean"
                )

            elif aggregation == "std":

                representation_type = (
                    "rolling_variability"
                )

            elif aggregation == "max":

                representation_type = (
                    "rolling_maximum"
                )

            else:

                representation_type = (
                    "rolling_summary"
                )


            temporal_scope = (
                f"{window_days}_day_window"
            )


        else:

            representation_type = (
                "unknown_engineered"
            )

            aggregation = (
                suffix
            )

            window_days = None

            temporal_scope = (
                "unknown"
            )


    return {
        "mapped":
            True,

        "original_feature":
            feature,

        "base_feature":
            base_feature,

        "factor_id":
            factor_id,

        "factor_name":
            FACTOR_NAME_LOOKUP[
                factor_id
            ],

        "category":
            CATEGORY_LOOKUP[
                factor_id
            ],

        "semantic_status":
            SEMANTIC_STATUS_LOOKUP[
                factor_id
            ],

        "representation_type":
            representation_type,

        "aggregation":
            aggregation,

        "window_days":
            window_days,

        "temporal_scope":
            temporal_scope,
    }


# ==========================================================
# CHECK REQUIRED FILES
# ==========================================================

required_files = [

    POPULATION_SHAP_FILE,

    FAILURE_SHAP_FILE,

    PCMCI_STABLE_EXACT_FILE,

    PCMCI_STABLE_PAIR_FILE,

    ATT_SUMMARY_FILE,

    ATT_ROBUSTNESS_FILE,
]


missing_required_files = [

    str(
        file
    )

    for file in required_files

    if not file.exists()
]


if missing_required_files:

    raise FileNotFoundError(
        "Missing required evidence files:\n"
        +
        "\n".join(
            missing_required_files
        )
    )


# ==========================================================
# LOAD GLOBAL SHAP FEATURE LISTS
# ==========================================================

population_shap = pd.read_csv(
    POPULATION_SHAP_FILE
)


failure_shap = pd.read_csv(
    FAILURE_SHAP_FILE
)


if "feature" not in population_shap.columns:

    raise RuntimeError(
        "Population SHAP file has no feature column."
    )


if "feature" not in failure_shap.columns:

    raise RuntimeError(
        "Failure-focused SHAP file has no feature column."
    )


# ==========================================================
# COLLECT ALL SHAP FEATURE NAMES
# ==========================================================

shap_feature_sets = [

    set(
        population_shap[
            "feature"
        ].astype(str)
    ),

    set(
        failure_shap[
            "feature"
        ].astype(str)
    ),
]


# ----------------------------------------------------------
# Add local SHAP feature lists when available.
# ----------------------------------------------------------

local_shap_files = [

    LOCAL_SHAP_DIR
    / "TP_local_shap.csv",

    LOCAL_SHAP_DIR
    / "FP_local_shap.csv",

    LOCAL_SHAP_DIR
    / "FN_local_shap.csv",

    LOCAL_SHAP_DIR
    / "TN_local_shap.csv",
]


for local_file in local_shap_files:

    if not local_file.exists():

        continue


    local_df = pd.read_csv(
        local_file
    )


    if "feature" in local_df.columns:

        shap_feature_sets.append(
            set(
                local_df[
                    "feature"
                ].astype(str)
            )
        )


all_shap_features = sorted(
    set().union(
        *shap_feature_sets
    )
)


# ==========================================================
# PARSE SHAP FEATURES
# ==========================================================

shap_mapping_rows = [

    parse_shap_feature(
        feature
    )

    for feature in all_shap_features
]


shap_mapping_df = pd.DataFrame(
    shap_mapping_rows
)


# ==========================================================
# UNKNOWN SHAP REPRESENTATION CHECK
# ==========================================================

unknown_engineered = shap_mapping_df[
    (
        shap_mapping_df[
            "mapped"
        ] == True
    )
    &
    (
        shap_mapping_df[
            "representation_type"
        ]
        ==
        "unknown_engineered"
    )
].copy()


# ==========================================================
# LOAD PCMCI VARIABLES
# ==========================================================

stable_exact = pd.read_csv(
    PCMCI_STABLE_EXACT_FILE
)


stable_pairs = pd.read_csv(
    PCMCI_STABLE_PAIR_FILE
)


pcmci_variables = sorted(
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


pcmci_rows = []


for variable in pcmci_variables:

    factor_id = (
        PCMCI_TO_FACTOR.get(
            variable
        )
    )


    pcmci_rows.append(
        {
            "pcmci_variable":
                variable,

            "mapped":
                factor_id
                is not None,

            "factor_id":
                factor_id,

            "factor_name":
                (
                    FACTOR_NAME_LOOKUP.get(
                        factor_id
                    )
                    if factor_id
                    else None
                ),

            "category":
                (
                    CATEGORY_LOOKUP.get(
                        factor_id
                    )
                    if factor_id
                    else None
                ),

            "evidence_type":
                "temporal_dependency",
        }
    )


pcmci_mapping_df = pd.DataFrame(
    pcmci_rows
)


# ==========================================================
# LOAD ATT TREATMENTS
# ==========================================================

att_summary = pd.read_csv(
    ATT_SUMMARY_FILE
)


att_robustness = pd.read_csv(
    ATT_ROBUSTNESS_FILE
)


att_treatments = sorted(
    set(
        att_summary[
            "treatment"
        ].astype(str)
    )
    |
    set(
        att_robustness[
            "treatment"
        ].astype(str)
    )
)


att_rows = []


for treatment in att_treatments:

    factor_id = (
        ATT_TO_FACTOR.get(
            treatment
        )
    )


    att_rows.append(
        {
            "treatment":
                treatment,

            "mapped":
                factor_id
                is not None,

            "factor_id":
                factor_id,

            "factor_name":
                (
                    FACTOR_NAME_LOOKUP.get(
                        factor_id
                    )
                    if factor_id
                    else None
                ),

            "category":
                (
                    CATEGORY_LOOKUP.get(
                        factor_id
                    )
                    if factor_id
                    else None
                ),

            "evidence_type":
                "matched_failure_effect",
        }
    )


att_mapping_df = pd.DataFrame(
    att_rows
)


# ==========================================================
# MAPPING COVERAGE
# ==========================================================

coverage_rows = []


# ----------------------------------------------------------
# SHAP coverage
# ----------------------------------------------------------

coverage_rows.append(
    {
        "evidence_source":
            "SHAP",

        "total_inputs":
            len(
                shap_mapping_df
            ),

        "mapped_inputs":
            int(
                shap_mapping_df[
                    "mapped"
                ].sum()
            ),

        "unmapped_inputs":
            int(
                (
                    ~shap_mapping_df[
                        "mapped"
                    ]
                ).sum()
            ),

        "coverage_pct":
            (
                shap_mapping_df[
                    "mapped"
                ].mean()
                * 100
            ),
    }
)


# ----------------------------------------------------------
# PCMCI coverage
# ----------------------------------------------------------

coverage_rows.append(
    {
        "evidence_source":
            "PCMCI",

        "total_inputs":
            len(
                pcmci_mapping_df
            ),

        "mapped_inputs":
            int(
                pcmci_mapping_df[
                    "mapped"
                ].sum()
            ),

        "unmapped_inputs":
            int(
                (
                    ~pcmci_mapping_df[
                        "mapped"
                    ]
                ).sum()
            ),

        "coverage_pct":
            (
                pcmci_mapping_df[
                    "mapped"
                ].mean()
                * 100
            ),
    }
)


# ----------------------------------------------------------
# ATT coverage
# ----------------------------------------------------------

coverage_rows.append(
    {
        "evidence_source":
            "MATCHED_ATT",

        "total_inputs":
            len(
                att_mapping_df
            ),

        "mapped_inputs":
            int(
                att_mapping_df[
                    "mapped"
                ].sum()
            ),

        "unmapped_inputs":
            int(
                (
                    ~att_mapping_df[
                        "mapped"
                    ]
                ).sum()
            ),

        "coverage_pct":
            (
                att_mapping_df[
                    "mapped"
                ].mean()
                * 100
            ),
    }
)


coverage_df = pd.DataFrame(
    coverage_rows
)


# ==========================================================
# UNMAPPED INPUT AUDIT
# ==========================================================

unmapped_rows = []


for _, row in shap_mapping_df[
    ~shap_mapping_df[
        "mapped"
    ]
].iterrows():

    unmapped_rows.append(
        {
            "evidence_source":
                "SHAP",

            "input_name":
                row[
                    "original_feature"
                ],

            "reason":
                "No SMART-to-factor mapping",
        }
    )


for _, row in pcmci_mapping_df[
    ~pcmci_mapping_df[
        "mapped"
    ]
].iterrows():

    unmapped_rows.append(
        {
            "evidence_source":
                "PCMCI",

            "input_name":
                row[
                    "pcmci_variable"
                ],

            "reason":
                "No PCMCI-to-factor mapping",
        }
    )


for _, row in att_mapping_df[
    ~att_mapping_df[
        "mapped"
    ]
].iterrows():

    unmapped_rows.append(
        {
            "evidence_source":
                "MATCHED_ATT",

            "input_name":
                row[
                    "treatment"
                ],

            "reason":
                "No treatment-to-factor mapping",
        }
    )


unmapped_df = pd.DataFrame(
    unmapped_rows,
    columns=[
        "evidence_source",
        "input_name",
        "reason",
    ]
)


# ==========================================================
# FACTOR EVIDENCE AVAILABILITY
# ==========================================================

factor_availability_rows = []


for _, factor in manifest_df.iterrows():

    factor_id = (
        factor[
            "factor_id"
        ]
    )


    shap_available = bool(
        (
            shap_mapping_df[
                "factor_id"
            ]
            ==
            factor_id
        ).any()
    )


    pcmci_available = bool(
        (
            pcmci_mapping_df[
                "factor_id"
            ]
            ==
            factor_id
        ).any()
    )


    att_available = bool(
        (
            att_mapping_df[
                "factor_id"
            ]
            ==
            factor_id
        ).any()
    )


    factor_availability_rows.append(
        {
            "factor_id":
                factor_id,

            "factor_name":
                factor[
                    "factor_name"
                ],

            "category":
                factor[
                    "category"
                ],

            "semantic_status":
                factor[
                    "semantic_status"
                ],

            "shap_available":
                shap_available,

            "pcmci_available":
                pcmci_available,

            "matched_att_available":
                att_available,

            "evidence_stream_count":
                int(
                    shap_available
                )
                +
                int(
                    pcmci_available
                )
                +
                int(
                    att_available
                ),
        }
    )


factor_availability_df = pd.DataFrame(
    factor_availability_rows
)


factor_availability_df = (
    factor_availability_df
    .sort_values(
        [
            "evidence_stream_count",
            "factor_id",
        ],
        ascending=[
            False,
            True,
        ]
    )
)


# ==========================================================
# SHAP REPRESENTATION SUMMARY
# ==========================================================

shap_representation_summary = (

    shap_mapping_df[
        shap_mapping_df[
            "mapped"
        ]
    ]
    .groupby(
        [
            "factor_id",
            "factor_name",
            "representation_type",
            "window_days",
        ],
        dropna=False,
        as_index=False
    )
    .agg(
        feature_count=(
            "original_feature",
            "count"
        )
    )
)


# ==========================================================
# SAVE OUTPUTS
# ==========================================================

manifest_file = (
    OUTPUT_DIR
    / "canonical_factor_manifest.csv"
)


shap_mapping_file = (
    OUTPUT_DIR
    / "shap_feature_to_factor_mapping.csv"
)


pcmci_mapping_file = (
    OUTPUT_DIR
    / "pcmci_variable_to_factor_mapping.csv"
)


att_mapping_file = (
    OUTPUT_DIR
    / "att_treatment_to_factor_mapping.csv"
)


coverage_file = (
    OUTPUT_DIR
    / "evidence_mapping_coverage.csv"
)


unmapped_file = (
    OUTPUT_DIR
    / "unmapped_evidence_inputs.csv"
)


availability_file = (
    OUTPUT_DIR
    / "factor_evidence_availability.csv"
)


representation_file = (
    OUTPUT_DIR
    / "shap_representation_summary.csv"
)


unknown_engineered_file = (
    OUTPUT_DIR
    / "unknown_shap_engineered_features.csv"
)


manifest_df.to_csv(
    manifest_file,
    index=False
)


shap_mapping_df.to_csv(
    shap_mapping_file,
    index=False
)


pcmci_mapping_df.to_csv(
    pcmci_mapping_file,
    index=False
)


att_mapping_df.to_csv(
    att_mapping_file,
    index=False
)


coverage_df.to_csv(
    coverage_file,
    index=False
)


unmapped_df.to_csv(
    unmapped_file,
    index=False
)


factor_availability_df.to_csv(
    availability_file,
    index=False
)


shap_representation_summary.to_csv(
    representation_file,
    index=False
)


unknown_engineered.to_csv(
    unknown_engineered_file,
    index=False
)


# ==========================================================
# HARD VALIDATION
# ==========================================================

unmapped_total = (

    len(
        unmapped_df
    )
)


unknown_engineered_total = len(
    unknown_engineered
)


all_sources_mapped = (
    unmapped_total
    ==
    0
)


all_shap_representations_known = (
    unknown_engineered_total
    ==
    0
)


# ==========================================================
# PRINT RESULTS
# ==========================================================

print("\n")
print("=" * 78)

print(
    "STEP 13B - CANONICAL FACTOR MAPPING COMPLETE"
)

print("=" * 78)


print(
    "\nCanonical factors:",
    len(
        manifest_df
    )
)


print(
    "\nMapping coverage:\n"
)


print(
    coverage_df.to_string(
        index=False
    )
)


print(
    "\nUnknown SHAP engineered representations:",
    unknown_engineered_total
)


print(
    "\nTotal unmapped evidence inputs:",
    unmapped_total
)


if unmapped_total > 0:

    print(
        "\nUNMAPPED INPUTS:\n"
    )


    print(
        unmapped_df.to_string(
            index=False
        )
    )


print(
    "\nFactor evidence availability:\n"
)


print(
    factor_availability_df[
        [
            "factor_id",
            "factor_name",
            "semantic_status",
            "shap_available",
            "pcmci_available",
            "matched_att_available",
            "evidence_stream_count",
        ]
    ].to_string(
        index=False
    )
)


print(
    "\nSHAP representations detected:\n"
)


representation_counts = (

    shap_mapping_df[
        "representation_type"
    ]
    .value_counts(
        dropna=False
    )
    .rename_axis(
        "representation_type"
    )
    .reset_index(
        name="feature_count"
    )
)


print(
    representation_counts.to_string(
        index=False
    )
)


print(
    "\nAll evidence inputs mapped:",
    all_sources_mapped
)


print(
    "All SHAP representations recognized:",
    all_shap_representations_known
)


print(
    "\nOutputs saved to:"
)


print(
    OUTPUT_DIR
)