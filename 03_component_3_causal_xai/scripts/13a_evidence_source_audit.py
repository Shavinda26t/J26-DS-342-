from pathlib import Path

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

FIGURES_ROOT = (
    PROJECT_ROOT
    / "figures"
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
# FILE TYPES WE CARE ABOUT
# ==========================================================

VALID_EXTENSIONS = {

    ".csv",
    ".json",
    ".parquet",
    ".npz",
    ".joblib",
    ".png",
    ".jpg",
    ".jpeg",
}


# ==========================================================
# EVIDENCE KEYWORDS
# ==========================================================

KEYWORD_GROUPS = {

    "reference_model": [
        "reference",
        "xgboost",
        "model",
    ],

    "global_shap": [
        "global_shap",
        "shap_global",
        "global",
        "shap",
    ],

    "local_shap": [
        "local_shap",
        "shap_local",
        "local",
        "shap",
    ],

    "pcmci": [
        "pcmci",
        "causal_discovery",
        "stability",
    ],

    "matched_att": [
        "matched_att",
        "att",
        "causal_effect",
    ],

    "matching": [
        "matching",
        "matched_risksets",
        "riskset",
    ],
}


# ==========================================================
# RECURSIVE FILE SCAN
# ==========================================================

SEARCH_ROOTS = [

    RESULTS_ROOT,

    DATA_ROOT,

    FIGURES_ROOT,
]


file_rows = []


for search_root in SEARCH_ROOTS:

    if not search_root.exists():

        continue


    for path in search_root.rglob("*"):

        if not path.is_file():

            continue


        if path.suffix.lower() not in VALID_EXTENSIONS:

            continue


        relative_path = path.relative_to(
            PROJECT_ROOT
        )


        lower_text = str(
            relative_path
        ).lower()


        matched_groups = []


        for group_name, keywords in (
            KEYWORD_GROUPS.items()
        ):

            matches = sum(

                keyword in lower_text

                for keyword in keywords
            )


            # Require at least one meaningful keyword.
            if matches > 0:

                matched_groups.append(
                    group_name
                )


        file_rows.append(
            {
                "file_name":
                    path.name,

                "extension":
                    path.suffix.lower(),

                "relative_path":
                    str(
                        relative_path
                    ),

                "absolute_path":
                    str(
                        path
                    ),

                "size_bytes":
                    path.stat().st_size,

                "size_mb":
                    path.stat().st_size
                    / (
                        1024
                        * 1024
                    ),

                "evidence_groups":
                    ";".join(
                        matched_groups
                    ),

                "matched_group_count":
                    len(
                        matched_groups
                    ),
            }
        )


files_df = pd.DataFrame(
    file_rows
)


if files_df.empty:

    raise RuntimeError(
        "No evidence files were found."
    )


files_df = (
    files_df
    .sort_values(
        [
            "matched_group_count",
            "relative_path",
        ],
        ascending=[
            False,
            True,
        ]
    )
    .reset_index(
        drop=True
    )
)


# ==========================================================
# SAVE COMPLETE INVENTORY
# ==========================================================

inventory_file = (
    OUTPUT_DIR
    / "component3_file_inventory.csv"
)


files_df.to_csv(
    inventory_file,
    index=False
)


# ==========================================================
# IMPORTANT EXPECTED FILES
# ==========================================================

EXPECTED_PATHS = {

    "reference_model_30d":
        PROJECT_ROOT
        / "models"
        / "reference_model"
        / "30d"
        / "xgboost_reference.joblib",

    "pcmci_stable_exact_links":
        RESULTS_ROOT
        / "causal_discovery"
        / "pcmci_stability"
        / "stable_exact_links.csv",

    "pcmci_stable_pairs":
        RESULTS_ROOT
        / "causal_discovery"
        / "pcmci_stability"
        / "stable_pair_links.csv",

    "pcmci_exact_stability":
        RESULTS_ROOT
        / "causal_discovery"
        / "pcmci_stability"
        / "exact_link_stability.csv",

    "pcmci_pair_stability":
        RESULTS_ROOT
        / "causal_discovery"
        / "pcmci_stability"
        / "pair_level_stability.csv",

    "matched_att_summary":
        RESULTS_ROOT
        / "causal_effect"
        / "matched_att"
        / "matched_att_effect_summary.csv",

    "matched_att_roles":
        RESULTS_ROOT
        / "causal_effect"
        / "matched_att"
        / "causal_effect_analysis_roles.csv",

    "matched_att_robustness":
        RESULTS_ROOT
        / "causal_effect"
        / "matched_att_robustness"
        / "clustered_att_robustness_summary.csv",

    "history_corrected_matching":
        DATA_ROOT
        / "interim"
        / "riskset_matching_refined"
        / "history_corrected_matching_summary.csv",
}


expected_rows = []


for evidence_name, path in (
    EXPECTED_PATHS.items()
):

    expected_rows.append(
        {
            "evidence_name":
                evidence_name,

            "exists":
                path.exists(),

            "absolute_path":
                str(
                    path
                ),

            "size_bytes":
                (
                    path.stat().st_size
                    if path.exists()
                    else None
                ),
        }
    )


expected_df = pd.DataFrame(
    expected_rows
)


expected_file = (
    OUTPUT_DIR
    / "expected_evidence_files.csv"
)


expected_df.to_csv(
    expected_file,
    index=False
)


# ==========================================================
# SHAP-SPECIFIC DISCOVERY
#
# We intentionally discover these rather than assuming
# old filenames.
# ==========================================================

shap_candidates = files_df[
    files_df[
        "relative_path"
    ]
    .str
    .lower()
    .str
    .contains(
        "shap",
        na=False
    )
].copy()


shap_candidate_file = (
    OUTPUT_DIR
    / "shap_file_candidates.csv"
)


shap_candidates.to_csv(
    shap_candidate_file,
    index=False
)


# ==========================================================
# PCMCI-SPECIFIC DISCOVERY
# ==========================================================

pcmci_candidates = files_df[
    files_df[
        "relative_path"
    ]
    .str
    .lower()
    .str
    .contains(
        "pcmci",
        na=False
    )
].copy()


pcmci_candidate_file = (
    OUTPUT_DIR
    / "pcmci_file_candidates.csv"
)


pcmci_candidates.to_csv(
    pcmci_candidate_file,
    index=False
)


# ==========================================================
# CAUSAL-EFFECT-SPECIFIC DISCOVERY
# ==========================================================

causal_effect_candidates = files_df[
    (
        files_df[
            "relative_path"
        ]
        .str
        .lower()
        .str
        .contains(
            "matched_att",
            na=False
        )
    )
    |
    (
        files_df[
            "relative_path"
        ]
        .str
        .lower()
        .str
        .contains(
            "riskset_matching",
            na=False
        )
    )
].copy()


causal_effect_candidate_file = (
    OUTPUT_DIR
    / "causal_effect_file_candidates.csv"
)


causal_effect_candidates.to_csv(
    causal_effect_candidate_file,
    index=False
)


# ==========================================================
# HIGH-PRIORITY CSV PREVIEW
#
# Show columns and row counts for CSVs likely to become
# integrated evidence inputs.
# ==========================================================

priority_csv_keywords = [

    "shap",

    "stable_exact_links",

    "stable_pair_links",

    "exact_link_stability",

    "pair_level_stability",

    "matched_att_effect_summary",

    "clustered_att_robustness_summary",
]


preview_rows = []


for _, row in files_df.iterrows():

    path = Path(
        row[
            "absolute_path"
        ]
    )


    if path.suffix.lower() != ".csv":

        continue


    lower_path = str(
        path
    ).lower()


    if not any(
        keyword in lower_path
        for keyword in priority_csv_keywords
    ):

        continue


    try:

        temp = pd.read_csv(
            path
        )


        preview_rows.append(
            {
                "file_name":
                    path.name,

                "relative_path":
                    row[
                        "relative_path"
                    ],

                "rows":
                    len(
                        temp
                    ),

                "columns":
                    ";".join(
                        temp.columns.astype(
                            str
                        )
                    ),
            }
        )


    except Exception as error:

        preview_rows.append(
            {
                "file_name":
                    path.name,

                "relative_path":
                    row[
                        "relative_path"
                    ],

                "rows":
                    None,

                "columns":
                    f"ERROR: {error}",
            }
        )


preview_df = pd.DataFrame(
    preview_rows
)


preview_file = (
    OUTPUT_DIR
    / "priority_evidence_csv_structure.csv"
)


preview_df.to_csv(
    preview_file,
    index=False
)


# ==========================================================
# PRINT SUMMARY
# ==========================================================

print("\n")
print("=" * 76)

print(
    "STEP 13A - EVIDENCE SOURCE AUDIT COMPLETE"
)

print("=" * 76)


print(
    "\nTotal candidate files found:",
    f"{len(files_df):,}"
)


print(
    "\nExpected evidence files:\n"
)


print(
    expected_df.to_string(
        index=False
    )
)


print(
    "\nSHAP candidate files:",
    len(
        shap_candidates
    )
)


if len(
    shap_candidates
) > 0:

    print(
        "\nSHAP files:\n"
    )


    print(
        shap_candidates[
            [
                "file_name",
                "relative_path",
                "size_mb",
            ]
        ]
        .to_string(
            index=False
        )
    )


print(
    "\nPriority evidence CSV structures:\n"
)


if len(
    preview_df
) > 0:

    print(
        preview_df.to_string(
            index=False
        )
    )

else:

    print(
        "No priority CSV files detected."
    )


print(
    "\nAudit outputs saved to:"
)


print(
    OUTPUT_DIR
)