from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

import tigramite
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.robust_parcorr import RobustParCorr


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PANEL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pcmci_pilot"
    / "pcmci_pilot_panel.parquet"
)

VARIABLE_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "pcmci_pilot"
    / "pcmci_variable_manifest.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "causal_discovery"
    / "pcmci_pilot"
)

AUDIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "pcmci_pilot_results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

AUDIT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# SETTINGS
# ==========================================================

TAU_MIN = 1
TAU_MAX = 7

PC_ALPHA = 0.05
FINAL_ALPHA = 0.05

FDR_METHOD = "fdr_bh"

# Keep pilot computationally controlled.
MAX_CONDS_DIM = 5
MAX_CONDS_PY = 5
MAX_CONDS_PX = 5

MAX_COMBINATIONS = 1


# ==========================================================
# LOAD VARIABLE MANIFEST
# ==========================================================

manifest = pd.read_csv(
    VARIABLE_MANIFEST
)

manifest = manifest.sort_values(
    "variable_index"
)

FEATURES = (
    manifest[
        "variable"
    ]
    .tolist()
)


print(
    "PCMCI variables:",
    len(FEATURES)
)

for index, feature in enumerate(
    FEATURES
):

    print(
        f"  {index:02d}: {feature}"
    )


# ==========================================================
# LOAD PANEL
# ==========================================================

print(
    "\nLoading PCMCI pilot panel..."
)

panel = pd.read_parquet(
    PANEL_FILE
)

panel[
    "date"
] = pd.to_datetime(
    panel[
        "date"
    ],
    errors="coerce"
)


panel = panel.sort_values(
    [
        "pcmci_dataset_id",
        "date"
    ]
).reset_index(
    drop=True
)


print(
    "Rows:",
    f"{len(panel):,}"
)

print(
    "HDDs:",
    f"{panel['pcmci_dataset_id'].nunique():,}"
)


# ==========================================================
# VARIABLE VARIABILITY AUDIT
# ==========================================================

variability_rows = []


for feature in FEATURES:

    values = pd.to_numeric(
        panel[
            feature
        ],
        errors="coerce"
    ).astype(
        "float64"
    )


    finite_values = values[
        np.isfinite(
            values
        )
    ]


    unique_values = int(
        finite_values.nunique()
    )


    zero_pct = (
        float(
            (
                finite_values == 0
            ).mean()
            * 100
        )
        if len(
            finite_values
        ) > 0
        else np.nan
    )


    variability_rows.append(
        {
            "feature":
                feature,

            "rows":
                len(values),

            "finite_rows":
                len(finite_values),

            "missing_or_infinite_rows":
                int(
                    len(values)
                    - len(finite_values)
                ),

            "unique_values":
                unique_values,

            "zero_pct":
                zero_pct,

            "mean":
                (
                    float(
                        finite_values.mean()
                    )
                    if len(
                        finite_values
                    ) > 0
                    else np.nan
                ),

            "std":
                (
                    float(
                        finite_values.std()
                    )
                    if len(
                        finite_values
                    ) > 1
                    else np.nan
                ),

            "min":
                (
                    float(
                        finite_values.min()
                    )
                    if len(
                        finite_values
                    ) > 0
                    else np.nan
                ),

            "max":
                (
                    float(
                        finite_values.max()
                    )
                    if len(
                        finite_values
                    ) > 0
                    else np.nan
                ),
        }
    )


variability_df = pd.DataFrame(
    variability_rows
)


variability_file = (
    AUDIT_DIR
    / "pcmci_variable_variability.csv"
)


variability_df.to_csv(
    variability_file,
    index=False
)


print(
    "\nVariable variability:\n"
)

print(
    variability_df[
        [
            "feature",
            "unique_values",
            "zero_pct",
            "std"
        ]
    ].to_string(
        index=False
    )
)


# ==========================================================
# CHECK CONSTANT VARIABLES
# ==========================================================

constant_features = (
    variability_df.loc[
        variability_df[
            "unique_values"
        ] <= 1,
        "feature"
    ]
    .tolist()
)


if constant_features:

    raise RuntimeError(
        "Cannot run PCMCI because these variables "
        f"are constant: {constant_features}"
    )


# ==========================================================
# BUILD MULTIPLE-TIME-SERIES DICTIONARY
# ==========================================================

print(
    "\nBuilding Tigramite multiple-dataset structure..."
)


data_dict = {}

dataset_map_rows = []


for dataset_index, (
    serial,
    drive
) in enumerate(
    panel.groupby(
        "pcmci_dataset_id",
        sort=True
    )
):

    drive = drive.sort_values(
        "date"
    )


    values = (
        drive[
            FEATURES
        ]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
        .to_numpy(
            dtype=np.float64
        )
    )


    if not np.isfinite(
        values
    ).all():

        raise RuntimeError(
            "NaN or infinite value found in "
            f"HDD {serial}"
        )


    # Verify daily continuity again.
    date_differences = (
        drive[
            "date"
        ]
        .diff()
        .dt.days
        .dropna()
    )


    if not (
        date_differences == 1
    ).all():

        raise RuntimeError(
            "Non-daily transition found in "
            f"HDD {serial}"
        )


    # Integer keys make the Tigramite structure simple.
    data_dict[
        dataset_index
    ] = values


    dataset_map_rows.append(
        {
            "dataset_index":
                dataset_index,

            "serial_number":
                serial,

            "start_date":
                drive[
                    "date"
                ].min(),

            "end_date":
                drive[
                    "date"
                ].max(),

            "rows":
                len(
                    drive
                ),
        }
    )


dataset_map_df = pd.DataFrame(
    dataset_map_rows
)


dataset_map_file = (
    AUDIT_DIR
    / "pcmci_dataset_map.csv"
)


dataset_map_df.to_csv(
    dataset_map_file,
    index=False
)


print(
    "Separate time series:",
    len(data_dict)
)


# ==========================================================
# CREATE TIGRAMITE DATAFRAME
# ==========================================================

dataframe = pp.DataFrame(

    data=data_dict,

    analysis_mode="multiple",

    var_names=FEATURES,
)


# ==========================================================
# CONDITIONAL INDEPENDENCE TEST
# ==========================================================

cond_ind_test = RobustParCorr(
    significance="analytic"
)


# ==========================================================
# CREATE PCMCI
# ==========================================================

pcmci = PCMCI(

    dataframe=dataframe,

    cond_ind_test=cond_ind_test,

    verbosity=0,
)


# ==========================================================
# RUN PCMCI
# ==========================================================

print("\n")
print("=" * 70)
print("RUNNING PCMCI PILOT")
print("=" * 70)


print(
    "tau_min:",
    TAU_MIN
)

print(
    "tau_max:",
    TAU_MAX
)

print(
    "PC screening alpha:",
    PC_ALPHA
)

print(
    "Final alpha:",
    FINAL_ALPHA
)

print(
    "FDR correction:",
    FDR_METHOD
)

print(
    "Maximum conditioning dimension:",
    MAX_CONDS_DIM
)


start_time = time.time()


results = pcmci.run_pcmci(

    tau_min=TAU_MIN,

    tau_max=TAU_MAX,

    pc_alpha=PC_ALPHA,

    max_conds_dim=MAX_CONDS_DIM,

    max_combinations=MAX_COMBINATIONS,

    max_conds_py=MAX_CONDS_PY,

    max_conds_px=MAX_CONDS_PX,

    alpha_level=FINAL_ALPHA,

    fdr_method=FDR_METHOD,
)


runtime_seconds = (
    time.time()
    - start_time
)


print(
    "\nPCMCI completed in:",
    round(
        runtime_seconds,
        2
    ),
    "seconds"
)


# ==========================================================
# RESULT MATRICES
# ==========================================================

graph = results[
    "graph"
]

p_matrix = results[
    "p_matrix"
]

val_matrix = results[
    "val_matrix"
]


# ==========================================================
# SAVE RAW MATRICES
# ==========================================================

matrix_file = (
    OUTPUT_DIR
    / "pcmci_result_matrices.npz"
)


np.savez_compressed(

    matrix_file,

    graph=graph,

    p_matrix=p_matrix,

    val_matrix=val_matrix,
)


# ==========================================================
# CREATE ALL LAG-TEST TABLE
# ==========================================================

test_rows = []


for source_index, source in enumerate(
    FEATURES
):

    for target_index, target in enumerate(
        FEATURES
    ):

        for lag in range(
            TAU_MIN,
            TAU_MAX + 1
        ):

            p_value = float(
                p_matrix[
                    source_index,
                    target_index,
                    lag
                ]
            )


            test_value = float(
                val_matrix[
                    source_index,
                    target_index,
                    lag
                ]
            )


            graph_mark = str(
                graph[
                    source_index,
                    target_index,
                    lag
                ]
            )


            significant = (
                np.isfinite(
                    p_value
                )
                and
                p_value
                <= FINAL_ALPHA
                and
                graph_mark != ""
            )


            test_rows.append(
                {
                    "source":
                        source,

                    "target":
                        target,

                    "lag_days":
                        lag,

                    "graph_mark":
                        graph_mark,

                    # Because fdr_method='fdr_bh',
                    # this is the FDR-adjusted p-value.
                    "fdr_p_value":
                        p_value,

                    # RobustParCorr conditional-dependence
                    # statistic. Do NOT call this a
                    # causal effect size.
                    "mci_value":
                        test_value,

                    "abs_mci_value":
                        abs(
                            test_value
                        ),

                    "significant":
                        significant,

                    "self_link":
                        (
                            source
                            == target
                        ),
                }
            )


all_tests_df = pd.DataFrame(
    test_rows
)


all_tests_file = (
    OUTPUT_DIR
    / "pcmci_all_lag_tests.csv"
)


all_tests_df.to_csv(
    all_tests_file,
    index=False
)


# ==========================================================
# SIGNIFICANT LINKS
# ==========================================================

significant_df = (
    all_tests_df[
        all_tests_df[
            "significant"
        ]
    ]
    .copy()
)


significant_df = (
    significant_df
    .sort_values(
        [
            "fdr_p_value",
            "abs_mci_value"
        ],
        ascending=[
            True,
            False
        ]
    )
    .reset_index(
        drop=True
    )
)


significant_file = (
    OUTPUT_DIR
    / "pcmci_significant_links.csv"
)


significant_df.to_csv(
    significant_file,
    index=False
)


# ==========================================================
# CROSS-VARIABLE LINKS
#
# Keep self-lag links in the full result, but create
# a separate table because the research question is
# especially interested in relationships between
# different SMART processes.
# ==========================================================

cross_variable_df = (
    significant_df[
        significant_df[
            "self_link"
        ] == False
    ]
    .copy()
)


cross_variable_file = (
    OUTPUT_DIR
    / "pcmci_cross_variable_links.csv"
)


cross_variable_df.to_csv(
    cross_variable_file,
    index=False
)


# ==========================================================
# VARIABLE-LEVEL LINK SUMMARY
# ==========================================================

summary_rows = []


for feature in FEATURES:

    incoming = (
        cross_variable_df[
            cross_variable_df[
                "target"
            ] == feature
        ]
    )


    outgoing = (
        cross_variable_df[
            cross_variable_df[
                "source"
            ] == feature
        ]
    )


    summary_rows.append(
        {
            "feature":
                feature,

            "significant_incoming_links":
                len(
                    incoming
                ),

            "significant_outgoing_links":
                len(
                    outgoing
                ),

            "total_cross_variable_links":
                (
                    len(
                        incoming
                    )
                    +
                    len(
                        outgoing
                    )
                ),
        }
    )


link_summary_df = pd.DataFrame(
    summary_rows
)


link_summary_df = (
    link_summary_df
    .sort_values(
        "total_cross_variable_links",
        ascending=False
    )
)


link_summary_file = (
    OUTPUT_DIR
    / "pcmci_variable_link_summary.csv"
)


link_summary_df.to_csv(
    link_summary_file,
    index=False
)


# ==========================================================
# RUN METADATA
# ==========================================================

metadata = {

    "method":
        "PCMCI",

    "conditional_independence_test":
        "RobustParCorr",

    "analysis_mode":
        "multiple",

    "pilot_hdds":
        int(
            len(
                data_dict
            )
        ),

    "pilot_rows":
        int(
            len(
                panel
            )
        ),

    "variables":
        FEATURES,

    "number_of_variables":
        len(
            FEATURES
        ),

    "tau_min":
        TAU_MIN,

    "tau_max":
        TAU_MAX,

    "pc_alpha":
        PC_ALPHA,

    "alpha_level":
        FINAL_ALPHA,

    "fdr_method":
        FDR_METHOD,

    "max_conds_dim":
        MAX_CONDS_DIM,

    "max_conds_py":
        MAX_CONDS_PY,

    "max_conds_px":
        MAX_CONDS_PX,

    "max_combinations":
        MAX_COMBINATIONS,

    "runtime_seconds":
        runtime_seconds,

    "significant_links":
        int(
            len(
                significant_df
            )
        ),

    "significant_self_links":
        int(
            significant_df[
                "self_link"
            ].sum()
        )
        if len(
            significant_df
        ) > 0
        else 0,

    "significant_cross_variable_links":
        int(
            len(
                cross_variable_df
            )
        ),

    "tigramite_version":
        getattr(
            tigramite,
            "__version__",
            "unknown"
        ),
}


metadata_file = (
    OUTPUT_DIR
    / "pcmci_run_metadata.json"
)


with open(
    metadata_file,
    "w"
) as f:

    json.dump(
        metadata,
        f,
        indent=4
    )


# ==========================================================
# PRINT RESULTS
# ==========================================================

print("\n")
print("=" * 70)
print("PCMCI PILOT CAUSAL DISCOVERY COMPLETE")
print("=" * 70)


print(
    "Significant lagged links:",
    len(
        significant_df
    )
)


print(
    "Significant self-lag links:",
    int(
        significant_df[
            "self_link"
        ].sum()
    )
    if len(
        significant_df
    ) > 0
    else 0
)


print(
    "Significant cross-variable links:",
    len(
        cross_variable_df
    )
)


print(
    "\nTop cross-variable temporal links:\n"
)


if len(
    cross_variable_df
) > 0:

    print(
        cross_variable_df[
            [
                "source",
                "target",
                "lag_days",
                "mci_value",
                "fdr_p_value"
            ]
        ]
        .head(
            30
        )
        .to_string(
            index=False
        )
    )

else:

    print(
        "No cross-variable links survived FDR correction."
    )


print(
    "\nVariable link summary:\n"
)

print(
    link_summary_df.to_string(
        index=False
    )
)


print(
    "\nResults saved to:"
)

print(
    OUTPUT_DIR
)