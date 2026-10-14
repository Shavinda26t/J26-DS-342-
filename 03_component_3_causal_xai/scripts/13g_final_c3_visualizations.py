from pathlib import Path
from datetime import datetime
import os
import textwrap
import uuid

import numpy as np
import pandas as pd

import matplotlib

# Use non-interactive backend.
# This is safer for scripts and prevents GUI backend problems.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch


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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "figures"
    / "integrated_evidence"
    / "final"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# INPUT FILES
# ==========================================================

MATRIX_FILE = (
    ROOT_CAUSE_DIR
    / "integrated_root_cause_evidence_matrix.csv"
)

CANDIDATE_FILE = (
    ROOT_CAUSE_DIR
    / "root_cause_candidate_summary.csv"
)

PATHWAY_FILE = (
    ROOT_CAUSE_DIR
    / "degradation_temporal_pathways.csv"
)

CLASS_SUMMARY_FILE = (
    ROOT_CAUSE_DIR
    / "integrated_evidence_class_summary.csv"
)

STREAM_SUMMARY_FILE = (
    ROOT_CAUSE_DIR
    / "integrated_evidence_stream_summary.csv"
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

    MATRIX_FILE,

    CANDIDATE_FILE,

    PATHWAY_FILE,

    CLASS_SUMMARY_FILE,

    STREAM_SUMMARY_FILE,

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
        "Missing required visualization inputs:\n"
        +
        "\n".join(
            missing_files
        )
    )


# ==========================================================
# LOAD DATA
# ==========================================================

matrix = pd.read_csv(
    MATRIX_FILE
)

candidates = pd.read_csv(
    CANDIDATE_FILE
)

pathways = pd.read_csv(
    PATHWAY_FILE
)

class_summary = pd.read_csv(
    CLASS_SUMMARY_FILE
)

stream_summary = pd.read_csv(
    STREAM_SUMMARY_FILE
)

pcmci_pairs = pd.read_csv(
    PCMCI_PAIR_FILE
)

att = pd.read_csv(
    ATT_FILE
)


# ==========================================================
# REQUIRED COLUMN VALIDATION
# ==========================================================

required_matrix_columns = {

    "factor_id",

    "factor_name",

    "consensus_rank",

    "mean_normalized_shap_share_pct",

    "pcmci_available",

    "matched_att_available",

    "root_cause_candidate",

    "integrated_evidence_class",
}


required_pcmci_columns = {

    "source_factor_id",

    "target_factor_id",

    "source_factor_name",

    "target_factor_name",

    "temporal_pair_rank",

    "temporal_magnitude",
}


required_att_columns = {

    "factor_id",

    "factor_name",

    "failure_risk_difference_pp",

    "failure_risk_ci_lower_pp",

    "failure_risk_ci_upper_pp",

    "mcnemar_exact_p_value",

    "treated_events",
}


required_class_columns = {

    "integrated_evidence_class",

    "factor_count",
}


required_pathway_columns = {

    "source_factor_id",

    "target_factor_id",

    "source_factor_name",

    "target_factor_name",
}


for label, frame, required_columns in [

    (
        "integrated evidence matrix",
        matrix,
        required_matrix_columns
    ),

    (
        "PCMCI factor-pair evidence",
        pcmci_pairs,
        required_pcmci_columns
    ),

    (
        "matched ATT evidence",
        att,
        required_att_columns
    ),

    (
        "integrated class summary",
        class_summary,
        required_class_columns
    ),

    (
        "degradation pathways",
        pathways,
        required_pathway_columns
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
# BOOLEAN HELPER
#
# Handles bool columns even if CSV parsing returns strings.
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
# GENERAL FIGURE SETTINGS
# ==========================================================

plt.rcParams.update(
    {
        "font.size":
            10,

        "axes.titlesize":
            14,

        "axes.labelsize":
            11,

        "figure.titlesize":
            15,

        "legend.fontsize":
            9,

        "savefig.facecolor":
            "white",

        "figure.facecolor":
            "white",

        "axes.facecolor":
            "white",
    }
)


# ==========================================================
# PNG VALIDATION
# ==========================================================

PNG_SIGNATURE = (
    b"\x89PNG\r\n\x1a\n"
)


def validate_png(
    path
):

    path = Path(
        path
    )


    if not path.exists():

        raise RuntimeError(
            f"PNG file was not created: {path}"
        )


    size_bytes = path.stat().st_size


    if size_bytes < 1000:

        raise RuntimeError(
            f"PNG file appears too small/corrupt: "
            f"{path} ({size_bytes} bytes)"
        )


    with open(
        path,
        "rb"
    ) as file:

        signature = file.read(
            8
        )


    if signature != PNG_SIGNATURE:

        raise RuntimeError(
            f"Invalid PNG signature detected: {path}"
        )


    return size_bytes


# ==========================================================
# FALLBACK OUTPUT NAME
# ==========================================================

def create_fallback_path(
    original_path
):

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )


    candidate = (
        original_path.parent
        /
        (
            f"{original_path.stem}"
            f"_new_{timestamp}"
            f"{original_path.suffix}"
        )
    )


    counter = 1


    while candidate.exists():

        candidate = (
            original_path.parent
            /
            (
                f"{original_path.stem}"
                f"_new_{timestamp}_{counter}"
                f"{original_path.suffix}"
            )
        )


        counter += 1


    return candidate


# ==========================================================
# SAFE FIGURE SAVE
#
# Workflow:
#
# 1. Render to temporary PNG.
# 2. Validate PNG header and size.
# 3. Attempt atomic replacement of requested filename.
# 4. If Windows locks old file, save using timestamped name.
# 5. Validate final output again.
# ==========================================================

def save_figure(
    fig,
    filename
):

    final_path = (
        OUTPUT_DIR
        / filename
    )


    temp_path = (
        OUTPUT_DIR
        /
        (
            f".{final_path.stem}_"
            f"{uuid.uuid4().hex[:8]}"
            f".tmp.png"
        )
    )


    saved_path = None


    try:

        # --------------------------------------------------
        # First render to a fresh temporary PNG.
        # --------------------------------------------------

        fig.savefig(

            temp_path,

            format="png",

            dpi=300,

            bbox_inches="tight",

            facecolor="white"
        )


        temp_size = validate_png(
            temp_path
        )


        # --------------------------------------------------
        # Attempt final replacement.
        # --------------------------------------------------

        try:

            os.replace(
                temp_path,
                final_path
            )


            saved_path = (
                final_path
            )


        except OSError as error:

            # ------------------------------------------------
            # Existing PNG is likely locked by Windows Photos,
            # Explorer preview pane, etc.
            # ------------------------------------------------

            fallback_path = create_fallback_path(
                final_path
            )


            print(
                f"\nWARNING: Could not overwrite "
                f"{final_path.name}"
            )


            print(
                f"Windows returned: {error}"
            )


            print(
                "Saving valid replacement as:"
            )


            print(
                fallback_path.name
            )


            os.replace(
                temp_path,
                fallback_path
            )


            saved_path = (
                fallback_path
            )


        # --------------------------------------------------
        # Validate final PNG.
        # --------------------------------------------------

        final_size = validate_png(
            saved_path
        )


        print(
            "Saved:",
            saved_path
        )


        print(
            "  PNG validation: PASS"
        )


        print(
            f"  File size: "
            f"{final_size:,} bytes"
        )


        return saved_path


    finally:

        plt.close(
            fig
        )


        # --------------------------------------------------
        # Remove unused temp file after exceptions.
        # --------------------------------------------------

        if temp_path.exists():

            try:

                temp_path.unlink()

            except OSError:

                pass


# ==========================================================
# FIGURE 1
# GLOBAL INTEGRATED FACTOR EVIDENCE
# ==========================================================

plot_df = (

    matrix

    .sort_values(
        "consensus_rank",
        ascending=False
    )

    .copy()
)


fig, ax = plt.subplots(
    figsize=(
        12,
        9
    )
)


y = np.arange(
    len(
        plot_df
    )
)


ax.barh(

    y,

    plot_df[
        "mean_normalized_shap_share_pct"
    ],

    color="#2C7FB8"
)


ax.set_yticks(
    y
)


ax.set_yticklabels(

    plot_df[
        "factor_name"
    ]
)


ax.set_xlabel(
    "Mean normalized global SHAP share (%)"
)


ax.set_title(
    "Integrated HDD Factor Evidence: "
    "Predictive Importance with Causal-Support Markers"
)


max_share = float(
    plot_df[
        "mean_normalized_shap_share_pct"
    ].max()
)


# ----------------------------------------------------------
# Add evidence markers
# ----------------------------------------------------------

for index, (_, row) in enumerate(
    plot_df.iterrows()
):

    x = float(
        row[
            "mean_normalized_shap_share_pct"
        ]
    )


    if as_bool(
        row[
            "pcmci_available"
        ]
    ):

        ax.scatter(

            x + 0.18,

            index,

            marker="o",

            s=55,

            color="#F28E2B",

            edgecolor="black",

            linewidth=0.3,

            zorder=5
        )


    if as_bool(
        row[
            "matched_att_available"
        ]
    ):

        ax.scatter(

            x + 0.45,

            index,

            marker="*",

            s=120,

            color="#D62728",

            edgecolor="black",

            linewidth=0.3,

            zorder=6
        )


    if as_bool(
        row[
            "root_cause_candidate"
        ]
    ):

        ax.text(

            x + 0.72,

            index,

            "RC",

            va="center",

            fontsize=8,

            fontweight="bold"
        )


legend_elements = [

    Line2D(
        [0],
        [0],

        marker="o",

        linestyle="None",

        markerfacecolor="#F28E2B",

        markeredgecolor="black",

        label="Stable PCMCI temporal evidence",
    ),

    Line2D(
        [0],
        [0],

        marker="*",

        linestyle="None",

        markerfacecolor="#D62728",

        markeredgecolor="black",

        markersize=11,

        label="Matched failure-risk evidence",
    ),

    Line2D(
        [0],
        [0],

        linestyle="None",

        marker="None",

        label="RC = root-cause candidate",
    ),
]


ax.legend(
    handles=legend_elements,
    loc="lower right"
)


ax.set_xlim(
    0,
    max_share + 2.2
)


ax.grid(
    axis="x",
    alpha=0.22
)


fig.tight_layout()


FIGURE_1_PATH = save_figure(
    fig,
    "01_integrated_factor_evidence.png"
)


# ==========================================================
# FIGURE 2
# MATCHED ATT FOREST PLOT
# ==========================================================

att_plot = (

    att

    .sort_values(
        "failure_risk_difference_pp",
        ascending=True
    )

    .reset_index(
        drop=True
    )
)


fig, ax = plt.subplots(
    figsize=(
        10.5,
        5.5
    )
)


y = np.arange(
    len(
        att_plot
    )
)


effect = att_plot[
    "failure_risk_difference_pp"
].to_numpy(
    dtype=float
)


lower = att_plot[
    "failure_risk_ci_lower_pp"
].to_numpy(
    dtype=float
)


upper = att_plot[
    "failure_risk_ci_upper_pp"
].to_numpy(
    dtype=float
)


left_error = (
    effect
    -
    lower
)


right_error = (
    upper
    -
    effect
)


ax.errorbar(

    effect,

    y,

    xerr=np.vstack(
        [
            left_error,
            right_error,
        ]
    ),

    fmt="o",

    color="#2C7FB8",

    ecolor="#2C7FB8",

    capsize=6,

    linewidth=2,

    markersize=7
)


ax.axvline(

    0,

    linestyle="--",

    linewidth=1.3,

    color="#555555"
)


ax.set_yticks(
    y
)


ax.set_yticklabels(

    att_plot[
        "factor_name"
    ]
)


ax.set_xlabel(
    "Matched 30-day failure-risk difference "
    "(percentage points)"
)


ax.set_title(
    "Matched Incident Failure-Risk Effects"
)


# ----------------------------------------------------------
# Annotation labels
# ----------------------------------------------------------

for i, row in att_plot.iterrows():

    label = (

        f"{row['failure_risk_difference_pp']:.2f} pp "
        f"[{row['failure_risk_ci_lower_pp']:.2f}, "
        f"{row['failure_risk_ci_upper_pp']:.2f}]"
    )


    ax.text(

        float(
            row[
                "failure_risk_ci_upper_pp"
            ]
        )
        + 0.45,

        i,

        label,

        va="center",

        fontsize=9
    )


max_upper = float(
    att_plot[
        "failure_risk_ci_upper_pp"
    ].max()
)


ax.set_xlim(
    min(
        -1.0,
        float(
            lower.min()
        ) - 0.5
    ),
    max_upper + 5
)


ax.grid(
    axis="x",
    alpha=0.22
)


fig.tight_layout()


FIGURE_2_PATH = save_figure(
    fig,
    "02_matched_att_forest_plot.png"
)


# ==========================================================
# FIGURE 3
# PCMCI STABLE TEMPORAL NETWORK
#
# Strongest 15 relationships plus all relationships
# involving degradation indicators.
# ==========================================================

top_pcmci = (

    pcmci_pairs

    .sort_values(
        "temporal_pair_rank"
    )

    .head(
        15
    )

    .copy()
)


DEGRADATION_IDS = {

    "REALLOCATED_SECTORS",

    "REPORTED_UNCORRECTABLE",

    "PENDING_SECTORS",
}


degradation_links = pcmci_pairs[
    (
        pcmci_pairs[
            "source_factor_id"
        ].isin(
            DEGRADATION_IDS
        )
    )
    |
    (
        pcmci_pairs[
            "target_factor_id"
        ].isin(
            DEGRADATION_IDS
        )
    )
].copy()


network_df = (

    pd.concat(
        [
            top_pcmci,

            degradation_links,
        ],
        ignore_index=True
    )

    .drop_duplicates(
        subset=[
            "source_factor_id",

            "target_factor_id",
        ]
    )

    .sort_values(
        "temporal_pair_rank"
    )

    .reset_index(
        drop=True
    )
)


nodes = sorted(
    set(
        network_df[
            "source_factor_name"
        ]
    )
    |
    set(
        network_df[
            "target_factor_name"
        ]
    )
)


node_count = len(
    nodes
)


angles = np.linspace(
    0,
    2 * np.pi,
    node_count,
    endpoint=False
)


positions = {}


for node, angle in zip(
    nodes,
    angles
):

    positions[
        node
    ] = (

        1.15
        *
        np.cos(
            angle
        ),

        1.15
        *
        np.sin(
            angle
        ),
    )


fig, ax = plt.subplots(
    figsize=(
        12,
        12
    )
)


ax.set_aspect(
    "equal"
)


ax.axis(
    "off"
)


# ----------------------------------------------------------
# Determine degradation factor display names
# ----------------------------------------------------------

degradation_names = set(
    matrix.loc[
        matrix[
            "factor_id"
        ].isin(
            DEGRADATION_IDS
        ),
        "factor_name"
    ]
)


# ----------------------------------------------------------
# Draw nodes
# ----------------------------------------------------------

for node in nodes:

    x, y_pos = positions[
        node
    ]


    if node in degradation_names:

        node_color = (
            "#F6C85F"
        )

    else:

        node_color = (
            "#BFD7EA"
        )


    ax.scatter(

        x,

        y_pos,

        s=1800,

        color=node_color,

        edgecolor="#333333",

        linewidth=1,

        zorder=5
    )


    wrapped_name = "\n".join(
        textwrap.wrap(
            node,
            width=18
        )
    )


    ax.text(

        x,

        y_pos,

        wrapped_name,

        ha="center",

        va="center",

        fontsize=8,

        zorder=6
    )


# ----------------------------------------------------------
# Draw directed links
# ----------------------------------------------------------

max_magnitude = float(
    network_df[
        "temporal_magnitude"
    ].max()
)


for _, row in network_df.iterrows():

    source = row[
        "source_factor_name"
    ]


    target = row[
        "target_factor_name"
    ]


    source_pos = positions[
        source
    ]


    target_pos = positions[
        target
    ]


    magnitude = float(
        row[
            "temporal_magnitude"
        ]
    )


    if max_magnitude > 0:

        line_width = (

            1.0

            +

            4.0
            *
            magnitude
            /
            max_magnitude
        )

    else:

        line_width = 1.0


    if (
        row[
            "source_factor_id"
        ]
        in DEGRADATION_IDS
        or
        row[
            "target_factor_id"
        ]
        in DEGRADATION_IDS
    ):

        edge_color = (
            "#B04A3F"
        )

        edge_alpha = (
            0.85
        )

    else:

        edge_color = (
            "#666666"
        )

        edge_alpha = (
            0.55
        )


    arrow = FancyArrowPatch(

        source_pos,

        target_pos,

        arrowstyle="-|>",

        mutation_scale=13,

        linewidth=line_width,

        color=edge_color,

        alpha=edge_alpha,

        connectionstyle="arc3,rad=0.08",

        shrinkA=32,

        shrinkB=32,

        zorder=2
    )


    ax.add_patch(
        arrow
    )


ax.set_title(
    "Stable PCMCI Temporal-Dependency Network\n"
    "Top Temporal Relationships from Repeated Stability Analysis"
)


fig.tight_layout()


FIGURE_3_PATH = save_figure(
    fig,
    "03_pcmci_stable_temporal_network.png"
)


# ==========================================================
# FIGURE 4
# DEGRADATION / ROOT-CAUSE PATHWAY
# ==========================================================

fig, ax = plt.subplots(
    figsize=(
        13,
        6
    )
)


ax.set_xlim(
    0,
    1
)


ax.set_ylim(
    0,
    1
)


ax.axis(
    "off"
)


# ----------------------------------------------------------
# Main node positions
# ----------------------------------------------------------

reallocated_pos = (
    0.14,
    0.58
)

uncorrectable_pos = (
    0.50,
    0.58
)

failure_pos = (
    0.86,
    0.58
)


# ----------------------------------------------------------
# Draw node helper
# ----------------------------------------------------------

def draw_node(
    ax,
    position,
    text,
    facecolor
):

    ax.text(

        position[
            0
        ],

        position[
            1
        ],

        text,

        ha="center",

        va="center",

        fontsize=11,

        fontweight="bold",

        bbox={
            "boxstyle":
                "round,pad=0.65",

            "facecolor":
                facecolor,

            "edgecolor":
                "#333333",

            "linewidth":
                1.2,
        },

        zorder=5
    )


draw_node(

    ax,

    reallocated_pos,

    "Reallocated\nSectors",

    "#F6C85F"
)


draw_node(

    ax,

    uncorrectable_pos,

    "Reported\nUncorrectable Errors",

    "#F6C85F"
)


draw_node(

    ax,

    failure_pos,

    "30-Day HDD\nFailure Risk",

    "#D8EAF8"
)


# ----------------------------------------------------------
# Find actual degradation temporal pathway
# ----------------------------------------------------------

realloc_uncorrect = pathways[
    (
        pathways[
            "source_factor_id"
        ]
        ==
        "REALLOCATED_SECTORS"
    )
    &
    (
        pathways[
            "target_factor_id"
        ]
        ==
        "REPORTED_UNCORRECTABLE"
    )
]


if len(
    realloc_uncorrect
) > 0:

    pathway_row = (
        realloc_uncorrect.iloc[
            0
        ]
    )


    if (
        "strongest_lag_days"
        in pathway_row.index
        and
        pd.notna(
            pathway_row[
                "strongest_lag_days"
            ]
        )
    ):

        lag_text = (

            f"strongest exact lag = "
            f"{int(pathway_row['strongest_lag_days'])} days"
        )


    elif (
        "modal_lag_days"
        in pathway_row.index
        and
        pd.notna(
            pathway_row[
                "modal_lag_days"
            ]
        )
    ):

        lag_text = (

            f"modal lag = "
            f"{int(pathway_row['modal_lag_days'])} days"
        )


    else:

        lag_text = (
            "lag varies across stable links"
        )


    if (
        "temporal_direction"
        in pathway_row.index
        and
        pd.notna(
            pathway_row[
                "temporal_direction"
            ]
        )
    ):

        direction_text = str(
            pathway_row[
                "temporal_direction"
            ]
        ).replace(
            "_",
            " "
        ).lower()


    else:

        direction_text = (
            "stable temporal dependency"
        )


else:

    lag_text = (
        "stable temporal relationship"
    )


    direction_text = (
        "conditional-dependency evidence"
    )


# ----------------------------------------------------------
# Arrow 1:
# Reallocated -> Reported Uncorrectable
# ----------------------------------------------------------

arrow_1 = FancyArrowPatch(

    (
        0.22,
        0.58
    ),

    (
        0.42,
        0.58
    ),

    arrowstyle="-|>",

    mutation_scale=20,

    linewidth=2.6,

    color="#B04A3F",

    connectionstyle="arc3,rad=0.0",

    zorder=3
)


ax.add_patch(
    arrow_1
)


ax.text(

    0.32,

    0.70,

    (
        "Stable PCMCI temporal dependency\n"
        f"{lag_text}\n"
        f"{direction_text}"
    ),

    ha="center",

    va="center",

    fontsize=9
)


# ----------------------------------------------------------
# Matched ATT rows
# ----------------------------------------------------------

uncorrectable_row = att[
    att[
        "factor_id"
    ]
    ==
    "REPORTED_UNCORRECTABLE"
]


reallocated_row = att[
    att[
        "factor_id"
    ]
    ==
    "REALLOCATED_SECTORS"
]


pending_row = att[
    att[
        "factor_id"
    ]
    ==
    "PENDING_SECTORS"
]


# ----------------------------------------------------------
# Arrow 2:
# Reported Uncorrectable -> Failure risk evidence
# ----------------------------------------------------------

if len(
    uncorrectable_row
) == 1:

    row = uncorrectable_row.iloc[
        0
    ]


    arrow_2 = FancyArrowPatch(

        (
            0.60,
            0.58
        ),

        (
            0.78,
            0.58
        ),

        arrowstyle="-|>",

        mutation_scale=20,

        linewidth=3.0,

        color="#2C7FB8",

        connectionstyle="arc3,rad=0.0",

        zorder=3
    )


    ax.add_patch(
        arrow_2
    )


    ax.text(

        0.69,

        0.70,

        (
            "Matched incident risk contrast\n"
            f"+{row['failure_risk_difference_pp']:.2f} pp\n"
            f"95% CI "
            f"[{row['failure_risk_ci_lower_pp']:.2f}, "
            f"{row['failure_risk_ci_upper_pp']:.2f}]"
        ),

        ha="center",

        va="center",

        fontsize=9
    )


# ----------------------------------------------------------
# Reallocated direct evidence note
# ----------------------------------------------------------

if len(
    reallocated_row
) == 1:

    row = reallocated_row.iloc[
        0
    ]


    ax.text(

        0.14,

        0.28,

        (
            "Direct matched evidence\n"
            f"+{row['failure_risk_difference_pp']:.2f} pp\n"
            f"95% CI "
            f"[{row['failure_risk_ci_lower_pp']:.2f}, "
            f"{row['failure_risk_ci_upper_pp']:.2f}]\n"
            f"McNemar p = "
            f"{row['mcnemar_exact_p_value']:.3f}\n"
            "Supportive but statistically uncertain"
        ),

        ha="center",

        va="center",

        fontsize=9,

        bbox={
            "boxstyle":
                "round,pad=0.45",

            "facecolor":
                "#FFF6D8",

            "edgecolor":
                "#888888",
        }
    )


# ----------------------------------------------------------
# Pending sectors exploratory evidence
# ----------------------------------------------------------

if len(
    pending_row
) == 1:

    row = pending_row.iloc[
        0
    ]


    ax.text(

        0.50,

        0.24,

        (
            "Current Pending Sectors\n"
            f"+{row['failure_risk_difference_pp']:.2f} pp "
            "matched difference\n"
            f"95% CI "
            f"[{row['failure_risk_ci_lower_pp']:.2f}, "
            f"{row['failure_risk_ci_upper_pp']:.2f}]\n"
            f"{int(row['treated_events'])} treated failures\n"
            "Exploratory sparse evidence"
        ),

        ha="center",

        va="center",

        fontsize=9,

        bbox={
            "boxstyle":
                "round,pad=0.45",

            "facecolor":
                "#FDEDEC",

            "edgecolor":
                "#888888",
        }
    )


# ----------------------------------------------------------
# Main title
# ----------------------------------------------------------

ax.text(

    0.50,

    0.94,

    "Root-Cause-Oriented HDD Degradation Evidence Pathway",

    ha="center",

    va="center",

    fontsize=15,

    fontweight="bold"
)


# ----------------------------------------------------------
# Interpretation warning
# ----------------------------------------------------------

ax.text(

    0.50,

    0.055,

    (
        "Interpretation: observational temporal and matched "
        "risk evidence. The pathway supports root-cause-oriented "
        "reasoning but does not prove a physical causal mechanism."
    ),

    ha="center",

    va="center",

    fontsize=9,

    style="italic"
)


fig.tight_layout()


FIGURE_4_PATH = save_figure(
    fig,
    "04_degradation_root_cause_pathway.png"
)


# ==========================================================
# FIGURE 5
# INTEGRATED EVIDENCE CLASS SUMMARY
# ==========================================================

class_plot = (

    class_summary

    .sort_values(
        "factor_count",
        ascending=True
    )

    .copy()
)


class_plot[
    "display_name"
] = (

    class_plot[
        "integrated_evidence_class"
    ]

    .str.replace(
        "_",
        " ",
        regex=False
    )

    .str.title()
)


fig, ax = plt.subplots(
    figsize=(
        11,
        6.5
    )
)


ax.barh(

    class_plot[
        "display_name"
    ],

    class_plot[
        "factor_count"
    ],

    color="#2C7FB8"
)


ax.set_xlabel(
    "Number of canonical HDD factors"
)


ax.set_title(
    "Distribution of Integrated Causal-XAI Evidence Classes"
)


for index, row in class_plot.reset_index(
    drop=True
).iterrows():

    ax.text(

        float(
            row[
                "factor_count"
            ]
        )
        + 0.08,

        index,

        str(
            int(
                row[
                    "factor_count"
                ]
            )
        ),

        va="center"
    )


ax.grid(
    axis="x",
    alpha=0.22
)


fig.tight_layout()


FIGURE_5_PATH = save_figure(
    fig,
    "05_integrated_evidence_class_summary.png"
)


# ==========================================================
# FIGURE MANIFEST
#
# Uses the ACTUAL output filenames.
# This matters when a Windows-locked file is saved with
# a timestamped fallback filename.
# ==========================================================

figure_manifest = pd.DataFrame(
    [
        {
            "figure":
                FIGURE_1_PATH.name,

            "absolute_path":
                str(
                    FIGURE_1_PATH
                ),

            "purpose":
                (
                    "Shows factor-level global SHAP "
                    "importance with PCMCI, matched ATT "
                    "and root-cause-candidate markers."
                ),
        },

        {
            "figure":
                FIGURE_2_PATH.name,

            "absolute_path":
                str(
                    FIGURE_2_PATH
                ),

            "purpose":
                (
                    "Shows matched 30-day failure-risk "
                    "differences and clustered-bootstrap "
                    "95% confidence intervals."
                ),
        },

        {
            "figure":
                FIGURE_3_PATH.name,

            "absolute_path":
                str(
                    FIGURE_3_PATH
                ),

            "purpose":
                (
                    "Shows the strongest stable PCMCI "
                    "cross-factor temporal dependencies."
                ),
        },

        {
            "figure":
                FIGURE_4_PATH.name,

            "absolute_path":
                str(
                    FIGURE_4_PATH
                ),

            "purpose":
                (
                    "Combines degradation temporal "
                    "evidence with matched failure-risk "
                    "evidence for root-cause-oriented "
                    "reasoning."
                ),
        },

        {
            "figure":
                FIGURE_5_PATH.name,

            "absolute_path":
                str(
                    FIGURE_5_PATH
                ),

            "purpose":
                (
                    "Summarizes the number of factors "
                    "in each integrated evidence class."
                ),
        },
    ]
)


figure_manifest_file = (
    OUTPUT_DIR
    / "figure_manifest.csv"
)


figure_manifest.to_csv(
    figure_manifest_file,
    index=False
)


# ==========================================================
# FINAL OUTPUT VALIDATION
# ==========================================================

final_figure_paths = [

    FIGURE_1_PATH,

    FIGURE_2_PATH,

    FIGURE_3_PATH,

    FIGURE_4_PATH,

    FIGURE_5_PATH,
]


validation_rows = []


for path in final_figure_paths:

    size_bytes = validate_png(
        path
    )


    validation_rows.append(
        {
            "figure":
                path.name,

            "valid_png":
                True,

            "size_bytes":
                size_bytes,

            "size_kb":
                size_bytes
                /
                1024,
        }
    )


validation_df = pd.DataFrame(
    validation_rows
)


validation_file = (
    OUTPUT_DIR
    / "figure_validation.csv"
)


validation_df.to_csv(
    validation_file,
    index=False
)


# ==========================================================
# PRINT SUMMARY
# ==========================================================

print("\n")
print("=" * 82)

print(
    "STEP 13G - FINAL C3 VISUALIZATIONS COMPLETE"
)

print("=" * 82)


print(
    "\nFigures created and validated:\n"
)


print(
    validation_df[
        [
            "figure",

            "valid_png",

            "size_kb",
        ]
    ].to_string(
        index=False
    )
)


print(
    "\nFigure directory:"
)


print(
    OUTPUT_DIR
)


print(
    "\nFigure manifest:"
)


print(
    figure_manifest_file
)


print(
    "\nValidation file:"
)


print(
    validation_file
)