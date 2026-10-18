from pathlib import Path

import numpy as np
import pandas as pd


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PAIR_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "matched_att"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "causal_effect"
    / "matched_att_robustness"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# SETTINGS
# ==========================================================

TREATMENTS = [

    "treat_reallocated",

    "treat_uncorrectable",

    "treat_pending",
]


BOOTSTRAP_ITERATIONS = 20000

RANDOM_STATE = 42


# ==========================================================
# STORAGE
# ==========================================================

summary_rows = []


# ==========================================================
# PROCESS EACH TREATMENT
# ==========================================================

for treatment in TREATMENTS:

    print("\n")
    print("=" * 74)

    print(
        f"CONTROL-HDD CLUSTERED BOOTSTRAP: {treatment}"
    )

    print("=" * 74)


    pair_file = (
        PAIR_DIR
        / f"{treatment}_matched_pairs.csv"
    )


    if not pair_file.exists():

        raise FileNotFoundError(
            f"Missing pair file: {pair_file}"
        )


    pairs = pd.read_csv(
        pair_file
    )


    # ======================================================
    # VALIDATE
    # ==========================================================

    required = [

        "treated_serial_number",

        "control_serial_number",

        "treated_outcome",

        "control_outcome",

        "pair_difference",
    ]


    missing = [

        col

        for col in required

        if col not in pairs.columns
    ]


    if missing:

        raise RuntimeError(
            f"Missing columns: {missing}"
        )


    pairs[
        "treated_outcome"
    ] = pd.to_numeric(
        pairs[
            "treated_outcome"
        ],
        errors="raise"
    )


    pairs[
        "control_outcome"
    ] = pd.to_numeric(
        pairs[
            "control_outcome"
        ],
        errors="raise"
    )


    pairs[
        "pair_difference"
    ] = (
        pairs[
            "treated_outcome"
        ]
        -
        pairs[
            "control_outcome"
        ]
    )


    # ======================================================
    # POINT ESTIMATE
    # ==========================================================

    point_att = float(
        pairs[
            "pair_difference"
        ].mean()
    )


    treated_risk = float(
        pairs[
            "treated_outcome"
        ].mean()
    )


    control_risk = float(
        pairs[
            "control_outcome"
        ].mean()
    )


    # ======================================================
    # CONTROL CLUSTERS
    #
    # Every unique matched control HDD is a cluster.
    #
    # If a control HDD was reused for more than one treated
    # HDD, all corresponding pairs stay together.
    # ==========================================================

    control_ids = (

        pairs[
            "control_serial_number"
        ]
        .drop_duplicates()
        .to_numpy()
    )


    n_clusters = len(
        control_ids
    )


    cluster_to_pairs = {

        control_id:

            pairs[
                pairs[
                    "control_serial_number"
                ]
                == control_id
            ].copy()

        for control_id in control_ids
    }


    print(
        "Matched pairs:",
        len(
            pairs
        )
    )


    print(
        "Unique control-HDD clusters:",
        n_clusters
    )


    print(
        "Point ATT:",
        f"{point_att * 100:.3f}",
        "percentage points"
    )


    # ======================================================
    # CLUSTER BOOTSTRAP
    #
    # Sample control HDD clusters with replacement.
    #
    # When a cluster is selected, include every matched pair
    # associated with that control HDD.
    # ==========================================================

    rng = np.random.default_rng(
        RANDOM_STATE
    )


    bootstrap_att = np.empty(
        BOOTSTRAP_ITERATIONS,
        dtype=np.float64
    )


    for b in range(
        BOOTSTRAP_ITERATIONS
    ):

        sampled_control_ids = rng.choice(

            control_ids,

            size=n_clusters,

            replace=True,
        )


        sampled_differences = []


        for control_id in (
            sampled_control_ids
        ):

            cluster = (
                cluster_to_pairs[
                    control_id
                ]
            )


            sampled_differences.extend(

                cluster[
                    "pair_difference"
                ]
                .to_numpy(
                    dtype=np.float64
                )
                .tolist()
            )


        bootstrap_att[
            b
        ] = np.mean(
            sampled_differences
        )


    # ======================================================
    # CONFIDENCE INTERVAL
    # ==========================================================

    ci_lower = float(
        np.percentile(
            bootstrap_att,
            2.5
        )
    )


    ci_upper = float(
        np.percentile(
            bootstrap_att,
            97.5
        )
    )


    positive_fraction = float(
        (
            bootstrap_att > 0
        ).mean()
    )


    # ======================================================
    # CONTROL REUSE
    # ==========================================================

    reuse_counts = (

        pairs[
            "control_serial_number"
        ]
        .value_counts()
    )


    reused_controls = int(
        (
            reuse_counts > 1
        ).sum()
    )


    max_reuse = int(
        reuse_counts.max()
    )


    # ======================================================
    # SAVE DISTRIBUTION
    # ==========================================================

    np.savez_compressed(

        OUTPUT_DIR
        / f"{treatment}_cluster_bootstrap.npz",

        bootstrap_att=
            bootstrap_att,
    )


    # ======================================================
    # SUMMARY
    # ==========================================================

    summary_rows.append(
        {
            "treatment":
                treatment,

            "matched_pairs":
                len(
                    pairs
                ),

            "unique_control_clusters":
                n_clusters,

            "reused_control_hdds":
                reused_controls,

            "maximum_control_reuse":
                max_reuse,

            "treated_risk":
                treated_risk,

            "control_risk":
                control_risk,

            "att_risk_difference":
                point_att,

            "att_percentage_points":
                point_att
                * 100,

            "cluster_bootstrap_ci_lower":
                ci_lower,

            "cluster_bootstrap_ci_upper":
                ci_upper,

            "cluster_bootstrap_ci_lower_pp":
                ci_lower
                * 100,

            "cluster_bootstrap_ci_upper_pp":
                ci_upper
                * 100,

            "bootstrap_positive_fraction":
                positive_fraction,
        }
    )


    print(
        "Clustered-bootstrap 95% CI:",
        f"[{ci_lower * 100:.3f}, "
        f"{ci_upper * 100:.3f}]",
        "percentage points"
    )


    print(
        "Bootstrap estimates > 0:",
        f"{positive_fraction * 100:.2f}%"
    )


    print(
        "Controls reused:",
        reused_controls
    )


    print(
        "Maximum reuse:",
        max_reuse
    )


# ==========================================================
# SAVE SUMMARY
# ==========================================================

summary_df = pd.DataFrame(
    summary_rows
)


summary_file = (
    OUTPUT_DIR
    / "clustered_att_robustness_summary.csv"
)


summary_df.to_csv(
    summary_file,
    index=False
)


# ==========================================================
# FINAL PRINT
# ==========================================================

print("\n")
print("=" * 74)

print(
    "CLUSTERED ATT ROBUSTNESS COMPLETE"
)

print("=" * 74)


print(
    "\nSummary:\n"
)


print(
    summary_df[
        [
            "treatment",

            "matched_pairs",

            "unique_control_clusters",

            "att_percentage_points",

            "cluster_bootstrap_ci_lower_pp",

            "cluster_bootstrap_ci_upper_pp",

            "bootstrap_positive_fraction",
        ]
    ].to_string(
        index=False
    )
)


print(
    "\nSaved to:"
)


print(
    OUTPUT_DIR
)