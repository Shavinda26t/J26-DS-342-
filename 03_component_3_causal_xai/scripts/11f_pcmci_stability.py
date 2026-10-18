from pathlib import Path
import time

import numpy as np
import pandas as pd

from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.robust_parcorr import RobustParCorr


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "causal_final"
    / "ST12000NM0008_Q3_Q4"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "causal_discovery"
    / "pcmci_stability"
)

AUDIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "pcmci_stability"
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

SEEDS = [
    42,
    101,
    202,
    303,
    404,
]

HDDS_PER_RUN = 500

MIN_CONTIGUOUS_DAYS = 60

TAU_MIN = 1
TAU_MAX = 7

PC_ALPHA = 0.05
FINAL_ALPHA = 0.05

MAX_CONDS_DIM = 5
MAX_CONDS_PY = 5
MAX_CONDS_PX = 5
MAX_COMBINATIONS = 1


# ==========================================================
# REFINED DISCOVERY VARIABLES
#
# Removed:
# - temperature_change_1d:
#   mathematically derived from temperature
#
# - crc_error_increase_1d:
#   essentially invariant in pilot sample
# ==========================================================

FEATURES = [

    "reallocated_increase_1d",

    "reported_uncorrectable_increase_1d",

    "pending_sector_change_1d",

    "temperature",

    "start_stop_activity_1d",

    "power_cycle_activity_1d",

    "poweroff_retract_activity_1d",

    "load_cycle_activity_1d",

    "write_workload_1d",

    "read_workload_1d",
]


# ==========================================================
# LOAD FULL CAUSAL DATASET
# ==========================================================

files = sorted(
    INPUT_DIR.glob("*.parquet")
)

if not files:

    raise FileNotFoundError(
        f"No files found in {INPUT_DIR}"
    )


parts = []

columns = (
    [
        "date",
        "serial_number",
    ]
    + FEATURES
)


for i, file in enumerate(
    files,
    start=1
):

    print(
        f"[{i}/{len(files)}] "
        f"Loading {file.name}"
    )

    temp = pd.read_parquet(
        file,
        columns=columns
    )

    parts.append(
        temp
    )


df = pd.concat(
    parts,
    ignore_index=True
)

del parts


df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)


for feature in FEATURES:

    df[feature] = pd.to_numeric(
        df[feature],
        errors="coerce"
    ).astype(
        "float64"
    )


# ==========================================================
# REMOVE MISSING ROWS BEFORE SEGMENT CONSTRUCTION
# ==========================================================

before = len(df)


df = df.dropna(
    subset=[
        "date",
        "serial_number",
    ]
    + FEATURES
).copy()


print(
    "\nRows loaded:",
    f"{before:,}"
)

print(
    "Rows retained:",
    f"{len(df):,}"
)

print(
    "Unique HDDs:",
    f"{df['serial_number'].nunique():,}"
)


# ==========================================================
# FIND CONTIGUOUS DAILY SEGMENTS
# ==========================================================

df = (
    df
    .sort_values(
        [
            "serial_number",
            "date"
        ]
    )
    .reset_index(
        drop=True
    )
)


date_gap = (
    df
    .groupby(
        "serial_number"
    )["date"]
    .diff()
    .dt.days
)


new_segment = (
    date_gap.isna()
    |
    (
        date_gap != 1
    )
)


df[
    "segment_number"
] = (
    new_segment
    .groupby(
        df[
            "serial_number"
        ]
    )
    .cumsum()
)


segment_summary = (
    df
    .groupby(
        [
            "serial_number",
            "segment_number"
        ],
        as_index=False
    )
    .agg(
        start_date=(
            "date",
            "min"
        ),

        end_date=(
            "date",
            "max"
        ),

        segment_length=(
            "date",
            "size"
        ),
    )
)


eligible_segments = (
    segment_summary[
        segment_summary[
            "segment_length"
        ] >= MIN_CONTIGUOUS_DAYS
    ]
    .copy()
)


# Keep longest eligible segment for each HDD

eligible_segments = (
    eligible_segments
    .sort_values(
        [
            "serial_number",
            "segment_length"
        ],
        ascending=[
            True,
            False
        ]
    )
)


eligible_drives = (
    eligible_segments
    .groupby(
        "serial_number",
        as_index=False
    )
    .first()
)


print(
    "\nEligible HDDs:",
    f"{len(eligible_drives):,}"
)


# ==========================================================
# STORAGE FOR ALL RUNS
# ==========================================================

all_link_results = []

sample_membership_rows = []

variability_rows = []

run_summary_rows = []


# ==========================================================
# STABILITY RUNS
# ==========================================================

for run_number, seed in enumerate(
    SEEDS,
    start=1
):

    print("\n")
    print("=" * 70)

    print(
        f"STABILITY RUN "
        f"{run_number}/{len(SEEDS)} "
        f"- SEED {seed}"
    )

    print("=" * 70)


    # ======================================================
    # SAMPLE HDDs
    # ======================================================

    selected = (
        eligible_drives
        .sample(
            n=min(
                HDDS_PER_RUN,
                len(
                    eligible_drives
                )
            ),
            random_state=seed
        )
        .copy()
    )


    selected[
        "seed"
    ] = seed


    sample_membership_rows.append(
        selected[
            [
                "seed",
                "serial_number",
                "segment_number",
                "start_date",
                "end_date",
                "segment_length",
            ]
        ]
    )


    # ======================================================
    # EXTRACT SELECTED SEGMENTS
    # ======================================================

    sample_df = df.merge(

        selected[
            [
                "serial_number",
                "segment_number",
            ]
        ],

        on=[
            "serial_number",
            "segment_number"
        ],

        how="inner"
    )


    sample_df = (
        sample_df
        .sort_values(
            [
                "serial_number",
                "date"
            ]
        )
        .reset_index(
            drop=True
        )
    )


    print(
        "Sample HDDs:",
        sample_df[
            "serial_number"
        ].nunique()
    )

    print(
        "Sample rows:",
        f"{len(sample_df):,}"
    )


    # ======================================================
    # VARIABILITY AUDIT FOR THIS SEED
    # ======================================================

    for feature in FEATURES:

        values = (
            sample_df[
                feature
            ]
        )


        zero_count = int(
            (
                values == 0
            ).sum()
        )


        variability_rows.append(
            {
                "seed":
                    seed,

                "feature":
                    feature,

                "rows":
                    len(
                        values
                    ),

                "unique_values":
                    int(
                        values.nunique()
                    ),

                "zero_rows":
                    zero_count,

                "zero_pct":
                    (
                        zero_count
                        / len(
                            values
                        )
                        * 100
                    ),

                "nonzero_rows":
                    int(
                        (
                            values != 0
                        ).sum()
                    ),

                "std":
                    float(
                        values.std()
                    ),
            }
        )


    # ======================================================
    # BUILD MULTIPLE DATASETS
    # ======================================================

    data_dict = {}


    for dataset_index, (
        serial,
        drive
    ) in enumerate(
        sample_df.groupby(
            "serial_number",
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
            .to_numpy(
                dtype=np.float64
            )
        )


        if not np.isfinite(
            values
        ).all():

            raise RuntimeError(
                f"Invalid values in HDD {serial}"
            )


        gaps = (
            drive[
                "date"
            ]
            .diff()
            .dt.days
            .dropna()
        )


        if not (
            gaps == 1
        ).all():

            raise RuntimeError(
                f"Non-daily gap found in HDD {serial}"
            )


        data_dict[
            dataset_index
        ] = values


    # ======================================================
    # TIGRAMITE DATAFRAME
    # ======================================================

    dataframe = pp.DataFrame(

        data=data_dict,

        analysis_mode="multiple",

        var_names=FEATURES,
    )


    cond_ind_test = RobustParCorr(
        significance="analytic"
    )


    pcmci = PCMCI(

        dataframe=dataframe,

        cond_ind_test=cond_ind_test,

        verbosity=0,
    )


    # ======================================================
    # RUN PCMCI
    # ======================================================

    start = time.time()


    results = pcmci.run_pcmci(

        tau_min=TAU_MIN,

        tau_max=TAU_MAX,

        pc_alpha=PC_ALPHA,

        max_conds_dim=MAX_CONDS_DIM,

        max_combinations=MAX_COMBINATIONS,

        max_conds_py=MAX_CONDS_PY,

        max_conds_px=MAX_CONDS_PX,

        alpha_level=FINAL_ALPHA,

        fdr_method="fdr_bh",
    )


    runtime = (
        time.time()
        - start
    )


    print(
        "Runtime:",
        round(
            runtime,
            2
        ),
        "seconds"
    )


    graph = results[
        "graph"
    ]

    p_matrix = results[
        "p_matrix"
    ]

    val_matrix = results[
        "val_matrix"
    ]


    # ======================================================
    # COLLECT CROSS-VARIABLE LINKS
    # ======================================================

    run_links = []


    for source_index, source in enumerate(
        FEATURES
    ):

        for target_index, target in enumerate(
            FEATURES
        ):

            if source == target:

                continue


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


                mci_value = float(
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
                    p_value <= FINAL_ALPHA
                    and
                    graph_mark != ""
                )


                if significant:

                    row = {
                        "seed":
                            seed,

                        "source":
                            source,

                        "target":
                            target,

                        "lag_days":
                            lag,

                        "mci_value":
                            mci_value,

                        "abs_mci_value":
                            abs(
                                mci_value
                            ),

                        "sign":
                            (
                                1
                                if mci_value > 0
                                else -1
                            ),

                        "fdr_p_value":
                            p_value,

                        "graph_mark":
                            graph_mark,
                    }


                    run_links.append(
                        row
                    )

                    all_link_results.append(
                        row
                    )


    run_links_df = pd.DataFrame(
        run_links
    )


    run_file = (
        OUTPUT_DIR
        / f"seed_{seed}_significant_links.csv"
    )


    run_links_df.to_csv(
        run_file,
        index=False
    )


    run_summary_rows.append(
        {
            "seed":
                seed,

            "hdds":
                len(
                    data_dict
                ),

            "rows":
                len(
                    sample_df
                ),

            "significant_cross_links":
                len(
                    run_links_df
                ),

            "runtime_seconds":
                runtime,
        }
    )


    print(
        "Significant cross-variable links:",
        len(
            run_links_df
        )
    )


# ==========================================================
# COMBINE ALL RUNS
# ==========================================================

links_df = pd.DataFrame(
    all_link_results
)


if links_df.empty:

    raise RuntimeError(
        "No significant links found in stability runs."
    )


links_file = (
    OUTPUT_DIR
    / "all_seed_significant_links.csv"
)


links_df.to_csv(
    links_file,
    index=False
)


# ==========================================================
# EXACT LINK STABILITY
#
# Same source + target + lag
# ==========================================================

exact_rows = []


for (
    source,
    target,
    lag
), group in links_df.groupby(
    [
        "source",
        "target",
        "lag_days"
    ]
):

    detected_runs = int(
        group[
            "seed"
        ].nunique()
    )


    positive_runs = int(
        (
            group[
                "mci_value"
            ] > 0
        ).sum()
    )


    negative_runs = int(
        (
            group[
                "mci_value"
            ] < 0
        ).sum()
    )


    sign_consistency = (
        max(
            positive_runs,
            negative_runs
        )
        / len(
            group
        )
    )


    exact_rows.append(
        {
            "source":
                source,

            "target":
                target,

            "lag_days":
                lag,

            "runs_detected":
                detected_runs,

            "detection_rate":
                detected_runs
                / len(
                    SEEDS
                ),

            "positive_runs":
                positive_runs,

            "negative_runs":
                negative_runs,

            "sign_consistency":
                sign_consistency,

            "mean_mci":
                group[
                    "mci_value"
                ].mean(),

            "median_mci":
                group[
                    "mci_value"
                ].median(),

            "median_abs_mci":
                group[
                    "abs_mci_value"
                ].median(),

            "max_abs_mci":
                group[
                    "abs_mci_value"
                ].max(),
        }
    )


exact_stability = pd.DataFrame(
    exact_rows
)


exact_stability = (
    exact_stability
    .sort_values(
        [
            "detection_rate",
            "sign_consistency",
            "median_abs_mci",
        ],
        ascending=[
            False,
            False,
            False,
        ]
    )
)


exact_file = (
    OUTPUT_DIR
    / "exact_link_stability.csv"
)


exact_stability.to_csv(
    exact_file,
    index=False
)


# ==========================================================
# STABLE EXACT LINKS
#
# Present in >= 4 of 5 runs
# and same sign >= 80%
# ==========================================================

stable_exact = (
    exact_stability[
        (
            exact_stability[
                "detection_rate"
            ]
            >= 0.80
        )
        &
        (
            exact_stability[
                "sign_consistency"
            ]
            >= 0.80
        )
    ]
    .copy()
)


stable_exact_file = (
    OUTPUT_DIR
    / "stable_exact_links.csv"
)


stable_exact.to_csv(
    stable_exact_file,
    index=False
)


# ==========================================================
# PAIR-LEVEL STABILITY
#
# A source -> target pair counts once per seed even if
# multiple lags are significant.
#
# Select strongest lag within each seed for that pair.
# ==========================================================

strongest_per_seed_pair = (
    links_df
    .sort_values(
        "abs_mci_value",
        ascending=False
    )
    .groupby(
        [
            "seed",
            "source",
            "target"
        ],
        as_index=False
    )
    .first()
)


pair_rows = []


for (
    source,
    target
), group in strongest_per_seed_pair.groupby(
    [
        "source",
        "target"
    ]
):

    detected_runs = int(
        group[
            "seed"
        ].nunique()
    )


    positive_runs = int(
        (
            group[
                "mci_value"
            ] > 0
        ).sum()
    )


    negative_runs = int(
        (
            group[
                "mci_value"
            ] < 0
        ).sum()
    )


    modal_lag = int(
        group[
            "lag_days"
        ]
        .mode()
        .iloc[0]
    )


    pair_rows.append(
        {
            "source":
                source,

            "target":
                target,

            "runs_detected":
                detected_runs,

            "detection_rate":
                detected_runs
                / len(
                    SEEDS
                ),

            "modal_lag_days":
                modal_lag,

            "min_lag_days":
                int(
                    group[
                        "lag_days"
                    ].min()
                ),

            "max_lag_days":
                int(
                    group[
                        "lag_days"
                    ].max()
                ),

            "positive_runs":
                positive_runs,

            "negative_runs":
                negative_runs,

            "sign_consistency":
                (
                    max(
                        positive_runs,
                        negative_runs
                    )
                    / len(
                        group
                    )
                ),

            "median_mci":
                group[
                    "mci_value"
                ].median(),

            "median_abs_mci":
                group[
                    "abs_mci_value"
                ].median(),
        }
    )


pair_stability = pd.DataFrame(
    pair_rows
)


pair_stability = (
    pair_stability
    .sort_values(
        [
            "detection_rate",
            "sign_consistency",
            "median_abs_mci",
        ],
        ascending=[
            False,
            False,
            False,
        ]
    )
)


pair_file = (
    OUTPUT_DIR
    / "pair_level_stability.csv"
)


pair_stability.to_csv(
    pair_file,
    index=False
)


stable_pairs = (
    pair_stability[
        (
            pair_stability[
                "detection_rate"
            ]
            >= 0.80
        )
        &
        (
            pair_stability[
                "sign_consistency"
            ]
            >= 0.80
        )
    ]
    .copy()
)


stable_pair_file = (
    OUTPUT_DIR
    / "stable_pair_links.csv"
)


stable_pairs.to_csv(
    stable_pair_file,
    index=False
)


# ==========================================================
# SAVE AUDITS
# ==========================================================

pd.concat(
    sample_membership_rows,
    ignore_index=True
).to_csv(
    AUDIT_DIR
    / "sample_membership.csv",
    index=False
)


pd.DataFrame(
    variability_rows
).to_csv(
    AUDIT_DIR
    / "variable_variability_by_seed.csv",
    index=False
)


run_summary_df = pd.DataFrame(
    run_summary_rows
)


run_summary_df.to_csv(
    OUTPUT_DIR
    / "stability_run_summary.csv",
    index=False
)


# ==========================================================
# PRINT RESULTS
# ==========================================================

print("\n")
print("=" * 70)
print("PCMCI STABILITY ANALYSIS COMPLETE")
print("=" * 70)


print(
    "\nRun summary:\n"
)

print(
    run_summary_df.to_string(
        index=False
    )
)


print(
    "\nStable exact links "
    "(>=80% runs, >=80% sign consistency):",
    len(
        stable_exact
    )
)


print(
    "\nTop stable exact links:\n"
)

print(
    stable_exact[
        [
            "source",
            "target",
            "lag_days",
            "detection_rate",
            "sign_consistency",
            "median_mci",
            "median_abs_mci",
        ]
    ]
    .head(
        30
    )
    .to_string(
        index=False
    )
)


print(
    "\nStable source-target pairs:",
    len(
        stable_pairs
    )
)


print(
    "\nTop stable pairs:\n"
)

print(
    stable_pairs[
        [
            "source",
            "target",
            "detection_rate",
            "modal_lag_days",
            "min_lag_days",
            "max_lag_days",
            "sign_consistency",
            "median_mci",
            "median_abs_mci",
        ]
    ]
    .head(
        30
    )
    .to_string(
        index=False
    )
)


print(
    "\nResults saved to:"
)

print(
    OUTPUT_DIR
)