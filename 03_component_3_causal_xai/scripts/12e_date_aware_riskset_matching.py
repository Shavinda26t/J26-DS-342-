from pathlib import Path

import gc
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "causal_effect"
    / "incident_risksets"
)

MANIFEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "pretreatment_cohort"
    / "pretreatment_adjustment_manifest.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "causal_effect"
    / "matched_risksets"
)

AUDIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "riskset_matching"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "figures"
    / "causal_effect"
    / "riskset_matching"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

AUDIT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# SETTINGS
# ==========================================================

TARGET = "failure_within_30d"

CONTROLS_PER_TREATED = 5

RANDOM_STATE = 42


TREATMENTS = [

    "treat_reallocated",

    "treat_uncorrectable",

    "treat_pending",
]


# ==========================================================
# MATCHING VARIABLES
#
# These are all measured before treatment at t-1.
#
# calendar_day_index is NOT needed for nearest-neighbour
# distance because matching is already exact by date.
#
# Prior event flags are also not distance variables because
# the risk-set design already requires no previous event.
# ==========================================================

MATCH_FEATURES = [

    "baseline_drive_age_hours_lag1",

    "baseline_temperature_lag1",

    "baseline_reallocated_sectors_lag1",

    "baseline_reported_uncorrectable_lag1",

    "baseline_pending_sectors_lag1",

    "baseline_crc_errors_lag1",

    "baseline_start_stop_activity_1d_lag1",

    "baseline_power_cycle_activity_1d_lag1",

    "baseline_poweroff_retract_activity_1d_lag1",

    "baseline_load_cycle_activity_1d_lag1",

    "baseline_write_workload_1d_lag1",

    "baseline_read_workload_1d_lag1",
]


# ==========================================================
# EXACT-MATCH DEGRADATION STATE
#
# Instead of allowing an HDD with no prior degradation state
# to match one that already has accumulated degradation,
# match the binary presence/absence pattern exactly.
# ==========================================================

EXACT_STATE_FEATURES = [

    "baseline_reallocated_sectors_lag1",

    "baseline_reported_uncorrectable_lag1",

    "baseline_pending_sectors_lag1",
]


# ==========================================================
# LOAD FULL BALANCE FEATURE LIST
# ==========================================================

manifest = pd.read_csv(
    MANIFEST_FILE
)


BALANCE_FEATURES = (
    manifest[
        "adjustment_feature"
    ]
    .tolist()
)


print(
    "Balance variables:",
    len(
        BALANCE_FEATURES
    )
)


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def weighted_mean(
    values,
    weights
):

    values = np.asarray(
        values,
        dtype=np.float64
    )

    weights = np.asarray(
        weights,
        dtype=np.float64
    )


    valid = (
        np.isfinite(
            values
        )
        &
        np.isfinite(
            weights
        )
        &
        (
            weights >= 0
        )
    )


    values = values[
        valid
    ]

    weights = weights[
        valid
    ]


    if (
        len(values) == 0
        or
        weights.sum() == 0
    ):

        return np.nan


    return np.average(
        values,
        weights=weights
    )


def weighted_variance(
    values,
    weights
):

    values = np.asarray(
        values,
        dtype=np.float64
    )

    weights = np.asarray(
        weights,
        dtype=np.float64
    )


    valid = (
        np.isfinite(
            values
        )
        &
        np.isfinite(
            weights
        )
        &
        (
            weights >= 0
        )
    )


    values = values[
        valid
    ]

    weights = weights[
        valid
    ]


    if (
        len(values) == 0
        or
        weights.sum() == 0
    ):

        return np.nan


    mean = np.average(
        values,
        weights=weights
    )


    return np.average(
        (
            values
            - mean
        ) ** 2,
        weights=weights
    )


def calculate_smd(
    treated_values,
    control_values,
    treated_weights=None,
    control_weights=None,
):

    treated_values = np.asarray(
        treated_values,
        dtype=np.float64
    )

    control_values = np.asarray(
        control_values,
        dtype=np.float64
    )


    if treated_weights is None:

        treated_weights = np.ones(
            len(
                treated_values
            )
        )


    if control_weights is None:

        control_weights = np.ones(
            len(
                control_values
            )
        )


    treated_mean = weighted_mean(
        treated_values,
        treated_weights
    )


    control_mean = weighted_mean(
        control_values,
        control_weights
    )


    treated_var = weighted_variance(
        treated_values,
        treated_weights
    )


    control_var = weighted_variance(
        control_values,
        control_weights
    )


    pooled_sd = np.sqrt(
        (
            treated_var
            +
            control_var
        )
        / 2
    )


    if (
        not np.isfinite(
            pooled_sd
        )
        or
        pooled_sd == 0
    ):

        if np.isclose(
            treated_mean,
            control_mean
        ):

            return 0.0

        return np.nan


    return (
        treated_mean
        -
        control_mean
    ) / pooled_sd


# ==========================================================
# STORAGE
# ==========================================================

overall_summary_rows = []

all_balance_results = []

all_match_quality_rows = []


# ==========================================================
# PROCESS EACH TREATMENT SEPARATELY
# ==========================================================

for treatment in TREATMENTS:

    print("\n")
    print("=" * 72)

    print(
        f"DATE-AWARE RISK-SET MATCHING: {treatment}"
    )

    print("=" * 72)


    input_file = (
        INPUT_DIR
        / f"{treatment}_incident_riskset.parquet"
    )


    if not input_file.exists():

        raise FileNotFoundError(
            f"Missing risk-set file: {input_file}"
        )


    # ======================================================
    # LOAD
    # ======================================================

    df = pd.read_parquet(
        input_file
    )


    df[
        "date"
    ] = pd.to_datetime(
        df[
            "date"
        ],
        errors="coerce"
    )


    df[
        TARGET
    ] = pd.to_numeric(
        df[
            TARGET
        ],
        errors="coerce"
    )


    df[
        "incident_treatment"
    ] = pd.to_numeric(
        df[
            "incident_treatment"
        ],
        errors="coerce"
    )


    for feature in BALANCE_FEATURES:

        df[
            feature
        ] = pd.to_numeric(
            df[
                feature
            ],
            errors="coerce"
        )


    if df[
        MATCH_FEATURES
    ].isna().any().any():

        raise RuntimeError(
            f"Missing matching variables in {treatment}"
        )


    print(
        "Risk-set rows:",
        f"{len(df):,}"
    )


    treated_original = (
        df[
            "incident_treatment"
        ]
        == 1
    )


    control_original = (
        df[
            "incident_treatment"
        ]
        == 0
    )


    print(
        "Treated:",
        f"{treated_original.sum():,}"
    )


    print(
        "Eligible controls:",
        f"{control_original.sum():,}"
    )


    # ======================================================
    # EXACT STATE INDICATORS
    # ======================================================

    df[
        "state_reallocated_present"
    ] = (
        df[
            "baseline_reallocated_sectors_lag1"
        ] > 0
    ).astype(
        "int8"
    )


    df[
        "state_uncorrectable_present"
    ] = (
        df[
            "baseline_reported_uncorrectable_lag1"
        ] > 0
    ).astype(
        "int8"
    )


    df[
        "state_pending_present"
    ] = (
        df[
            "baseline_pending_sectors_lag1"
        ] > 0
    ).astype(
        "int8"
    )


    df[
        "exact_state_key"
    ] = (
        df[
            [
                "state_reallocated_present",
                "state_uncorrectable_present",
                "state_pending_present",
            ]
        ]
        .astype(str)
        .agg(
            "_".join,
            axis=1
        )
    )


    # ======================================================
    # STANDARDIZATION PARAMETERS
    #
    # Standardize only variables used for distance.
    # ======================================================

    means = (
        df[
            MATCH_FEATURES
        ]
        .mean()
    )


    stds = (
        df[
            MATCH_FEATURES
        ]
        .std()
    )


    stds = stds.replace(
        0,
        1.0
    )


    stds = stds.fillna(
        1.0
    )


    # ======================================================
    # PRE-MATCHING BALANCE
    # ======================================================

    before_balance = {}


    for feature in BALANCE_FEATURES:

        before_balance[
            feature
        ] = calculate_smd(

            df.loc[
                treated_original,
                feature
            ],

            df.loc[
                control_original,
                feature
            ],
        )


    # ======================================================
    # MATCHING
    # ======================================================

    rng = np.random.default_rng(
        RANDOM_STATE
    )


    matched_rows = []

    match_quality_rows = []

    unmatched_rows = []


    match_set_counter = 0


    # Only dates containing treated incident cases need to
    # be processed.

    treated_dates = set(
        df.loc[
            treated_original,
            "date"
        ]
        .drop_duplicates()
        .tolist()
    )


    for date_value, date_df in df.groupby(
        "date",
        sort=True
    ):

        if date_value not in treated_dates:

            continue


        # --------------------------------------------------
        # Exact state strata inside the exact date
        # --------------------------------------------------

        treated_strata = (
            date_df.loc[
                date_df[
                    "incident_treatment"
                ] == 1,
                "exact_state_key"
            ]
            .unique()
        )


        for state_key in treated_strata:

            stratum = date_df[
                date_df[
                    "exact_state_key"
                ]
                == state_key
            ]


            treated_group = stratum[
                stratum[
                    "incident_treatment"
                ]
                == 1
            ]


            control_group = stratum[
                stratum[
                    "incident_treatment"
                ]
                == 0
            ]


            if len(
                control_group
            ) == 0:

                for treated_index in (
                    treated_group.index
                ):

                    unmatched_rows.append(
                        {
                            "treatment":
                                treatment,

                            "treated_index":
                                int(
                                    treated_index
                                ),

                            "serial_number":
                                df.loc[
                                    treated_index,
                                    "serial_number"
                                ],

                            "date":
                                date_value,

                            "exact_state_key":
                                state_key,

                            "reason":
                                "no_exact_state_control",
                        }
                    )

                continue


            # ------------------------------------------------
            # Standardized matching matrices
            # ------------------------------------------------

            control_x = (
                (
                    control_group[
                        MATCH_FEATURES
                    ]
                    -
                    means
                )
                /
                stds
            ).to_numpy(
                dtype=np.float64
            )


            treated_x = (
                (
                    treated_group[
                        MATCH_FEATURES
                    ]
                    -
                    means
                )
                /
                stds
            ).to_numpy(
                dtype=np.float64
            )


            control_indices = (
                control_group
                .index
                .to_numpy()
            )


            treated_indices = (
                treated_group
                .index
                .to_numpy()
            )


            # Controls cannot be reused within the same date
            # and exact-state stratum.

            control_used = np.zeros(
                len(
                    control_indices
                ),
                dtype=bool
            )


            treated_order = rng.permutation(
                len(
                    treated_indices
                )
            )


            for treated_position in treated_order:

                treated_index = (
                    treated_indices[
                        treated_position
                    ]
                )


                tx = treated_x[
                    treated_position
                ]


                available_positions = np.where(
                    ~control_used
                )[0]


                if len(
                    available_positions
                ) == 0:

                    unmatched_rows.append(
                        {
                            "treatment":
                                treatment,

                            "treated_index":
                                int(
                                    treated_index
                                ),

                            "serial_number":
                                df.loc[
                                    treated_index,
                                    "serial_number"
                                ],

                            "date":
                                date_value,

                            "exact_state_key":
                                state_key,

                            "reason":
                                "all_controls_already_used",
                        }
                    )

                    continue


                available_x = (
                    control_x[
                        available_positions
                    ]
                )


                squared_distance = np.sum(
                    (
                        available_x
                        -
                        tx
                    ) ** 2,
                    axis=1
                )


                number_to_select = min(

                    CONTROLS_PER_TREATED,

                    len(
                        available_positions
                    )
                )


                if number_to_select == 1:

                    local_best = np.array(
                        [
                            np.argmin(
                                squared_distance
                            )
                        ]
                    )

                else:

                    local_best = np.argpartition(

                        squared_distance,

                        number_to_select - 1

                    )[
                        :number_to_select
                    ]


                    local_best = local_best[
                        np.argsort(
                            squared_distance[
                                local_best
                            ]
                        )
                    ]


                selected_available_positions = (
                    available_positions[
                        local_best
                    ]
                )


                selected_control_indices = (
                    control_indices[
                        selected_available_positions
                    ]
                )


                selected_distances = np.sqrt(
                    squared_distance[
                        local_best
                    ]
                )


                control_used[
                    selected_available_positions
                ] = True


                match_set_counter += 1


                number_controls = len(
                    selected_control_indices
                )


                # --------------------------------------------
                # Add treated row
                # --------------------------------------------

                treated_row = (
                    df.loc[
                        treated_index
                    ]
                    .copy()
                )


                treated_row[
                    "match_set_id"
                ] = match_set_counter


                treated_row[
                    "match_role"
                ] = "treated"


                treated_row[
                    "match_weight"
                ] = 1.0


                treated_row[
                    "match_distance"
                ] = 0.0


                treated_row[
                    "controls_in_set"
                ] = number_controls


                matched_rows.append(
                    treated_row
                )


                # --------------------------------------------
                # Add matched controls
                #
                # Total control weight per treated set = 1.
                # --------------------------------------------

                control_weight = (
                    1.0
                    /
                    number_controls
                )


                for (
                    control_index,
                    distance
                ) in zip(
                    selected_control_indices,
                    selected_distances
                ):

                    control_row = (
                        df.loc[
                            control_index
                        ]
                        .copy()
                    )


                    control_row[
                        "match_set_id"
                    ] = match_set_counter


                    control_row[
                        "match_role"
                    ] = "control"


                    control_row[
                        "match_weight"
                    ] = control_weight


                    control_row[
                        "match_distance"
                    ] = float(
                        distance
                    )


                    control_row[
                        "controls_in_set"
                    ] = number_controls


                    matched_rows.append(
                        control_row
                    )


                match_quality_rows.append(
                    {
                        "treatment":
                            treatment,

                        "match_set_id":
                            match_set_counter,

                        "treated_serial_number":
                            df.loc[
                                treated_index,
                                "serial_number"
                            ],

                        "date":
                            date_value,

                        "exact_state_key":
                            state_key,

                        "controls_matched":
                            number_controls,

                        "mean_control_distance":
                            float(
                                np.mean(
                                    selected_distances
                                )
                            ),

                        "max_control_distance":
                            float(
                                np.max(
                                    selected_distances
                                )
                            ),

                        "min_control_distance":
                            float(
                                np.min(
                                    selected_distances
                                )
                            ),
                    }
                )


    # ======================================================
    # CREATE MATCHED DATASET
    # ======================================================

    if not matched_rows:

        raise RuntimeError(
            f"No matched observations for {treatment}"
        )


    matched_df = pd.DataFrame(
        matched_rows
    )


    matched_df[
        "date"
    ] = pd.to_datetime(
        matched_df[
            "date"
        ]
    )


    matched_treated = (
        matched_df[
            "match_role"
        ]
        == "treated"
    )


    matched_control = (
        matched_df[
            "match_role"
        ]
        == "control"
    )


    matched_treated_count = int(
        matched_treated.sum()
    )


    matched_control_count = int(
        matched_control.sum()
    )


    original_treated_count = int(
        treated_original.sum()
    )


    unmatched_treated_count = (
        original_treated_count
        -
        matched_treated_count
    )


    treated_retention_pct = (
        matched_treated_count
        /
        original_treated_count
        *
        100
    )


    # ======================================================
    # POST-MATCHING BALANCE
    # ======================================================

    balance_rows = []


    for feature in BALANCE_FEATURES:

        before_smd = (
            before_balance[
                feature
            ]
        )


        after_smd = calculate_smd(

            matched_df.loc[
                matched_treated,
                feature
            ],

            matched_df.loc[
                matched_control,
                feature
            ],

            treated_weights=
                matched_df.loc[
                    matched_treated,
                    "match_weight"
                ],

            control_weights=
                matched_df.loc[
                    matched_control,
                    "match_weight"
                ],
        )


        balance_rows.append(
            {
                "treatment":
                    treatment,

                "feature":
                    feature,

                "smd_before":
                    before_smd,

                "abs_smd_before":
                    (
                        abs(
                            before_smd
                        )
                        if np.isfinite(
                            before_smd
                        )
                        else np.nan
                    ),

                "smd_after_matching":
                    after_smd,

                "abs_smd_after_matching":
                    (
                        abs(
                            after_smd
                        )
                        if np.isfinite(
                            after_smd
                        )
                        else np.nan
                    ),
            }
        )


    balance_df = pd.DataFrame(
        balance_rows
    )


    balance_df = (
        balance_df
        .sort_values(
            "abs_smd_after_matching",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


    all_balance_results.append(
        balance_df
    )


    # ======================================================
    # BALANCE SUMMARY
    # ======================================================

    max_smd_before = (
        balance_df[
            "abs_smd_before"
        ]
        .max()
    )


    max_smd_after = (
        balance_df[
            "abs_smd_after_matching"
        ]
        .max()
    )


    features_ge_010_after = int(
        (
            balance_df[
                "abs_smd_after_matching"
            ]
            >= 0.10
        ).sum()
    )


    features_ge_005_after = int(
        (
            balance_df[
                "abs_smd_after_matching"
            ]
            >= 0.05
        ).sum()
    )


    # ======================================================
    # OUTCOME COUNTS
    #
    # These are diagnostic only.
    #
    # Formal uncertainty/effect estimation comes in Step 12F.
    # ======================================================

    treated_positive = int(
        (
            matched_df.loc[
                matched_treated,
                TARGET
            ]
            == 1
        ).sum()
    )


    control_positive_rows = int(
        (
            matched_df.loc[
                matched_control,
                TARGET
            ]
            == 1
        ).sum()
    )


    treated_failure_rate = weighted_mean(

        matched_df.loc[
            matched_treated,
            TARGET
        ],

        matched_df.loc[
            matched_treated,
            "match_weight"
        ],
    )


    weighted_control_failure_rate = weighted_mean(

        matched_df.loc[
            matched_control,
            TARGET
        ],

        matched_df.loc[
            matched_control,
            "match_weight"
        ],
    )


    # ======================================================
    # MATCH QUALITY TABLE
    # ======================================================

    match_quality_df = pd.DataFrame(
        match_quality_rows
    )


    treatment_quality = (
        match_quality_df[
            match_quality_df[
                "treatment"
            ] == treatment
        ]
        .copy()
    )


    if len(
        treatment_quality
    ) > 0:

        median_mean_distance = float(
            treatment_quality[
                "mean_control_distance"
            ].median()
        )


        p95_mean_distance = float(
            treatment_quality[
                "mean_control_distance"
            ].quantile(
                0.95
            )
        )


        max_distance = float(
            treatment_quality[
                "max_control_distance"
            ].max()
        )

    else:

        median_mean_distance = np.nan
        p95_mean_distance = np.nan
        max_distance = np.nan


    # ======================================================
    # READINESS CLASSIFICATION
    # ======================================================

    retention_ok = (
        treated_retention_pct
        >= 80.0
    )


    balance_ok = (
        max_smd_after
        < 0.10
    )


    if (
        retention_ok
        and
        balance_ok
    ):

        status = (
            "READY_FOR_MATCHED_EFFECT_ESTIMATION"
        )

    else:

        status = (
            "REVIEW_MATCHING_BEFORE_EFFECT_ESTIMATION"
        )


    # ======================================================
    # SAVE MATCHED DATA
    # ======================================================

    matched_output_file = (
        OUTPUT_DIR
        / f"{treatment}_matched.parquet"
    )


    matched_df.to_parquet(
        matched_output_file,
        index=False,
        compression="zstd"
    )


    # ======================================================
    # SAVE TREATMENT BALANCE
    # ======================================================

    treatment_balance_file = (
        AUDIT_DIR
        / f"{treatment}_balance.csv"
    )


    balance_df.to_csv(
        treatment_balance_file,
        index=False
    )


    # ======================================================
    # STORE MATCH QUALITY
    # ======================================================

    all_match_quality_rows.extend(
        treatment_quality.to_dict(
            orient="records"
        )
    )


    # ======================================================
    # FIGURE
    # ======================================================

    plot_df = (
        balance_df
        .sort_values(
            "abs_smd_before",
            ascending=True
        )
    )


    y_positions = np.arange(
        len(
            plot_df
        )
    )


    plt.figure(
        figsize=(
            11,
            8
        )
    )


    plt.scatter(

        plot_df[
            "smd_before"
        ],

        y_positions,

        label="Before matching"
    )


    plt.scatter(

        plot_df[
            "smd_after_matching"
        ],

        y_positions,

        label="After matching"
    )


    plt.axvline(
        0,
        linewidth=1
    )


    plt.axvline(
        -0.10,
        linestyle="--",
        linewidth=1
    )


    plt.axvline(
        0.10,
        linestyle="--",
        linewidth=1
    )


    plt.yticks(

        y_positions,

        plot_df[
            "feature"
        ]
    )


    plt.xlabel(
        "Standardized Mean Difference (SMD)"
    )


    plt.title(
        f"Date-aware risk-set matching balance: {treatment}"
    )


    plt.legend()


    plt.tight_layout()


    figure_file = (
        FIGURE_DIR
        / f"{treatment}_matching_balance.png"
    )


    plt.savefig(
        figure_file,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    # ======================================================
    # SUMMARY
    # ======================================================

    overall_summary_rows.append(
        {
            "treatment":
                treatment,

            "original_treated":
                original_treated_count,

            "matched_treated":
                matched_treated_count,

            "unmatched_treated":
                unmatched_treated_count,

            "treated_retention_pct":
                treated_retention_pct,

            "matched_control_rows":
                matched_control_count,

            "matched_control_hdds":
                matched_df.loc[
                    matched_control,
                    "serial_number"
                ].nunique(),

            "treated_positive_rows":
                treated_positive,

            "control_positive_rows_unweighted":
                control_positive_rows,

            "treated_failure_rate":
                treated_failure_rate,

            "weighted_control_failure_rate":
                weighted_control_failure_rate,

            "max_abs_smd_before":
                max_smd_before,

            "max_abs_smd_after_matching":
                max_smd_after,

            "features_abs_smd_ge_010_after":
                features_ge_010_after,

            "features_abs_smd_ge_005_after":
                features_ge_005_after,

            "median_mean_match_distance":
                median_mean_distance,

            "p95_mean_match_distance":
                p95_mean_distance,

            "maximum_match_distance":
                max_distance,

            "status":
                status,
        }
    )


    # ======================================================
    # PRINT TREATMENT RESULTS
    # ======================================================

    print(
        "\nMatching results:"
    )


    print(
        "Original treated:",
        original_treated_count
    )


    print(
        "Matched treated:",
        matched_treated_count
    )


    print(
        "Unmatched treated:",
        unmatched_treated_count
    )


    print(
        "Treated retention:",
        f"{treated_retention_pct:.2f}%"
    )


    print(
        "Matched control rows:",
        matched_control_count
    )


    print(
        "Treated positive outcomes:",
        treated_positive
    )


    print(
        "Maximum |SMD| before:",
        round(
            max_smd_before,
            4
        )
    )


    print(
        "Maximum |SMD| after:",
        round(
            max_smd_after,
            4
        )
    )


    print(
        "Features |SMD| >= 0.10 after:",
        features_ge_010_after
    )


    print(
        "\nWorst post-matching balance:\n"
    )


    print(
        balance_df[
            [
                "feature",
                "smd_before",
                "smd_after_matching",
            ]
        ]
        .head(
            10
        )
        .to_string(
            index=False
        )
    )


    print(
        "\nStatus:",
        status
    )


    print(
        "Saved:",
        matched_output_file
    )


    # ======================================================
    # MEMORY CLEANUP
    # ======================================================

    del df
    del matched_df
    del balance_df

    gc.collect()


# ==========================================================
# COMBINE RESULTS
# ==========================================================

summary_df = pd.DataFrame(
    overall_summary_rows
)


all_balance_df = pd.concat(
    all_balance_results,
    ignore_index=True
)


match_quality_df = pd.DataFrame(
    all_match_quality_rows
)


# ==========================================================
# SAVE
# ==========================================================

summary_file = (
    AUDIT_DIR
    / "riskset_matching_summary.csv"
)


balance_file = (
    AUDIT_DIR
    / "riskset_matching_balance_all.csv"
)


quality_file = (
    AUDIT_DIR
    / "riskset_match_quality.csv"
)


summary_df.to_csv(
    summary_file,
    index=False
)


all_balance_df.to_csv(
    balance_file,
    index=False
)


match_quality_df.to_csv(
    quality_file,
    index=False
)


# ==========================================================
# METADATA
# ==========================================================

metadata = {

    "matching_design":
        "exact-date risk-set nearest-neighbour matching",

    "estimand":
        "ATT among clean incident treatment HDDs",

    "controls_per_treated":
        CONTROLS_PER_TREATED,

    "control_replacement_within_date":
        False,

    "exact_date_matching":
        True,

    "exact_state_variables":
        EXACT_STATE_FEATURES,

    "distance_variables":
        MATCH_FEATURES,

    "balance_variables":
        BALANCE_FEATURES,

    "random_state":
        RANDOM_STATE,

    "primary_balance_threshold":
        0.10,
}


metadata_file = (
    AUDIT_DIR
    / "riskset_matching_metadata.json"
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
# FINAL PRINT
# ==========================================================

print("\n")
print("=" * 72)

print(
    "DATE-AWARE INCIDENT RISK-SET MATCHING COMPLETE"
)

print("=" * 72)


print(
    "\nSummary:\n"
)


display_columns = [

    "treatment",

    "original_treated",

    "matched_treated",

    "unmatched_treated",

    "treated_retention_pct",

    "matched_control_rows",

    "treated_positive_rows",

    "max_abs_smd_before",

    "max_abs_smd_after_matching",

    "features_abs_smd_ge_010_after",

    "status",
]


print(
    summary_df[
        display_columns
    ].to_string(
        index=False
    )
)


print(
    "\nMatched datasets saved to:"
)


print(
    OUTPUT_DIR
)


print(
    "\nDiagnostics saved to:"
)


print(
    AUDIT_DIR
)


print(
    "\nFigures saved to:"
)


print(
    FIGURE_DIR
)