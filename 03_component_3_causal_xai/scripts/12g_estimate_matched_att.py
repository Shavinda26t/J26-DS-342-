from pathlib import Path
import json

import numpy as np
import pandas as pd

from scipy.stats import binomtest


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "causal_effect"
    / "matched_risksets_refined"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "causal_effect"
    / "matched_att"
)

AUDIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "matched_att"
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

TARGET = "failure_within_30d"

TREATMENTS = [

    "treat_reallocated",

    "treat_uncorrectable",

    "treat_pending",
]


BOOTSTRAP_ITERATIONS = 20000

RANDOM_STATE = 42

CI_LOWER = 2.5
CI_UPPER = 97.5


# ==========================================================
# HELPER
# ==========================================================

def percentile_ci(
    values,
    lower=CI_LOWER,
    upper=CI_UPPER
):

    values = np.asarray(
        values,
        dtype=np.float64
    )


    values = values[
        np.isfinite(
            values
        )
    ]


    if len(values) == 0:

        return (
            np.nan,
            np.nan
        )


    return (
        float(
            np.percentile(
                values,
                lower
            )
        ),

        float(
            np.percentile(
                values,
                upper
            )
        ),
    )


# ==========================================================
# STORAGE
# ==========================================================

summary_rows = []

discordance_rows = []

bootstrap_summary_rows = []


# ==========================================================
# PROCESS EACH TREATMENT
# ==========================================================

for treatment in TREATMENTS:

    print("\n")
    print("=" * 74)

    print(
        f"MATCHED ATT ESTIMATION: {treatment}"
    )

    print("=" * 74)


    input_file = (
        INPUT_DIR
        / f"{treatment}_history_corrected_matched.parquet"
    )


    if not input_file.exists():

        raise FileNotFoundError(
            f"Missing matched dataset: {input_file}"
        )


    df = pd.read_parquet(
        input_file
    )


    # ======================================================
    # BASIC VALIDATION
    # ======================================================

    required_columns = [

        "match_set_id",

        "match_role",

        "serial_number",

        "date",

        TARGET,
    ]


    missing_columns = [

        column

        for column in required_columns

        if column not in df.columns
    ]


    if missing_columns:

        raise RuntimeError(
            f"Missing columns in {treatment}: "
            f"{missing_columns}"
        )


    df[
        TARGET
    ] = pd.to_numeric(
        df[
            TARGET
        ],
        errors="coerce"
    )


    if df[
        TARGET
    ].isna().any():

        raise RuntimeError(
            f"Missing outcome values in {treatment}"
        )


    unexpected_roles = (
        set(
            df[
                "match_role"
            ].unique()
        )
        -
        {
            "treated",
            "control"
        }
    )


    if unexpected_roles:

        raise RuntimeError(
            f"Unexpected match roles in {treatment}: "
            f"{unexpected_roles}"
        )


    # ======================================================
    # MATCH-SET AUDIT
    #
    # Every set must contain exactly:
    #
    # 1 treated HDD
    # 1 matched control HDD
    # ======================================================

    set_sizes = (
        df
        .groupby(
            "match_set_id"
        )
        .size()
    )


    bad_size_sets = int(
        (
            set_sizes != 2
        ).sum()
    )


    role_counts = (

        df
        .groupby(
            [
                "match_set_id",
                "match_role"
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )


    for role in [
        "treated",
        "control"
    ]:

        if role not in role_counts.columns:

            role_counts[
                role
            ] = 0


    bad_role_sets = int(
        (
            (
                role_counts[
                    "treated"
                ] != 1
            )
            |
            (
                role_counts[
                    "control"
                ] != 1
            )
        ).sum()
    )


    if (
        bad_size_sets > 0
        or
        bad_role_sets > 0
    ):

        raise RuntimeError(
            f"Invalid matched sets for {treatment}. "
            f"Bad sizes={bad_size_sets}, "
            f"bad roles={bad_role_sets}"
        )


    # ======================================================
    # BUILD ONE ROW PER MATCHED PAIR
    # ======================================================

    treated_df = (

        df[
            df[
                "match_role"
            ] == "treated"
        ][
            [
                "match_set_id",
                "serial_number",
                "date",
                TARGET,
            ]
        ]
        .rename(
            columns={
                "serial_number":
                    "treated_serial_number",

                TARGET:
                    "treated_outcome",
            }
        )
    )


    control_df = (

        df[
            df[
                "match_role"
            ] == "control"
        ][
            [
                "match_set_id",
                "serial_number",
                "date",
                TARGET,
            ]
        ]
        .rename(
            columns={
                "serial_number":
                    "control_serial_number",

                TARGET:
                    "control_outcome",
            }
        )
    )


    pairs = treated_df.merge(

        control_df,

        on=[
            "match_set_id",
            "date"
        ],

        how="inner",

        validate="one_to_one",
    )


    if len(
        pairs
    ) != len(
        role_counts
    ):

        raise RuntimeError(
            f"Pair merge mismatch for {treatment}"
        )


    # ======================================================
    # PAIR-LEVEL DIFFERENCE
    #
    # D_i = Y_treated - Y_control
    #
    # ATT_RD = mean(D_i)
    # ======================================================

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


    n_pairs = len(
        pairs
    )


    treated_events = int(
        pairs[
            "treated_outcome"
        ].sum()
    )


    control_events = int(
        pairs[
            "control_outcome"
        ].sum()
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


    att_risk_difference = (

        treated_risk

        -

        control_risk
    )


    # ======================================================
    # RISK RATIO
    #
    # Ordinary RR is infinite when control risk is zero.
    #
    # It is saved for completeness, but RD remains primary.
    # ======================================================

    if control_risk > 0:

        crude_matched_rr = (

            treated_risk

            /

            control_risk
        )

    else:

        crude_matched_rr = np.inf


    # ======================================================
    # HALDANE-ANSCOMBE CORRECTED RR
    #
    # Supplementary only.
    #
    # Add 0.5 events and 0.5 non-events to each group.
    # Do NOT use this as the primary causal estimate.
    # ======================================================

    treated_non_events = (
        n_pairs
        -
        treated_events
    )


    control_non_events = (
        n_pairs
        -
        control_events
    )


    treated_corrected_risk = (

        treated_events
        +
        0.5

    ) / (

        treated_events
        +
        treated_non_events
        +
        1.0
    )


    control_corrected_risk = (

        control_events
        +
        0.5

    ) / (

        control_events
        +
        control_non_events
        +
        1.0
    )


    corrected_rr = (

        treated_corrected_risk

        /

        control_corrected_risk
    )


    # ======================================================
    # DISCORDANT PAIRS
    #
    # n10 = treated failed, control did not
    # n01 = control failed, treated did not
    # n11 = both failed
    # n00 = neither failed
    # ======================================================

    n10 = int(
        (
            (
                pairs[
                    "treated_outcome"
                ] == 1
            )
            &
            (
                pairs[
                    "control_outcome"
                ] == 0
            )
        ).sum()
    )


    n01 = int(
        (
            (
                pairs[
                    "treated_outcome"
                ] == 0
            )
            &
            (
                pairs[
                    "control_outcome"
                ] == 1
            )
        ).sum()
    )


    n11 = int(
        (
            (
                pairs[
                    "treated_outcome"
                ] == 1
            )
            &
            (
                pairs[
                    "control_outcome"
                ] == 1
            )
        ).sum()
    )


    n00 = int(
        (
            (
                pairs[
                    "treated_outcome"
                ] == 0
            )
            &
            (
                pairs[
                    "control_outcome"
                ] == 0
            )
        ).sum()
    )


    discordant_total = (
        n10
        +
        n01
    )


    # ======================================================
    # EXACT McNEMAR TEST
    #
    # Under the null, discordant outcomes are equally likely
    # to favor treated or control.
    # ======================================================

    if discordant_total > 0:

        mcnemar_result = binomtest(

            k=min(
                n10,
                n01
            ),

            n=discordant_total,

            p=0.5,

            alternative="two-sided",
        )


        mcnemar_p_value = float(
            mcnemar_result.pvalue
        )

    else:

        mcnemar_p_value = np.nan


    # ======================================================
    # MATCHED-SET BOOTSTRAP
    #
    # Resample whole matched pairs.
    #
    # This preserves treatment-control pairing.
    # ======================================================

    rng = np.random.default_rng(
        RANDOM_STATE
    )


    differences = (
        pairs[
            "pair_difference"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )


    treated_values = (
        pairs[
            "treated_outcome"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )


    control_values = (
        pairs[
            "control_outcome"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )


    bootstrap_rd = np.empty(
        BOOTSTRAP_ITERATIONS,
        dtype=np.float64
    )


    bootstrap_treated_risk = np.empty(
        BOOTSTRAP_ITERATIONS,
        dtype=np.float64
    )


    bootstrap_control_risk = np.empty(
        BOOTSTRAP_ITERATIONS,
        dtype=np.float64
    )


    for bootstrap_index in range(
        BOOTSTRAP_ITERATIONS
    ):

        sample_indices = rng.integers(

            0,

            n_pairs,

            size=n_pairs,
        )


        sampled_differences = (
            differences[
                sample_indices
            ]
        )


        sampled_treated = (
            treated_values[
                sample_indices
            ]
        )


        sampled_control = (
            control_values[
                sample_indices
            ]
        )


        bootstrap_rd[
            bootstrap_index
        ] = np.mean(
            sampled_differences
        )


        bootstrap_treated_risk[
            bootstrap_index
        ] = np.mean(
            sampled_treated
        )


        bootstrap_control_risk[
            bootstrap_index
        ] = np.mean(
            sampled_control
        )


    # ======================================================
    # BOOTSTRAP CIs
    # ======================================================

    rd_ci_lower, rd_ci_upper = percentile_ci(
        bootstrap_rd
    )


    treated_ci_lower, treated_ci_upper = percentile_ci(
        bootstrap_treated_risk
    )


    control_ci_lower, control_ci_upper = percentile_ci(
        bootstrap_control_risk
    )


    # ======================================================
    # BOOTSTRAP SIGN PROBABILITY
    #
    # Descriptive robustness:
    # proportion of bootstrap estimates > 0.
    #
    # Not interpreted as a Bayesian posterior probability.
    # ======================================================

    bootstrap_positive_fraction = float(
        (
            bootstrap_rd > 0
        ).mean()
    )


    # ======================================================
    # CONTROL REUSE AUDIT
    #
    # Matching allowed replacement.
    # ======================================================

    control_use_counts = (

        pairs[
            "control_serial_number"
        ]
        .value_counts()
    )


    unique_control_hdds = int(
        control_use_counts.shape[
            0
        ]
    )


    maximum_control_reuse = int(
        control_use_counts.max()
    )


    controls_reused_more_than_once = int(
        (
            control_use_counts > 1
        ).sum()
    )


    # ======================================================
    # SAVE PAIR DATA
    # ======================================================

    pairs_file = (
        AUDIT_DIR
        / f"{treatment}_matched_pairs.csv"
    )


    pairs.to_csv(
        pairs_file,
        index=False
    )


    # ======================================================
    # SAVE BOOTSTRAP DISTRIBUTION
    # ======================================================

    bootstrap_file = (
        AUDIT_DIR
        / f"{treatment}_bootstrap_distribution.npz"
    )


    np.savez_compressed(

        bootstrap_file,

        risk_difference=
            bootstrap_rd,

        treated_risk=
            bootstrap_treated_risk,

        control_risk=
            bootstrap_control_risk,
    )


    # ======================================================
    # SUMMARY
    # ======================================================

    summary_rows.append(
        {
            "treatment":
                treatment,

            "matched_pairs":
                n_pairs,

            "treated_events":
                treated_events,

            "control_events":
                control_events,

            "treated_risk":
                treated_risk,

            "control_risk":
                control_risk,

            "att_risk_difference":
                att_risk_difference,

            "att_risk_difference_percentage_points":
                att_risk_difference
                * 100,

            "bootstrap_95ci_lower":
                rd_ci_lower,

            "bootstrap_95ci_upper":
                rd_ci_upper,

            "bootstrap_95ci_lower_percentage_points":
                rd_ci_lower
                * 100,

            "bootstrap_95ci_upper_percentage_points":
                rd_ci_upper
                * 100,

            "bootstrap_positive_fraction":
                bootstrap_positive_fraction,

            "matched_risk_ratio":
                crude_matched_rr,

            "haldane_anscombe_corrected_rr":
                corrected_rr,

            "mcnemar_exact_p_value":
                mcnemar_p_value,

            "n10_treated_only_failure":
                n10,

            "n01_control_only_failure":
                n01,

            "n11_both_failure":
                n11,

            "n00_neither_failure":
                n00,

            "unique_control_hdds":
                unique_control_hdds,

            "controls_reused_more_than_once":
                controls_reused_more_than_once,

            "maximum_control_reuse":
                maximum_control_reuse,
        }
    )


    discordance_rows.append(
        {
            "treatment":
                treatment,

            "n10_treated_failure_control_no_failure":
                n10,

            "n01_control_failure_treated_no_failure":
                n01,

            "n11_both_failure":
                n11,

            "n00_neither_failure":
                n00,

            "discordant_pairs":
                discordant_total,

            "mcnemar_exact_p_value":
                mcnemar_p_value,
        }
    )


    bootstrap_summary_rows.append(
        {
            "treatment":
                treatment,

            "bootstrap_iterations":
                BOOTSTRAP_ITERATIONS,

            "point_att_rd":
                att_risk_difference,

            "ci_lower":
                rd_ci_lower,

            "ci_upper":
                rd_ci_upper,

            "positive_bootstrap_fraction":
                bootstrap_positive_fraction,
        }
    )


    # ======================================================
    # PRINT
    # ======================================================

    print(
        "Matched pairs:",
        n_pairs
    )


    print(
        "Treated failures:",
        treated_events
    )


    print(
        "Control failures:",
        control_events
    )


    print(
        "Treated 30-day risk:",
        f"{treated_risk:.6f}"
    )


    print(
        "Control 30-day risk:",
        f"{control_risk:.6f}"
    )


    print(
        "\nATT risk difference:",
        f"{att_risk_difference:.6f}"
    )


    print(
        "ATT risk difference:",
        f"{att_risk_difference * 100:.3f}",
        "percentage points"
    )


    print(
        "Bootstrap 95% CI:",
        f"[{rd_ci_lower * 100:.3f}, "
        f"{rd_ci_upper * 100:.3f}]",
        "percentage points"
    )


    print(
        "Bootstrap estimates > 0:",
        f"{bootstrap_positive_fraction * 100:.2f}%"
    )


    print(
        "\nDiscordant pairs:"
    )


    print(
        " Treated failure / control no failure:",
        n10
    )


    print(
        " Control failure / treated no failure:",
        n01
    )


    print(
        " Both failure:",
        n11
    )


    print(
        " Neither failure:",
        n00
    )


    print(
        "McNemar exact p-value:",
        mcnemar_p_value
    )


    print(
        "\nOrdinary matched RR:",
        crude_matched_rr
    )


    print(
        "Corrected RR (supplementary):",
        corrected_rr
    )


    print(
        "\nUnique matched control HDDs:",
        unique_control_hdds
    )


    print(
        "Controls reused >1 time:",
        controls_reused_more_than_once
    )


    print(
        "Maximum reuse of one control HDD:",
        maximum_control_reuse
    )


# ==========================================================
# SAVE RESULTS
# ==========================================================

summary_df = pd.DataFrame(
    summary_rows
)


discordance_df = pd.DataFrame(
    discordance_rows
)


bootstrap_summary_df = pd.DataFrame(
    bootstrap_summary_rows
)


summary_file = (
    OUTPUT_DIR
    / "matched_att_effect_summary.csv"
)


discordance_file = (
    OUTPUT_DIR
    / "matched_pair_discordance.csv"
)


bootstrap_summary_file = (
    OUTPUT_DIR
    / "matched_att_bootstrap_summary.csv"
)


summary_df.to_csv(
    summary_file,
    index=False
)


discordance_df.to_csv(
    discordance_file,
    index=False
)


bootstrap_summary_df.to_csv(
    bootstrap_summary_file,
    index=False
)


# ==========================================================
# PRIMARY / SECONDARY ROLE
# ==========================================================

role_df = pd.DataFrame(
    [
        {
            "treatment":
                "treat_uncorrectable",

            "analysis_role":
                "PRIMARY",

            "reason":
                (
                    "Largest number of treated failure "
                    "outcomes after balanced incident matching."
                ),
        },

        {
            "treatment":
                "treat_reallocated",

            "analysis_role":
                "SECONDARY_EXPLORATORY",

            "reason":
                (
                    "Good matching balance but few treated "
                    "failure outcomes."
                ),
        },

        {
            "treatment":
                "treat_pending",

            "analysis_role":
                "EXPLORATORY_ONLY",

            "reason":
                (
                    "Very small incident treatment sample and "
                    "very few treated failure outcomes."
                ),
        },
    ]
)


role_file = (
    OUTPUT_DIR
    / "causal_effect_analysis_roles.csv"
)


role_df.to_csv(
    role_file,
    index=False
)


# ==========================================================
# METADATA
# ==========================================================

metadata = {

    "primary_estimand":
        "matched ATT risk difference",

    "outcome":
        TARGET,

    "outcome_window":
        "failure within 30 days",

    "bootstrap_unit":
        "matched treatment-control set",

    "bootstrap_iterations":
        BOOTSTRAP_ITERATIONS,

    "confidence_interval":
        "95% percentile bootstrap",

    "paired_test":
        "exact McNemar/binomial test",

    "risk_ratio_role":
        "supplementary",

    "primary_treatment":
        "treat_uncorrectable",

    "interpretation_caution":
        (
            "Observational matched effect estimate under "
            "measured-confounding and matching assumptions; "
            "not proof of physical causation."
        ),
}


metadata_file = (
    OUTPUT_DIR
    / "matched_att_metadata.json"
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
print("=" * 74)

print(
    "MATCHED ATT EFFECT ESTIMATION COMPLETE"
)

print("=" * 74)


print(
    "\nFinal effect summary:\n"
)


display_columns = [

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
]


print(
    summary_df[
        display_columns
    ].to_string(
        index=False
    )
)


print(
    "\nAnalysis roles:\n"
)


print(
    role_df.to_string(
        index=False
    )
)


print(
    "\nResults saved to:"
)


print(
    OUTPUT_DIR
)


print(
    "\nAudit files saved to:"
)


print(
    AUDIT_DIR
)