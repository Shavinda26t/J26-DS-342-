from pathlib import Path

import numpy as np
import pandas as pd


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

COHORT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "causal_effect"
    / "pretreatment_failure_cohort.parquet"
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
    / "incident_risksets"
)

AUDIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "incident_risksets"
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


TREATMENT_MAP = {

    "treat_reallocated":
        "isolated_reallocated_event",

    "treat_uncorrectable":
        "isolated_uncorrectable_event",

    "treat_pending":
        "isolated_pending_event",
}


TREATMENTS = list(
    TREATMENT_MAP.keys()
)


ISOLATED_FLAGS = list(
    TREATMENT_MAP.values()
)


# ==========================================================
# LOAD ADJUSTMENT VARIABLES
# ==========================================================

manifest = pd.read_csv(
    MANIFEST_FILE
)


ADJUSTMENT_FEATURES = (
    manifest[
        "adjustment_feature"
    ]
    .tolist()
)


print(
    "Adjustment variables:",
    len(
        ADJUSTMENT_FEATURES
    )
)


# ==========================================================
# LOAD COHORT
# ==========================================================

columns = (

    [
        "date",
        "serial_number",
        TARGET,
        "same_day_event_count",
    ]

    + TREATMENTS

    + ISOLATED_FLAGS

    + ADJUSTMENT_FEATURES
)


columns = list(
    dict.fromkeys(
        columns
    )
)


print(
    "\nLoading pre-treatment cohort..."
)


df = pd.read_parquet(
    COHORT_FILE,
    columns=columns
)


df[
    "date"
] = pd.to_datetime(
    df[
        "date"
    ],
    errors="coerce"
)


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


print(
    "Rows:",
    f"{len(df):,}"
)


print(
    "Unique HDDs:",
    f"{df['serial_number'].nunique():,}"
)


# ==========================================================
# VALIDATION
# ==========================================================

if df[
    "date"
].isna().any():

    raise RuntimeError(
        "Invalid dates found."
    )


if df[
    [
        TARGET
    ]
    +
    TREATMENTS
].isna().any().any():

    raise RuntimeError(
        "Missing target/treatment values found."
    )


# ==========================================================
# CURRENT ANY-DEGRADATION EVENT
#
# 1 if ANY of the three degradation events occur today.
# ==========================================================

df[
    "current_any_degradation_event"
] = (

    df[
        TREATMENTS
    ]
    .max(
        axis=1
    )
    .astype(
        "int8"
    )
)


# ==========================================================
# PRIOR EVENT COUNTS
#
# We calculate treatment history using only information
# observed BEFORE the current HDD-day.
# ==========================================================

for treatment in TREATMENTS:

    prior_name = (
        f"prior_count_{treatment}"
    )


    cumulative = (
        df
        .groupby(
            "serial_number",
            sort=False
        )[treatment]
        .cumsum()
    )


    df[
        prior_name
    ] = (
        cumulative
        -
        df[
            treatment
        ]
    )


# ==========================================================
# PRIOR ANY-DEGRADATION HISTORY
# ==========================================================

cumulative_any = (
    df
    .groupby(
        "serial_number",
        sort=False
    )[
        "current_any_degradation_event"
    ]
    .cumsum()
)


df[
    "prior_any_degradation_count"
] = (
    cumulative_any
    -
    df[
        "current_any_degradation_event"
    ]
)


# ==========================================================
# FIRST-EVENT FLAGS
# ==========================================================

for treatment in TREATMENTS:

    prior_name = (
        f"prior_count_{treatment}"
    )


    first_name = (
        f"first_{treatment}"
    )


    df[
        first_name
    ] = (
        (
            df[
                treatment
            ] == 1
        )
        &
        (
            df[
                prior_name
            ] == 0
        )
    ).astype(
        "int8"
    )


# ==========================================================
# CLEAN INCIDENT EVENTS
#
# Requirements:
#
# 1. Current treatment occurs.
# 2. It is the first observed occurrence of that treatment.
# 3. No previous degradation event of any of the three
#    treatment types has occurred.
# 4. No other degradation event occurs on the same day.
#
# This gives a much cleaner temporal intervention definition.
# ==========================================================

for treatment, isolated_flag in (
    TREATMENT_MAP.items()
):

    clean_name = (
        f"clean_incident_{treatment}"
    )


    first_name = (
        f"first_{treatment}"
    )


    df[
        clean_name
    ] = (
        (
            df[
                first_name
            ] == 1
        )
        &
        (
            df[
                "prior_any_degradation_count"
            ] == 0
        )
        &
        (
            df[
                isolated_flag
            ] == 1
        )
    ).astype(
        "int8"
    )


# ==========================================================
# AUDIT STORAGE
# ==========================================================

summary_rows = []

monthly_rows = []


# ==========================================================
# BUILD A SEPARATE RISK SET FOR EACH TREATMENT
# ==========================================================

for treatment, isolated_flag in (
    TREATMENT_MAP.items()
):

    print("\n")
    print("=" * 70)

    print(
        f"BUILDING INCIDENT RISK SET: {treatment}"
    )

    print("=" * 70)


    prior_specific = (
        f"prior_count_{treatment}"
    )


    first_specific = (
        f"first_{treatment}"
    )


    clean_incident = (
        f"clean_incident_{treatment}"
    )


    # ======================================================
    # COUNTS BEFORE PRIMARY RESTRICTION
    # ======================================================

    first_specific_mask = (
        df[
            first_specific
        ] == 1
    )


    clean_incident_mask = (
        df[
            clean_incident
        ] == 1
    )


    print(
        "First treatment events:",
        f"{first_specific_mask.sum():,}"
    )


    print(
        "Clean isolated incident events:",
        f"{clean_incident_mask.sum():,}"
    )


    print(
        "Clean incident HDDs:",
        f"{df.loc[clean_incident_mask, 'serial_number'].nunique():,}"
    )


    # ======================================================
    # TREATMENT DATES
    #
    # Controls are drawn from the same calendar dates as
    # incident treatment events.
    # ======================================================

    treatment_dates = (
        df.loc[
            clean_incident_mask,
            "date"
        ]
        .drop_duplicates()
        .sort_values()
    )


    if len(
        treatment_dates
    ) == 0:

        print(
            "No clean incident events. Skipping."
        )

        continue


    date_mask = (
        df[
            "date"
        ].isin(
            treatment_dates
        )
    )


    # ======================================================
    # TREATED
    # ======================================================

    treated_mask = (
        clean_incident_mask
        &
        date_mask
    )


    # ======================================================
    # CONTROL ELIGIBILITY
    #
    # Control HDD-day must:
    #
    # - occur on a treatment event date,
    # - have no degradation event today,
    # - have no previous observed event among the three
    #   degradation types.
    #
    # Thus controls are still "at risk" for their first
    # observed degradation event.
    # ======================================================

    control_mask = (
        date_mask
        &
        (
            df[
                "current_any_degradation_event"
            ] == 0
        )
        &
        (
            df[
                "prior_any_degradation_count"
            ] == 0
        )
    )


    # ======================================================
    # CREATE RISK SET
    # ======================================================

    riskset = df[
        treated_mask
        |
        control_mask
    ].copy()


    riskset[
        "incident_treatment"
    ] = (
        treated_mask.loc[
            riskset.index
        ]
        .astype(
            "int8"
        )
    )


    riskset[
        "treatment_name"
    ] = treatment


    # ======================================================
    # RISK-SET DATE ID
    #
    # Later matching will occur within / near these dates.
    # ======================================================

    riskset[
        "riskset_date"
    ] = riskset[
        "date"
    ]


    # ======================================================
    # FINAL SAFETY CHECK
    # ======================================================

    treated_rows = int(
        (
            riskset[
                "incident_treatment"
            ] == 1
        ).sum()
    )


    control_rows = int(
        (
            riskset[
                "incident_treatment"
            ] == 0
        ).sum()
    )


    if treated_rows == 0:

        raise RuntimeError(
            f"No treated rows in {treatment} risk set."
        )


    if control_rows == 0:

        raise RuntimeError(
            f"No controls in {treatment} risk set."
        )


    # ======================================================
    # OUTCOME COUNTS
    # ======================================================

    treated_positive = int(
        (
            (
                riskset[
                    "incident_treatment"
                ] == 1
            )
            &
            (
                riskset[
                    TARGET
                ] == 1
            )
        ).sum()
    )


    control_positive = int(
        (
            (
                riskset[
                    "incident_treatment"
                ] == 0
            )
            &
            (
                riskset[
                    TARGET
                ] == 1
            )
        ).sum()
    )


    treated_rate = (
        treated_positive
        / treated_rows
    )


    control_rate = (
        control_positive
        / control_rows
    )


    # ======================================================
    # SAVE RISK SET
    # ======================================================

    OUTPUT_COLUMNS = (

        [
            "date",

            "riskset_date",

            "serial_number",

            TARGET,

            "treatment_name",

            "incident_treatment",

            treatment,

            isolated_flag,

            "same_day_event_count",

            "prior_any_degradation_count",

            prior_specific,
        ]

        + ADJUSTMENT_FEATURES
    )


    OUTPUT_COLUMNS = list(
        dict.fromkeys(
            OUTPUT_COLUMNS
        )
    )


    output_file = (
        OUTPUT_DIR
        / f"{treatment}_incident_riskset.parquet"
    )


    riskset[
        OUTPUT_COLUMNS
    ].to_parquet(
        output_file,
        index=False,
        compression="zstd"
    )


    # ======================================================
    # SUMMARY
    # ======================================================

    summary_rows.append(
        {
            "treatment":
                treatment,

            "all_first_specific_events":
                int(
                    first_specific_mask.sum()
                ),

            "all_first_specific_hdds":
                df.loc[
                    first_specific_mask,
                    "serial_number"
                ].nunique(),

            "clean_incident_events":
                treated_rows,

            "clean_incident_hdds":
                riskset.loc[
                    riskset[
                        "incident_treatment"
                    ] == 1,
                    "serial_number"
                ].nunique(),

            "riskset_event_dates":
                len(
                    treatment_dates
                ),

            "eligible_control_rows":
                control_rows,

            "eligible_control_hdds":
                riskset.loc[
                    riskset[
                        "incident_treatment"
                    ] == 0,
                    "serial_number"
                ].nunique(),

            "treated_positive_rows":
                treated_positive,

            "control_positive_rows":
                control_positive,

            "treated_failure_rate":
                treated_rate,

            "control_failure_rate":
                control_rate,

            "crude_risk_difference":
                treated_rate
                -
                control_rate,

            "crude_risk_ratio":
                (
                    treated_rate
                    / control_rate
                    if control_rate > 0
                    else np.nan
                ),
        }
    )


    # ======================================================
    # MONTHLY SUMMARY
    # ======================================================

    riskset[
        "month"
    ] = (
        riskset[
            "date"
        ]
        .dt
        .to_period(
            "M"
        )
        .astype(
            str
        )
    )


    for month, month_df in (
        riskset.groupby(
            "month"
        )
    ):

        treated_month = (
            month_df[
                "incident_treatment"
            ] == 1
        )


        control_month = (
            month_df[
                "incident_treatment"
            ] == 0
        )


        monthly_rows.append(
            {
                "treatment":
                    treatment,

                "month":
                    month,

                "treated_rows":
                    int(
                        treated_month.sum()
                    ),

                "treated_hdds":
                    month_df.loc[
                        treated_month,
                        "serial_number"
                    ].nunique(),

                "control_rows":
                    int(
                        control_month.sum()
                    ),

                "control_hdds":
                    month_df.loc[
                        control_month,
                        "serial_number"
                    ].nunique(),

                "treated_positive_rows":
                    int(
                        (
                            treated_month
                            &
                            (
                                month_df[
                                    TARGET
                                ] == 1
                            )
                        ).sum()
                    ),
            }
        )


    print(
        "Risk-set treated rows:",
        f"{treated_rows:,}"
    )


    print(
        "Risk-set control rows:",
        f"{control_rows:,}"
    )


    print(
        "Treatment event dates:",
        f"{len(treatment_dates):,}"
    )


    print(
        "Treated failure rate:",
        f"{treated_rate:.6f}"
    )


    print(
        "Control failure rate:",
        f"{control_rate:.6f}"
    )


    print(
        "Saved:",
        output_file
    )


# ==========================================================
# SAVE SUMMARY FILES
# ==========================================================

summary_df = pd.DataFrame(
    summary_rows
)


monthly_df = pd.DataFrame(
    monthly_rows
)


summary_file = (
    AUDIT_DIR
    / "incident_riskset_summary.csv"
)


monthly_file = (
    AUDIT_DIR
    / "incident_riskset_monthly_summary.csv"
)


summary_df.to_csv(
    summary_file,
    index=False
)


monthly_df.to_csv(
    monthly_file,
    index=False
)


# ==========================================================
# PRINT FINAL RESULTS
# ==========================================================

print("\n")
print("=" * 70)

print(
    "INCIDENT TREATMENT RISK-SET PREPARATION COMPLETE"
)

print("=" * 70)


print(
    "\nSummary:\n"
)


print(
    summary_df.to_string(
        index=False
    )
)


print(
    "\nMonthly summary:\n"
)


print(
    monthly_df.to_string(
        index=False
    )
)


print(
    "\nRisk-set datasets saved to:"
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